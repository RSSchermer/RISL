//! Removes switch branches that contain probable UB.
//!
//! For the time being, the only UB we look for is dereferencing a fallback (null) pointer.
//!
//! If removing such branches from a switch node would leave the switch node with only a single
//! branch, that branch is inlined into the switch node's outer region and the switch node
//! itself is removed (see [inline_switch_node]). Otherwise, the branches are removed and the
//! switch node's branch-selector converter is adjusted (see [retain_switch_branches]).

use std::collections::VecDeque;

use rustc_hash::FxHashSet;
use smallbitvec::SmallBitVec;

use crate::Module;
use crate::rvsdg::transform::region_replication::inline_switch_branch;
use crate::rvsdg::transform::switch_branch_pruning::retain_switch_branches;
use crate::rvsdg::{Connectivity, Node, NodeKind, Region, Rvsdg, SimpleNode, ValueOrigin};

struct Job {
    switch_node: Node,
    retained_branches: SmallBitVec,
}

struct JobCollector<'a> {
    jobs: &'a mut VecDeque<Job>,
    fallback_pointer_origin_finder: &'a mut FallbackPointerOriginFinder,
}

impl JobCollector<'_> {
    /// Returns `true` if the `region` contains at least one of the kinds of UB we currently
    /// identify.
    fn collect(&mut self, rvsdg: &Rvsdg, region: Region) -> bool {
        self.jobs.clear();
        self.visit_region(rvsdg, region)
    }

    /// Returns `true` if the `region` contains at least one of the kinds of UB we currently
    /// identify.
    fn visit_region(&mut self, rvsdg: &Rvsdg, region: Region) -> bool {
        for node in rvsdg[region].nodes() {
            if self.visit_node(rvsdg, region, *node) {
                return true;
            }
        }

        false
    }

    /// Returns `true` if reaching the node results in at least one of the kinds of UB we currently
    /// identify.
    fn visit_node(&mut self, rvsdg: &Rvsdg, region: Region, node: Node) -> bool {
        use NodeKind::*;
        use SimpleNode::*;

        match rvsdg[node].kind() {
            Switch(_) => self.visit_switch_node(rvsdg, node),
            Loop(loop_data) => self.visit_region(rvsdg, loop_data.loop_region()),
            Simple(OpLoad(_)) | Simple(OpStore(_)) => {
                // Dereferencing a fallback pointer is UB.

                let ptr_origin = rvsdg[node].value_inputs()[0].origin;

                self.fallback_pointer_origin_finder
                    .trace(rvsdg, region, ptr_origin)
            }
            _ => false,
        }
    }

    /// Returns `true` if all branches of the switch node contain at least one of the kinds of UB we
    /// currently identify.
    fn visit_switch_node(&mut self, rvsdg: &Rvsdg, switch_node: Node) -> bool {
        let switch_data = rvsdg[switch_node].expect_switch();
        let branch_count = switch_data.branches().len();

        let mut retained_branches = SmallBitVec::with_capacity(branch_count);

        for branch in switch_data.branches() {
            let prior_job_count = self.jobs.len();

            if !self.visit_region(rvsdg, *branch) {
                retained_branches.push(true);
            } else {
                // The branch won't be retained, but we may have recorded jobs for nested switch
                // nodes. We'll truncate the job list back to the prior job count to discard those.
                self.jobs.truncate(prior_job_count);

                retained_branches.push(false);
            }
        }

        if retained_branches.all_false() {
            true
        } else {
            if !retained_branches.all_true() {
                self.jobs.push_back(Job {
                    switch_node,
                    retained_branches,
                });
            }

            false
        }
    }
}

struct FallbackPointerOriginFinder {
    visited: FxHashSet<(Region, ValueOrigin)>,
}

impl FallbackPointerOriginFinder {
    fn new() -> Self {
        Self {
            visited: FxHashSet::default(),
        }
    }

    fn trace(&mut self, rvsdg: &Rvsdg, region: Region, origin: ValueOrigin) -> bool {
        self.visited.clear();
        self.trace_recursive(rvsdg, region, origin)
    }

    fn trace_recursive(&mut self, rvsdg: &Rvsdg, region: Region, origin: ValueOrigin) -> bool {
        if !self.visited.insert((region, origin)) {
            return false;
        }

        match origin {
            ValueOrigin::Output { producer, .. } => {
                let producer_data = &rvsdg[producer];
                if let NodeKind::Simple(simple_node) = producer_data.kind() {
                    match simple_node {
                        SimpleNode::ConstFallback(_) => return true,
                        SimpleNode::ValueProxy(value_proxy) => {
                            return self.trace_recursive(
                                rvsdg,
                                producer_data.region(),
                                value_proxy.value_inputs()[0].origin,
                            );
                        }
                        _ => {}
                    }
                }
                false
            }
            ValueOrigin::Argument(arg) => {
                if region == rvsdg.global_region() {
                    return false;
                }

                let owner_node = rvsdg[region].owner();
                let owner_data = &rvsdg[owner_node];
                let outer_region = owner_data.region();

                match owner_data.kind() {
                    NodeKind::Switch(switch_node) => {
                        let input = arg + 1;

                        self.trace_recursive(
                            rvsdg,
                            outer_region,
                            switch_node.value_inputs()[input as usize].origin,
                        )
                    }
                    NodeKind::Loop(loop_node) => {
                        // Argument i is a loop-constant if the loop-region value result i + 1
                        // has an origin that is the same region-argument i.
                        let loop_region = loop_node.loop_region();
                        let res = arg + 1;
                        let result_origin = rvsdg[loop_region].value_results()[res as usize].origin;

                        // Only trace out of a loop region if the loop-value is a loop-constant,
                        // otherwise assume the pointer is valid.
                        if let ValueOrigin::Argument(res_arg) = result_origin
                            && res_arg == arg
                        {
                            self.trace_recursive(
                                rvsdg,
                                outer_region,
                                loop_node.value_inputs()[arg as usize].origin,
                            )
                        } else {
                            false
                        }
                    }
                    _ => false,
                }
            }
        }
    }
}

fn apply_elimination(
    module: &mut Module,
    rvsdg: &mut Rvsdg,
    switch_node: Node,
    valid_branches: &[usize],
) {
    assert!(!valid_branches.is_empty());

    let data = rvsdg[switch_node].expect_switch();
    let branch_count = data.branches().len();

    if valid_branches.len() == branch_count {
        return;
    }

    if valid_branches.len() == 1 {
        // There's only a single branch left: inline it.
        inline_switch_branch(module, rvsdg, switch_node, valid_branches[0]);
    } else {
        retain_switch_branches(rvsdg, switch_node, valid_branches);
    }
}

pub struct ConditionalUbEliminator {
    queue: VecDeque<Job>,
    fallback_pointer_origin_finder: FallbackPointerOriginFinder,
    retained_branches_buffer: Vec<usize>,
}

impl ConditionalUbEliminator {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            fallback_pointer_origin_finder: FallbackPointerOriginFinder::new(),
            retained_branches_buffer: vec![],
        }
    }

    /// Removes switch branches that contain probable UB, for switch nodes in the `region`.
    ///
    /// See the [module-level documentation](crate::rvsdg::transform::conditional_ub_elimination)
    /// for details.
    ///
    /// Note that this does not just apply to direct child nodes of the `region`; nested regions
    /// are also processed, recursively.
    ///
    /// Return `true` if the region contains unconditional UB (UB that cannot be eliminated by
    /// pruning branches).
    pub fn transform_region(
        &mut self,
        module: &mut Module,
        rvsdg: &mut Rvsdg,
        region: Region,
    ) -> bool {
        let mut job_collector = JobCollector {
            jobs: &mut self.queue,
            fallback_pointer_origin_finder: &mut self.fallback_pointer_origin_finder,
        };

        if job_collector.collect(rvsdg, region) {
            return true;
        }

        // Process jobs from the inside out. Inlining a switch node as a single branch reason does
        // not move the pre-existing nodes but rather creates copies. Processing from the outside
        // in would therefore risk running jobs on switch nodes that no longer exist.
        while let Some(job) = self.queue.pop_front() {
            self.retained_branches_buffer.clear();

            for (branch_index, retained) in job.retained_branches.iter().enumerate() {
                if retained {
                    self.retained_branches_buffer.push(branch_index);
                }
            }

            apply_elimination(
                module,
                rvsdg,
                job.switch_node,
                &self.retained_branches_buffer,
            )
        }

        false
    }
}

pub fn transform_entry_points(module: &mut Module, rvsdg: &mut Rvsdg) {
    let mut eliminator = ConditionalUbEliminator::new();
    let entry_points = module.entry_points.iter().map(|e| e.0).collect::<Vec<_>>();

    for entry_point in entry_points {
        let fn_node = rvsdg
            .get_function_node(entry_point)
            .expect("function not registered");
        let body_region = rvsdg[fn_node].expect_function().body_region();

        if eliminator.transform_region(module, rvsdg, body_region) {
            panic!("encountered unconditional UB in entry point function");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::rvsdg::{StateOrigin, ValueInput, ValueOutput};
    use crate::ty::{Int, TY_BOOL, TY_DUMMY, TY_PREDICATE, TY_PTR_U32, TY_U32};
    use crate::{BranchCase, FnSig, Function, Symbol};

    #[test]
    fn test_conditional_ub_elimination_bool() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref(""),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Symbol::from_ref(""),
                ty: TY_DUMMY,
                args: vec![],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, body) = rvsdg.register_function(&module, function, iter::empty());

        let bool_val = rvsdg.add_const_bool(body, true);
        let selector =
            rvsdg.add_op_bool_to_branch_selector(body, ValueInput::output(TY_BOOL, bool_val, 0));

        let switch = rvsdg.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, selector, 0)],
            vec![ValueOutput::new(TY_U32)],
            Some(StateOrigin::Argument),
        );

        let branch_0 = rvsdg.add_switch_branch(switch);
        let branch_0_value = rvsdg.add_const_u32(branch_0, 42);
        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: branch_0_value,
                output: 0,
            },
        );

        let branch_1 = rvsdg.add_switch_branch(switch);
        let ptr_fallback = rvsdg.add_const_fallback(branch_1, TY_PTR_U32);
        let load = rvsdg.add_op_load(
            branch_1,
            ValueInput::output(TY_PTR_U32, ptr_fallback, 0),
            StateOrigin::Argument,
        );
        rvsdg.reconnect_region_result(
            branch_1,
            0,
            ValueOrigin::Output {
                producer: load,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(
            body,
            0,
            ValueOrigin::Output {
                producer: switch,
                output: 0,
            },
        );

        let mut eliminator = ConditionalUbEliminator::new();

        eliminator.transform_region(&mut module, &mut rvsdg, body);

        let ValueOrigin::Output {
            producer,
            output: 0,
        } = rvsdg[body].value_results()[0].origin
        else {
            panic!("expected the function result to connect the first output of a node");
        };

        let val = rvsdg[producer].expect_const_u32();

        assert_eq!(val.value(), 42);
        assert!(!rvsdg.is_live_node(switch));
    }

    #[test]
    fn test_conditional_ub_elimination_case() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref(""),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Symbol::from_ref(""),
                ty: TY_DUMMY,
                args: vec![],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, body) = rvsdg.register_function(&module, function, iter::empty());

        let val = rvsdg.add_const_u32(body, 0);
        let selector = rvsdg.add_op_case_to_branch_selector(
            body,
            ValueInput::output(TY_U32, val, 0),
            Int::U32,
            [2u32, 0, 1].map(BranchCase::from),
        );

        let switch = rvsdg.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, selector, 0)],
            vec![ValueOutput::new(TY_U32)],
            Some(StateOrigin::Argument),
        );

        // Branch 0: Valid
        let branch_0 = rvsdg.add_switch_branch(switch);
        let branch_0_value = rvsdg.add_const_u32(branch_0, 0);
        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: branch_0_value,
                output: 0,
            },
        );

        // Branch 1: UB
        let branch_1 = rvsdg.add_switch_branch(switch);
        let ptr_fallback_1 = rvsdg.add_const_fallback(branch_1, TY_PTR_U32);
        let load_1 = rvsdg.add_op_load(
            branch_1,
            ValueInput::output(TY_PTR_U32, ptr_fallback_1, 0),
            StateOrigin::Argument,
        );
        rvsdg.reconnect_region_result(
            branch_1,
            0,
            ValueOrigin::Output {
                producer: load_1,
                output: 0,
            },
        );

        // Branch 2: Valid
        let branch_2 = rvsdg.add_switch_branch(switch);
        let branch_2_value = rvsdg.add_const_u32(branch_2, 2);
        rvsdg.reconnect_region_result(
            branch_2,
            0,
            ValueOrigin::Output {
                producer: branch_2_value,
                output: 0,
            },
        );

        // Branch 3: UB
        let branch_3 = rvsdg.add_switch_branch(switch);
        let ptr_fallback_3 = rvsdg.add_const_fallback(branch_3, TY_PTR_U32);
        let load_3 = rvsdg.add_op_load(
            branch_3,
            ValueInput::output(TY_PTR_U32, ptr_fallback_3, 0),
            StateOrigin::Argument,
        );
        rvsdg.reconnect_region_result(
            branch_3,
            0,
            ValueOrigin::Output {
                producer: load_3,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(
            body,
            0,
            ValueOrigin::Output {
                producer: switch,
                output: 0,
            },
        );

        let mut eliminator = ConditionalUbEliminator::new();

        eliminator.transform_region(&mut module, &mut rvsdg, body);

        // Verify that switch still exists but has 2 branches
        assert!(rvsdg.is_live_node(switch));
        let switch_node = rvsdg[switch].expect_switch();
        assert_eq!(switch_node.branches(), &[branch_0, branch_2]);

        // Verify that the selector was updated correctly
        let ValueOrigin::Output {
            producer: selector, ..
        } = switch_node.branch_selector().origin
        else {
            panic!("expected branch selector to be an output")
        };
        let cases = rvsdg[selector].expect_op_case_to_branch_selector().cases();
        assert_eq!(cases, &[BranchCase::from(2u32)]);
    }

    #[test]
    fn test_conditional_ub_elimination_loop_constant() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref(""),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Symbol::from_ref(""),
                ty: TY_DUMMY,
                args: vec![],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, body) = rvsdg.register_function(&module, function, iter::empty());

        let ptr_fallback = rvsdg.add_const_fallback(body, TY_PTR_U32);
        let dummy_input = rvsdg.add_const_u32(body, 0);

        let (loop_node, loop_region) = rvsdg.add_loop(
            body,
            vec![
                ValueInput::output(TY_PTR_U32, ptr_fallback, 0),
                ValueInput::output(TY_U32, dummy_input, 0),
            ],
            Some(StateOrigin::Argument),
        );

        let bool_val = rvsdg.add_const_bool(loop_region, true);
        let selector = rvsdg
            .add_op_bool_to_branch_selector(loop_region, ValueInput::output(TY_BOOL, bool_val, 0));

        let switch = rvsdg.add_switch(
            loop_region,
            vec![
                ValueInput::output(TY_PREDICATE, selector, 0),
                ValueInput::argument(TY_PTR_U32, 0),
            ],
            vec![ValueOutput::new(TY_U32)],
            Some(StateOrigin::Argument),
        );

        // Branch 0: contains UB
        let branch_0 = rvsdg.add_switch_branch(switch);
        let load = rvsdg.add_op_load(
            branch_0,
            ValueInput::argument(TY_PTR_U32, 0),
            StateOrigin::Argument,
        );
        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: load,
                output: 0,
            },
        );

        // Branch 1: valid
        let branch_1 = rvsdg.add_switch_branch(switch);
        let branch_1_value = rvsdg.add_const_u32(branch_1, 42);
        rvsdg.reconnect_region_result(
            branch_1,
            0,
            ValueOrigin::Output {
                producer: branch_1_value,
                output: 0,
            },
        );

        let reentry_condition = rvsdg.add_const_bool(loop_region, false);

        // Connect the loop region results.
        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: reentry_condition,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(loop_region, 1, ValueOrigin::Argument(0));
        rvsdg.reconnect_region_result(
            loop_region,
            2,
            ValueOrigin::Output {
                producer: switch,
                output: 0,
            },
        );

        // Connect the function body result
        rvsdg.reconnect_region_result(
            body,
            0,
            ValueOrigin::Output {
                producer: loop_node,
                output: 1,
            },
        );

        let mut eliminator = ConditionalUbEliminator::new();

        eliminator.transform_region(&mut module, &mut rvsdg, body);

        // Verify that switch in loop was inlined (since only branch 1 remains)
        assert!(!rvsdg.is_live_node(switch));

        // The loop result should now connect to branch_1_value (replicated)
        let loop_region_data = &rvsdg[loop_region];
        let ValueOrigin::Output { producer, .. } = loop_region_data.value_results()[2].origin
        else {
            panic!(
                "expected result 2 to be an output, but it was {:?}",
                loop_region_data.value_results()[2].origin
            );
        };
        // producer should be the replicated const_u32(42)
        let val = rvsdg[producer].expect_const_u32();
        assert_eq!(val.value(), 42);
    }

    #[test]
    fn test_conditional_ub_elimination_nested_all_ub() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref(""),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Symbol::from_ref(""),
                ty: TY_DUMMY,
                args: vec![],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, body) = rvsdg.register_function(&module, function, iter::empty());

        let outer_condition = rvsdg.add_const_bool(body, true);
        let outer_selector = rvsdg
            .add_op_bool_to_branch_selector(body, ValueInput::output(TY_BOOL, outer_condition, 0));
        let outer_switch = rvsdg.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, outer_selector, 0)],
            vec![ValueOutput::new(TY_U32)],
            Some(StateOrigin::Argument),
        );

        let outer_branch_0 = rvsdg.add_switch_branch(outer_switch);
        let outer_branch_0_value = rvsdg.add_const_u32(outer_branch_0, 42);
        rvsdg.reconnect_region_result(
            outer_branch_0,
            0,
            ValueOrigin::Output {
                producer: outer_branch_0_value,
                output: 0,
            },
        );

        let outer_branch_1 = rvsdg.add_switch_branch(outer_switch);
        let inner_condition = rvsdg.add_const_bool(outer_branch_1, true);
        let inner_selector = rvsdg.add_op_bool_to_branch_selector(
            outer_branch_1,
            ValueInput::output(TY_BOOL, inner_condition, 0),
        );
        let inner_switch = rvsdg.add_switch(
            outer_branch_1,
            vec![ValueInput::output(TY_PREDICATE, inner_selector, 0)],
            vec![ValueOutput::new(TY_U32)],
            Some(StateOrigin::Argument),
        );

        // Add two branches that both contain a load from a fallback pointer.
        for _ in 0..2 {
            let inner_branch = rvsdg.add_switch_branch(inner_switch);
            let ptr_fallback = rvsdg.add_const_fallback(inner_branch, TY_PTR_U32);
            let load = rvsdg.add_op_load(
                inner_branch,
                ValueInput::output(TY_PTR_U32, ptr_fallback, 0),
                StateOrigin::Argument,
            );
            rvsdg.reconnect_region_result(
                inner_branch,
                0,
                ValueOrigin::Output {
                    producer: load,
                    output: 0,
                },
            );
        }

        rvsdg.reconnect_region_result(
            outer_branch_1,
            0,
            ValueOrigin::Output {
                producer: inner_switch,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(
            body,
            0,
            ValueOrigin::Output {
                producer: outer_switch,
                output: 0,
            },
        );

        let mut eliminator = ConditionalUbEliminator::new();

        assert!(!eliminator.transform_region(&mut module, &mut rvsdg, body));

        assert!(!rvsdg.is_live_node(outer_switch));
        assert!(!rvsdg.is_live_node(inner_switch));

        let ValueOrigin::Output {
            producer,
            output: 0,
        } = rvsdg[body].value_results()[0].origin
        else {
            panic!("expected the function result to connect the first output of a node");
        };
        assert_eq!(rvsdg[producer].expect_const_u32().value(), 42);
        assert_eq!(*rvsdg[body].state_result(), StateOrigin::Argument);
    }
}
