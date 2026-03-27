use rustc_hash::FxHashSet;

use crate::Module;
use crate::rvsdg::transform::region_replication::{inline_switch_branch, replicate_region};
use crate::rvsdg::visit::region_nodes::RegionNodesVisitor;
use crate::rvsdg::{Connectivity, Node, NodeKind, Region, Rvsdg, SimpleNode, ValueOrigin, visit};

pub struct ConditionalUbEliminator {
    switch_collector: SwitchCollector,
    fallback_pointer_origin_finder: FallbackPointerOriginFinder,
    valid_branch_buffer: Vec<usize>,
}

impl ConditionalUbEliminator {
    pub fn new() -> Self {
        Self {
            switch_collector: SwitchCollector::new(),
            fallback_pointer_origin_finder: FallbackPointerOriginFinder::new(),
            valid_branch_buffer: vec![],
        }
    }

    pub fn transform_region(&mut self, module: &mut Module, rvsdg: &mut Rvsdg, region: Region) {
        self.switch_collector.collect(rvsdg, region);

        while let Some(node) = self.switch_collector.next() {
            self.valid_branch_buffer.clear();

            let data = rvsdg[node].expect_switch();

            for (i, branch) in data.branches().iter().enumerate() {
                if !self.branch_contains_ub(rvsdg, *branch) {
                    self.valid_branch_buffer.push(i);
                }
            }

            apply_elimination(module, rvsdg, node, &self.valid_branch_buffer)
        }
    }

    fn branch_contains_ub(&mut self, rvsdg: &Rvsdg, branch: Region) -> bool {
        // For now, we only test for one type of UB: memory operations that operate on invalid
        // pointers.

        self.branch_contains_invalid_mem_op(rvsdg, branch)
    }

    fn branch_contains_invalid_mem_op(&mut self, rvsdg: &Rvsdg, branch: Region) -> bool {
        // Find all memory operations in the branch and check if their pointer input is invalid.
        // We consider pointer input invalid if it traces back to a `ConstFallback` node.

        for node in rvsdg[branch].nodes() {
            match rvsdg[*node].kind() {
                NodeKind::Simple(SimpleNode::OpLoad(_))
                | NodeKind::Simple(SimpleNode::OpStore(_)) => {
                    let ptr_origin = rvsdg[*node].value_inputs()[0].origin;

                    if self
                        .fallback_pointer_origin_finder
                        .trace(rvsdg, branch, ptr_origin)
                    {
                        return true;
                    }
                }
                _ => {}
            }
        }

        false
    }
}

struct SwitchCollector {
    candidate_stack: Vec<Node>,
}

impl SwitchCollector {
    fn new() -> Self {
        Self {
            candidate_stack: Vec::new(),
        }
    }

    fn collect(&mut self, rvsdg: &Rvsdg, region: Region) {
        self.candidate_stack.clear();
        self.visit_region(rvsdg, region);
    }

    fn next(&mut self) -> Option<Node> {
        // Last in, first out: ensures we visit inner switch nodes before outer switch nodes. If we
        // in the future decide to mark entire switch nodes as UB (all branches exhibit UB), this
        // processing order ensures we'll only need a single pass.
        self.candidate_stack.pop()
    }
}

impl RegionNodesVisitor for SwitchCollector {
    fn visit_node(&mut self, rvsdg: &Rvsdg, node: Node) {
        if let NodeKind::Switch(_) = rvsdg[node].kind() {
            self.candidate_stack.push(node);
        }

        visit::region_nodes::visit_node(self, rvsdg, node);
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
    let data = rvsdg[switch_node].expect_switch();
    let branch_count = data.branches().len();

    if valid_branches.len() == branch_count {
        return;
    }

    if valid_branches.is_empty() {
        panic!("a switch is expected to have at least one valid branch");
    }

    if valid_branches.len() == 1 {
        // There's only a single branch left: inline it.
        inline_switch_branch(module, rvsdg, switch_node, valid_branches[0]);
    } else {
        // Multiple branches remain: remove the invalid branches and adjust the branch-selector.

        let region = rvsdg[switch_node].region();
        let selector_origin = rvsdg[switch_node].expect_switch().branch_selector().origin;

        let new_selector_origin = if let ValueOrigin::Output {
            producer,
            output: 0,
        } = selector_origin
        {
            match rvsdg[producer].kind() {
                NodeKind::Simple(SimpleNode::OpCaseToBranchSelector(n)) => {
                    let mut new_cases = Vec::new();

                    for i in valid_branches {
                        if let Some(case) = n.cases().get(*i) {
                            new_cases.push(*case);
                        }
                    }

                    if new_cases.len() == valid_branches.len() {
                        // There should be one less case than there are branches
                        new_cases.pop();
                    }

                    let new_node = rvsdg.add_op_case_to_branch_selector(
                        region,
                        rvsdg[producer].value_inputs()[0],
                        new_cases,
                    );

                    ValueOrigin::Output {
                        producer: new_node,
                        output: 0,
                    }
                }
                NodeKind::Simple(SimpleNode::OpU32ToBranchSelector(_)) => {
                    let mut new_cases = Vec::new();

                    for i in valid_branches {
                        new_cases.push(*i as u32);
                    }

                    // There should be one less case than there are branches
                    new_cases.pop();

                    let new_node = rvsdg.add_op_case_to_branch_selector(
                        region,
                        rvsdg[producer].value_inputs()[0],
                        new_cases,
                    );

                    ValueOrigin::Output {
                        producer: new_node,
                        output: 0,
                    }
                }
                NodeKind::Simple(SimpleNode::OpBoolToBranchSelector(_)) => {
                    panic!("should have had only one valid branch left")
                }
                _ => panic!(
                    "a switch node's branch-selector input should connect to a node-kind that \
                    produces a branch selector"
                ),
            }
        } else {
            panic!(
                "RVSDG should be in predicate-continuation form before applying this \
                transformation"
            );
        };

        // Keep only the valid branches, discard the invalid ones
        rvsdg.permute_switch_branches(switch_node, &valid_branches);

        // Remap the selector
        rvsdg.reconnect_value_input(switch_node, 0, new_selector_origin);
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

        eliminator.transform_region(module, rvsdg, body_region);
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::rvsdg::{StateOrigin, ValueInput, ValueOutput};
    use crate::ty::{TY_BOOL, TY_DUMMY, TY_PREDICATE, TY_PTR_U32, TY_U32};
    use crate::{FnSig, Function, Symbol};

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
            vec![2, 0, 1],
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
        assert_eq!(cases, &[2]);
    }

    #[test]
    fn test_conditional_ub_elimination_u32_single_ub() {
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
        let selector =
            rvsdg.add_op_u32_to_branch_selector(body, ValueInput::output(TY_U32, val, 0));

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

        // Branch 1: Valid
        let branch_1 = rvsdg.add_switch_branch(switch);
        let branch_1_value = rvsdg.add_const_u32(branch_1, 1);
        rvsdg.reconnect_region_result(
            branch_1,
            0,
            ValueOrigin::Output {
                producer: branch_1_value,
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
        let ptr_fallback = rvsdg.add_const_fallback(branch_3, TY_PTR_U32);
        let load = rvsdg.add_op_load(
            branch_3,
            ValueInput::output(TY_PTR_U32, ptr_fallback, 0),
            StateOrigin::Argument,
        );
        rvsdg.reconnect_region_result(
            branch_3,
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

        // Verify that switch still exists but has 3 branches
        assert!(rvsdg.is_live_node(switch));
        let switch_node = rvsdg[switch].expect_switch();
        assert_eq!(switch_node.branches(), &[branch_0, branch_1, branch_2]);

        // Verify that the selector was updated correctly
        let ValueOrigin::Output {
            producer: selector, ..
        } = switch_node.branch_selector().origin
        else {
            panic!("expected branch selector to be an output")
        };
        let cases = rvsdg[selector].expect_op_case_to_branch_selector().cases();
        assert_eq!(cases, &[0, 1]);
    }

    #[test]
    fn test_conditional_ub_elimination_u32_multiple_ub() {
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
        let selector =
            rvsdg.add_op_u32_to_branch_selector(body, ValueInput::output(TY_U32, val, 0));

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
        assert_eq!(cases, &[0]);
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
}
