use crate::Module;
use crate::rvsdg::analyse::region_identity::RegionIdentityChecker;
use crate::rvsdg::transform::region_replication::inline_switch_branch;
use crate::rvsdg::visit::region_nodes::RegionNodesVisitor;
use crate::rvsdg::{Node, NodeKind, Region, Rvsdg, visit};

struct JobCollector<'a> {
    checker: &'a mut RegionIdentityChecker,
    job_stack: &'a mut Vec<Node>,
}

impl RegionNodesVisitor for JobCollector<'_> {
    fn visit_node(&mut self, rvsdg: &Rvsdg, node: Node) {
        use NodeKind::*;

        match rvsdg[node].kind() {
            Switch(switch) => {
                let branches = switch.branches();

                if !branches.is_empty() {
                    let first_branch = branches[0];
                    let all_identical = branches[1..]
                        .iter()
                        .all(|&branch| self.checker.compare_regions(rvsdg, first_branch, branch));

                    if all_identical {
                        self.job_stack.push(node);

                        // If all branches are identical, we only continue looking for jobs in the
                        // first branch, as all other branches will be discarded anyway.

                        visit::region_nodes::visit_region(self, rvsdg, first_branch);

                        return;
                    }
                }
            }
            _ => (),
        }

        visit::region_nodes::visit_node(self, rvsdg, node);
    }
}

pub struct IdenticalBranchElimination {
    // We visit nodes from the "outside in", but we want to inline nodes from the "inside out"
    // (because if we inline from the outside in, inlining a node that contains more jobs will
    // invalidate our job list). Therefore, we pop from a stack to reverse the order.
    job_stack: Vec<Node>,
    checker: RegionIdentityChecker,
}

impl IdenticalBranchElimination {
    pub fn new() -> Self {
        Self {
            job_stack: Vec::new(),
            checker: RegionIdentityChecker::new(),
        }
    }

    pub fn inline_in_region(&mut self, module: &mut Module, rvsdg: &mut Rvsdg, region: Region) {
        self.job_stack.clear();

        JobCollector {
            checker: &mut self.checker,
            job_stack: &mut self.job_stack,
        }
        .visit_region(rvsdg, region);

        while let Some(switch_node) = self.job_stack.pop() {
            inline_switch_branch(module, rvsdg, switch_node, 0);
        }
    }
}

/// A transformation pass that eliminates `Switch` nodes where all branches are structurally
/// identical.
pub fn transform_entry_points(module: &mut Module, rvsdg: &mut Rvsdg) {
    let mut transformer = IdenticalBranchElimination::new();

    let entry_points = module
        .entry_points
        .iter()
        .map(|(f, _)| f)
        .collect::<Vec<_>>();

    for entry_point in entry_points {
        let fn_node = rvsdg
            .get_function_node(entry_point)
            .expect("Function must exist in RVSDG");
        let body_region = rvsdg[fn_node].expect_function().body_region();

        transformer.inline_in_region(module, rvsdg, body_region);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rvsdg::{Connectivity, ValueInput, ValueOrigin, ValueOutput};
    use crate::ty::{TY_DUMMY, TY_PREDICATE, TY_U32};
    use crate::{BinaryOperator, FnArg, FnSig, Function, Symbol};

    #[test]
    fn test_nested_identical_switches_elimination() {
        let module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let mut module = module;

        let function = Function {
            name: Symbol::from_ref("f"),
            module: Symbol::from_ref("m"),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
                ty: TY_DUMMY,
                args: vec![
                    FnArg {
                        ty: TY_U32,
                        shader_io_binding: None,
                    },
                    FnArg {
                        ty: TY_U32,
                        shader_io_binding: None,
                    },
                    FnArg {
                        ty: TY_PREDICATE,
                        shader_io_binding: None,
                    },
                    FnArg {
                        ty: TY_PREDICATE,
                        shader_io_binding: None,
                    },
                ],
                ret_ty: Some(TY_U32),
            },
        );

        let (_, region) = rvsdg.register_function(&module, function, Vec::new());

        // Outer switch node
        let outer_switch = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 2),
                ValueInput::argument(TY_U32, 0),
                ValueInput::argument(TY_U32, 1),
                ValueInput::argument(TY_PREDICATE, 3),
            ],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        for _ in 0..2 {
            let outer_branch = rvsdg.add_switch_branch(outer_switch);

            // Inner switch node
            let inner_switch = rvsdg.add_switch(
                outer_branch,
                vec![
                    ValueInput::argument(TY_PREDICATE, 2),
                    ValueInput::argument(TY_U32, 0),
                    ValueInput::argument(TY_U32, 1),
                ],
                vec![ValueOutput::new(TY_U32)],
                None,
            );

            for _ in 0..2 {
                let inner_branch = rvsdg.add_switch_branch(inner_switch);

                let add = rvsdg.add_op_binary(
                    inner_branch,
                    BinaryOperator::Add,
                    ValueInput::argument(TY_U32, 0),
                    ValueInput::argument(TY_U32, 1),
                );

                rvsdg.reconnect_region_result(
                    inner_branch,
                    0,
                    ValueOrigin::Output {
                        producer: add,
                        output: 0,
                    },
                );
            }

            rvsdg.reconnect_region_result(
                outer_branch,
                0,
                ValueOrigin::Output {
                    producer: inner_switch,
                    output: 0,
                },
            );
        }

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: outer_switch,
                output: 0,
            },
        );

        let mut transformer = IdenticalBranchElimination::new();

        transformer.inline_in_region(&mut module, &mut rvsdg, region);

        let res = &rvsdg[region].value_results()[0];

        let ValueOrigin::Output {
            producer,
            output: 0,
        } = &res.origin
        else {
            panic!("expected output origin");
        };

        let data = rvsdg[*producer].expect_op_binary();

        assert_eq!(data.intrinsic().operator, BinaryOperator::Add);
        assert_eq!(data.value_inputs()[0], ValueInput::argument(TY_U32, 0));
        assert_eq!(data.value_inputs()[1], ValueInput::argument(TY_U32, 1));
    }
}
