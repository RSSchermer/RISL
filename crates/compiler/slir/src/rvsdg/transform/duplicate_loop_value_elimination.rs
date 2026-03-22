use crate::rvsdg::visit::region_nodes::RegionNodesVisitor;
use crate::rvsdg::{Connectivity, Node, NodeKind, Region, Rvsdg, ValueOrigin, visit};

/// Collects the nodes in such a way that repeated calling `pop` will visit the loop nodes from the
/// inside out.
struct LoopNodeCollector<'a> {
    queue: &'a mut Vec<Node>,
}

impl RegionNodesVisitor for LoopNodeCollector<'_> {
    fn visit_node(&mut self, rvsdg: &Rvsdg, node: Node) {
        if let NodeKind::Loop(_) = rvsdg[node].kind() {
            self.queue.push(node);
        }

        visit::region_nodes::visit_node(self, rvsdg, node);
    }
}

pub struct DuplicateLoopValueEliminator {
    loop_node_queue: Vec<Node>,
    did_eliminate_duplicate_values: bool,
}

impl DuplicateLoopValueEliminator {
    pub fn new() -> Self {
        Self {
            loop_node_queue: Vec::new(),
            did_eliminate_duplicate_values: false,
        }
    }

    pub fn process_region(&mut self, rvsdg: &mut Rvsdg, region: Region) -> bool {
        self.loop_node_queue.clear();
        self.did_eliminate_duplicate_values = false;

        LoopNodeCollector {
            queue: &mut self.loop_node_queue,
        }
        .visit_region(rvsdg, region);

        while let Some(loop_node) = self.loop_node_queue.pop() {
            self.process_loop_node(rvsdg, loop_node);
        }

        self.did_eliminate_duplicate_values
    }

    fn process_loop_node(&mut self, rvsdg: &mut Rvsdg, loop_node: Node) {
        let value_count = rvsdg[loop_node].value_inputs().len();
        let loop_region = rvsdg[loop_node].expect_loop().loop_region();

        for i in 0..value_count {
            let input_origin_i = rvsdg[loop_node].value_inputs()[i].origin;
            let result_origin_i = rvsdg[loop_region].value_results()[i + 1].origin;

            for j in i + 1..value_count {
                let input_origin_j = rvsdg[loop_node].value_inputs()[j].origin;
                let result_origin_j = rvsdg[loop_region].value_results()[j + 1].origin;

                // Value-inputs must be duplicates
                if input_origin_i != input_origin_j {
                    continue;
                }

                // Loop region results must be also duplicates. They are duplicates if:
                //
                // 1. Both connect to the same origin.
                // 2. Both are loop constants (origins are their corresponding arguments); we've
                //    already verified above that both arguments represent the same value for the
                //    initial iteration, so if they are loop constants, they will remain the same
                //    for all iterations.
                let results_are_duplicates = if result_origin_i == result_origin_j {
                    true
                } else {
                    let i_is_loop_const = result_origin_i == ValueOrigin::Argument(i as u32);
                    let j_is_loop_const = result_origin_j == ValueOrigin::Argument(j as u32);

                    i_is_loop_const && j_is_loop_const
                };

                if results_are_duplicates {
                    // Reconnect users of duplicate argument in the loop region.
                    rvsdg.reconnect_value_users(
                        loop_region,
                        ValueOrigin::Argument(j as u32),
                        ValueOrigin::Argument(i as u32),
                    );

                    // Reconnect users of duplicate value-output in the outer region.
                    let outer_region = rvsdg[loop_node].region();
                    rvsdg.reconnect_value_users(
                        outer_region,
                        ValueOrigin::Output {
                            producer: loop_node,
                            output: j as u32,
                        },
                        ValueOrigin::Output {
                            producer: loop_node,
                            output: i as u32,
                        },
                    );

                    self.did_eliminate_duplicate_values = true;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rvsdg::{Connectivity, ValueInput};
    use crate::ty::{TY_DUMMY, TY_U32};
    use crate::{FnSig, Function, Module, Symbol};

    #[test]
    fn test_duplicate_loop_value_elimination() {
        let mut module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let function = Function {
            name: Symbol::from_ref("f"),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
                ty: TY_DUMMY,
                args: vec![],
                ret_ty: None,
            },
        );

        let (_, region) = rvsdg.register_function(&module, function, vec![]);

        let const_val = rvsdg.add_const_u32(region, 10);
        let val_input = ValueInput::output(TY_U32, const_val, 0);

        // Loop with 3 values: 0 and 2 are duplicates, 1 is not.
        // All inputs are the same.
        let (loop_node, loop_region) =
            rvsdg.add_loop(region, vec![val_input, val_input, val_input], None);

        // Reentry condition
        let reentry_pred = rvsdg.add_const_bool(loop_region, false);
        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: reentry_pred,
                output: 0,
            },
        );

        // Value 0: loop constant (reentry = argument 0)
        rvsdg.reconnect_region_result(loop_region, 1, ValueOrigin::Argument(0));

        // Value 1: not a duplicate (reentry = some other value)
        let other_const = rvsdg.add_const_u32(loop_region, 20);
        rvsdg.reconnect_region_result(
            loop_region,
            2,
            ValueOrigin::Output {
                producer: other_const,
                output: 0,
            },
        );

        // Value 2: loop constant (reentry = argument 2) - duplicate of value 0
        rvsdg.reconnect_region_result(loop_region, 3, ValueOrigin::Argument(2));

        // Users
        let user_of_arg1 = rvsdg.add_value_proxy(loop_region, ValueInput::argument(TY_U32, 1));
        let user_of_arg2 = rvsdg.add_value_proxy(loop_region, ValueInput::argument(TY_U32, 2));

        let user_of_out1 = rvsdg.add_value_proxy(region, ValueInput::output(TY_U32, loop_node, 1));
        let user_of_out2 = rvsdg.add_value_proxy(region, ValueInput::output(TY_U32, loop_node, 2));

        let mut eliminator = DuplicateLoopValueEliminator::new();
        let changed = eliminator.process_region(&mut rvsdg, region);

        assert!(changed);

        // Verify users of argument 2 are reconnected to argument 0
        assert_eq!(
            rvsdg[user_of_arg2].value_inputs()[0].origin,
            ValueOrigin::Argument(0)
        );

        // Verify users of output 2 are reconnected to output 0
        assert_eq!(
            rvsdg[user_of_out2].value_inputs()[0].origin,
            ValueOrigin::Output {
                producer: loop_node,
                output: 0
            }
        );

        // Verify that the users of argument/output 1 are still connected to the original values.
        assert_eq!(
            rvsdg[user_of_arg1].value_inputs()[0].origin,
            ValueOrigin::Argument(1)
        );

        assert_eq!(
            rvsdg[user_of_out1].value_inputs()[0].origin,
            ValueOrigin::Output {
                producer: loop_node,
                output: 1
            }
        );
    }

    #[test]
    fn test_duplicate_loop_value_elimination_same_origin() {
        let mut module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let function = Function {
            name: Symbol::from_ref("f"),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
                ty: TY_DUMMY,
                args: vec![],
                ret_ty: None,
            },
        );

        let (_, region) = rvsdg.register_function(&module, function, vec![]);

        let const_val = rvsdg.add_const_u32(region, 10);
        let val_input = ValueInput::output(TY_U32, const_val, 0);

        // Loop with 2 values
        let (loop_node, loop_region) = rvsdg.add_loop(region, vec![val_input, val_input], None);

        // Reentry condition
        let reentry_pred = rvsdg.add_const_bool(loop_region, false);
        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: reentry_pred,
                output: 0,
            },
        );

        // Both values reentry from the same origin in the loop region
        let internal_const = rvsdg.add_const_u32(loop_region, 30);
        let internal_origin = ValueOrigin::Output {
            producer: internal_const,
            output: 0,
        };
        rvsdg.reconnect_region_result(loop_region, 1, internal_origin);
        rvsdg.reconnect_region_result(loop_region, 2, internal_origin);

        let user_of_out1 = rvsdg.add_value_proxy(region, ValueInput::output(TY_U32, loop_node, 1));

        let mut eliminator = DuplicateLoopValueEliminator::new();
        let changed = eliminator.process_region(&mut rvsdg, region);

        assert!(changed);

        assert_eq!(
            rvsdg[user_of_out1].value_inputs()[0].origin,
            ValueOrigin::Output {
                producer: loop_node,
                output: 0
            }
        );
    }

    #[test]
    fn test_no_duplicate_loop_value_elimination_different_inputs() {
        let mut module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let function = Function {
            name: Symbol::from_ref("f"),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
                ty: TY_DUMMY,
                args: vec![],
                ret_ty: None,
            },
        );

        let (_, region) = rvsdg.register_function(&module, function, vec![]);

        let const10 = rvsdg.add_const_u32(region, 10);
        let const20 = rvsdg.add_const_u32(region, 20);
        let input0 = ValueInput::output(TY_U32, const10, 0);
        let input1 = ValueInput::output(TY_U32, const20, 0);

        // Loop with 2 values, different inputs
        let (_loop_node, loop_region) = rvsdg.add_loop(region, vec![input0, input1], None);

        // Reentry condition
        let reentry_pred = rvsdg.add_const_bool(loop_region, false);
        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: reentry_pred,
                output: 0,
            },
        );

        // Both are loop constants (reentry = their own argument)
        rvsdg.reconnect_region_result(loop_region, 1, ValueOrigin::Argument(0));
        rvsdg.reconnect_region_result(loop_region, 2, ValueOrigin::Argument(1));

        let mut eliminator = DuplicateLoopValueEliminator::new();
        let changed = eliminator.process_region(&mut rvsdg, region);

        assert!(!changed);
    }

    #[test]
    fn test_no_duplicate_loop_value_elimination_different_results() {
        let mut module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let function = Function {
            name: Symbol::from_ref("f"),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
                ty: TY_DUMMY,
                args: vec![],
                ret_ty: None,
            },
        );

        let (_, region) = rvsdg.register_function(&module, function, vec![]);

        let const10 = rvsdg.add_const_u32(region, 10);
        let input = ValueInput::output(TY_U32, const10, 0);

        // Loop with 2 values, same inputs
        let (_loop_node, loop_region) = rvsdg.add_loop(region, vec![input, input], None);

        // Reentry condition
        let reentry_pred = rvsdg.add_const_bool(loop_region, false);
        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: reentry_pred,
                output: 0,
            },
        );

        // Different results
        let internal1 = rvsdg.add_const_u32(loop_region, 100);
        let internal2 = rvsdg.add_const_u32(loop_region, 200);
        rvsdg.reconnect_region_result(
            loop_region,
            1,
            ValueOrigin::Output {
                producer: internal1,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            loop_region,
            2,
            ValueOrigin::Output {
                producer: internal2,
                output: 0,
            },
        );

        let mut eliminator = DuplicateLoopValueEliminator::new();
        let changed = eliminator.process_region(&mut rvsdg, region);

        assert!(!changed);
    }

    #[test]
    fn test_nested_duplicate_loop_value_elimination() {
        let mut module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let function = Function {
            name: Symbol::from_ref("f"),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
                ty: TY_DUMMY,
                args: vec![],
                ret_ty: None,
            },
        );

        let (_, region) = rvsdg.register_function(&module, function, vec![]);

        let const10 = rvsdg.add_const_u32(region, 10);
        let val_input = ValueInput::output(TY_U32, const10, 0);

        // Outer Loop
        let (outer_loop, outer_region) = rvsdg.add_loop(region, vec![val_input, val_input], None);
        let outer_reentry = rvsdg.add_const_bool(outer_region, false);
        rvsdg.reconnect_region_result(
            outer_region,
            0,
            ValueOrigin::Output {
                producer: outer_reentry,
                output: 0,
            },
        );

        // Inner Loop in Outer Region
        let inner_input = ValueInput::argument(TY_U32, 0);
        let (inner_loop, inner_region) =
            rvsdg.add_loop(outer_region, vec![inner_input, inner_input], None);
        let inner_reentry = rvsdg.add_const_bool(inner_region, false);
        rvsdg.reconnect_region_result(
            inner_region,
            0,
            ValueOrigin::Output {
                producer: inner_reentry,
                output: 0,
            },
        );

        // Inner loop duplicates
        rvsdg.reconnect_region_result(inner_region, 1, ValueOrigin::Argument(0));
        rvsdg.reconnect_region_result(inner_region, 2, ValueOrigin::Argument(1));

        // Outer loop results use inner loop outputs
        rvsdg.reconnect_region_result(
            outer_region,
            1,
            ValueOrigin::Output {
                producer: inner_loop,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            outer_region,
            2,
            ValueOrigin::Output {
                producer: inner_loop,
                output: 1,
            },
        );

        // User of inner loop output 1 (should be reconnected to 0)
        let inner_user =
            rvsdg.add_value_proxy(outer_region, ValueInput::output(TY_U32, inner_loop, 1));
        // User of outer loop output 1 (should be reconnected to 0)
        let outer_user = rvsdg.add_value_proxy(region, ValueInput::output(TY_U32, outer_loop, 1));

        let mut eliminator = DuplicateLoopValueEliminator::new();
        let changed = eliminator.process_region(&mut rvsdg, region);

        assert!(changed);

        // Verify inner loop reconnection
        assert_eq!(
            rvsdg[inner_user].value_inputs()[0].origin,
            ValueOrigin::Output {
                producer: inner_loop,
                output: 0
            }
        );

        // Verify that result 2 of the outer loop has been reconnected to inner loop output 0
        assert_eq!(
            rvsdg[outer_region].value_results()[2].origin,
            ValueOrigin::Output {
                producer: inner_loop,
                output: 0
            }
        );

        // Verify outer loop reconnection (nested processing should make outer loop results duplicates)
        assert_eq!(
            rvsdg[outer_user].value_inputs()[0].origin,
            ValueOrigin::Output {
                producer: outer_loop,
                output: 0
            }
        );
    }
}
