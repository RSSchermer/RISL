use crate::rvsdg::visit::region_nodes::RegionNodesVisitor;
use crate::rvsdg::{Connectivity, Node, NodeKind, Region, Rvsdg, ValueOrigin, visit};

/// Collects the nodes in such a way that repeated calling `pop` will visit the loop nodes from the 
/// inside out.
/// 
/// This is important because deduplicating outputs can uncover additional duplicate branch results
/// in an outer switch branch region.
struct SwitchNodeCollector<'a> {
    queue: &'a mut Vec<Node>,
}

impl RegionNodesVisitor for SwitchNodeCollector<'_> {
    fn visit_node(&mut self, rvsdg: &Rvsdg, node: Node) {
        if let NodeKind::Switch(_) = rvsdg[node].kind() {
            self.queue.push(node);
        }

        visit::region_nodes::visit_node(self, rvsdg, node);
    }
}

pub struct DuplicateSwitchOutputEliminator {
    switch_node_queue: Vec<Node>,
    did_eliminate_duplicate_outputs: bool,
}

impl DuplicateSwitchOutputEliminator {
    pub fn new() -> Self {
        Self {
            switch_node_queue: Vec::new(),
            did_eliminate_duplicate_outputs: false,
        }
    }

    pub fn process_region(&mut self, rvsdg: &mut Rvsdg, region: Region) -> bool {
        self.switch_node_queue.clear();
        self.did_eliminate_duplicate_outputs = false;

        SwitchNodeCollector {
            queue: &mut self.switch_node_queue,
        }
        .visit_region(rvsdg, region);

        while let Some(switch_node) = self.switch_node_queue.pop() {
            self.process_switch_node(rvsdg, switch_node);
        }

        self.did_eliminate_duplicate_outputs
    }

    fn process_switch_node(&mut self, rvsdg: &mut Rvsdg, switch_node: Node) {
        let output_count = rvsdg[switch_node].expect_switch().value_outputs().len();

        for i in 0..output_count {
            for j in i + 1..output_count {
                let branches = rvsdg[switch_node].expect_switch().branches().to_vec();

                let mut is_duplicate = true;
                for &branch in &branches {
                    let result_i = rvsdg[branch].value_results()[i].origin;
                    let result_j = rvsdg[branch].value_results()[j].origin;

                    if result_i != result_j {
                        is_duplicate = false;
                        break;
                    }
                }

                if is_duplicate {
                    self.did_eliminate_duplicate_outputs = true;

                    let region = rvsdg[switch_node].region();
                    rvsdg.reconnect_value_users(
                        region,
                        ValueOrigin::Output {
                            producer: switch_node,
                            output: j as u32,
                        },
                        ValueOrigin::Output {
                            producer: switch_node,
                            output: i as u32,
                        },
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rvsdg::{Connectivity, ValueInput, ValueOutput};
    use crate::ty::{TY_DUMMY, TY_PREDICATE, TY_U32};
    use crate::{FnSig, Function, Module, Symbol};

    #[test]
    fn test_duplicate_switch_output_elimination() {
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

        let pred_input = rvsdg.add_const_predicate(region, 0);
        let pred_val = ValueInput::output(TY_PREDICATE, pred_input, 0);

        // Switch with 3 outputs: 0 and 2 are duplicates, 1 is not
        let switch_node = rvsdg.add_switch(
            region,
            vec![pred_val],
            vec![
                ValueOutput::new(TY_U32),
                ValueOutput::new(TY_U32),
                ValueOutput::new(TY_U32),
            ],
            None,
        );

        let branch0 = rvsdg.add_switch_branch(switch_node);

        let const0 = rvsdg.add_const_u32(branch0, 10);
        let const1 = rvsdg.add_const_u32(branch0, 11);

        // Branch 0 results: output 0 and 2 are const0, output 1 is const1
        rvsdg.reconnect_region_result(
            branch0,
            0,
            ValueOrigin::Output {
                producer: const0,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            branch0,
            1,
            ValueOrigin::Output {
                producer: const1,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            branch0,
            2,
            ValueOrigin::Output {
                producer: const0,
                output: 0,
            },
        );

        let branch1 = rvsdg.add_switch_branch(switch_node);

        let const2 = rvsdg.add_const_u32(branch1, 20);
        let const3 = rvsdg.add_const_u32(branch1, 21);

        // Branch 1 results: output 0 and 2 are const2, output 1 is const3
        rvsdg.reconnect_region_result(
            branch1,
            0,
            ValueOrigin::Output {
                producer: const2,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            branch1,
            1,
            ValueOrigin::Output {
                producer: const3,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            branch1,
            2,
            ValueOrigin::Output {
                producer: const2,
                output: 0,
            },
        );

        // A user of the third output (index 2)
        let user_of_2 = rvsdg.add_value_proxy(region, ValueInput::output(TY_U32, switch_node, 2));
        // A user of the second output (index 1)
        let user_of_1 = rvsdg.add_value_proxy(region, ValueInput::output(TY_U32, switch_node, 1));

        let mut eliminator = DuplicateSwitchOutputEliminator::new();
        let changed = eliminator.process_region(&mut rvsdg, region);

        assert!(changed);

        assert_eq!(rvsdg[switch_node].expect_switch().value_outputs().len(), 3);

        // User of 2 should be reconnected to output 0
        assert_eq!(
            rvsdg[user_of_2].value_inputs()[0].origin,
            ValueOrigin::Output {
                producer: switch_node,
                output: 0
            }
        );

        // User of 1 should still be connected to output 1
        assert_eq!(
            rvsdg[user_of_1].value_inputs()[0].origin,
            ValueOrigin::Output {
                producer: switch_node,
                output: 1
            }
        );
    }

    #[test]
    fn test_nested_duplicate_switch_output_elimination() {
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

        let pred_input = rvsdg.add_const_predicate(region, 0);
        let pred_val = ValueInput::output(TY_PREDICATE, pred_input, 0);

        // Outer switch
        let outer_switch = rvsdg.add_switch(
            region,
            vec![pred_val],
            vec![ValueOutput::new(TY_U32), ValueOutput::new(TY_U32)],
            None,
        );

        let outer_branch0 = rvsdg.add_switch_branch(outer_switch);

        // Inner switch in outer_branch0
        let inner_switch = rvsdg.add_switch(
            outer_branch0,
            vec![pred_val],
            vec![ValueOutput::new(TY_U32), ValueOutput::new(TY_U32)],
            None,
        );
        let inner_branch0 = rvsdg.add_switch_branch(inner_switch);
        let const0 = rvsdg.add_const_u32(inner_branch0, 10);
        rvsdg.reconnect_region_result(
            inner_branch0,
            0,
            ValueOrigin::Output {
                producer: const0,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            inner_branch0,
            1,
            ValueOrigin::Output {
                producer: const0,
                output: 0,
            },
        );

        let inner_branch1 = rvsdg.add_switch_branch(inner_switch);
        let const1 = rvsdg.add_const_u32(inner_branch1, 20);
        rvsdg.reconnect_region_result(
            inner_branch1,
            0,
            ValueOrigin::Output {
                producer: const1,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            inner_branch1,
            1,
            ValueOrigin::Output {
                producer: const1,
                output: 0,
            },
        );

        // Outer branch 0 results use inner switch outputs
        rvsdg.reconnect_region_result(
            outer_branch0,
            0,
            ValueOrigin::Output {
                producer: inner_switch,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            outer_branch0,
            1,
            ValueOrigin::Output {
                producer: inner_switch,
                output: 1,
            },
        );

        let outer_branch1 = rvsdg.add_switch_branch(outer_switch);
        let const2 = rvsdg.add_const_u32(outer_branch1, 30);
        rvsdg.reconnect_region_result(
            outer_branch1,
            0,
            ValueOrigin::Output {
                producer: const2,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            outer_branch1,
            1,
            ValueOrigin::Output {
                producer: const2,
                output: 0,
            },
        );

        let outer_user = rvsdg.add_value_proxy(region, ValueInput::output(TY_U32, outer_switch, 1));

        let mut eliminator = DuplicateSwitchOutputEliminator::new();
        let changed = eliminator.process_region(&mut rvsdg, region);

        assert!(changed);

        // Verify that the inner switch output `1` users are reconnected to output `0`.
        assert_eq!(
            rvsdg[outer_branch0].value_results()[1].origin,
            ValueOrigin::Output {
                producer: inner_switch,
                output: 0
            }
        );

        // Verify that outer switch output `1` users are reconnected to output `0`.
        assert_eq!(
            rvsdg[outer_user].value_inputs()[0].origin,
            ValueOrigin::Output {
                producer: outer_switch,
                output: 0
            }
        );

        // Output counts should remain the same
        assert_eq!(rvsdg[inner_switch].expect_switch().value_outputs().len(), 2);
        assert_eq!(rvsdg[outer_switch].expect_switch().value_outputs().len(), 2);
    }

    #[test]
    fn test_all_duplicate_switch_output_elimination() {
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

        let pred_input = rvsdg.add_const_predicate(region, 0);
        let pred_val = ValueInput::output(TY_PREDICATE, pred_input, 0);

        // Switch with 3 outputs: all are duplicates
        let switch_node = rvsdg.add_switch(
            region,
            vec![pred_val],
            vec![
                ValueOutput::new(TY_U32),
                ValueOutput::new(TY_U32),
                ValueOutput::new(TY_U32),
            ],
            None,
        );

        let branch0 = rvsdg.add_switch_branch(switch_node);
        let const0 = rvsdg.add_const_u32(branch0, 10);

        // Branch 0 results: all outputs are const0
        rvsdg.reconnect_region_result(
            branch0,
            0,
            ValueOrigin::Output {
                producer: const0,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            branch0,
            1,
            ValueOrigin::Output {
                producer: const0,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            branch0,
            2,
            ValueOrigin::Output {
                producer: const0,
                output: 0,
            },
        );

        let branch1 = rvsdg.add_switch_branch(switch_node);
        let const1 = rvsdg.add_const_u32(branch1, 20);

        // Branch 1 results: all outputs are const1
        rvsdg.reconnect_region_result(
            branch1,
            0,
            ValueOrigin::Output {
                producer: const1,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            branch1,
            1,
            ValueOrigin::Output {
                producer: const1,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            branch1,
            2,
            ValueOrigin::Output {
                producer: const1,
                output: 0,
            },
        );

        // Users of the second and third outputs
        let user_of_1 = rvsdg.add_value_proxy(region, ValueInput::output(TY_U32, switch_node, 1));
        let user_of_2 = rvsdg.add_value_proxy(region, ValueInput::output(TY_U32, switch_node, 2));

        let mut eliminator = DuplicateSwitchOutputEliminator::new();
        let changed = eliminator.process_region(&mut rvsdg, region);

        assert!(changed);

        assert_eq!(rvsdg[switch_node].expect_switch().value_outputs().len(), 3);

        // User of 1 should be reconnected to output 0
        assert_eq!(
            rvsdg[user_of_1].value_inputs()[0].origin,
            ValueOrigin::Output {
                producer: switch_node,
                output: 0
            }
        );

        // User of 2 should be reconnected to output 0
        assert_eq!(
            rvsdg[user_of_2].value_inputs()[0].origin,
            ValueOrigin::Output {
                producer: switch_node,
                output: 0
            }
        );
    }

    #[test]
    fn test_no_duplicate_switch_output_elimination() {
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

        let pred_input = rvsdg.add_const_predicate(region, 0);
        let pred_val = ValueInput::output(TY_PREDICATE, pred_input, 0);

        // Switch with 2 outputs
        let switch_node = rvsdg.add_switch(
            region,
            vec![pred_val],
            vec![ValueOutput::new(TY_U32), ValueOutput::new(TY_U32)],
            None,
        );

        let branch0 = rvsdg.add_switch_branch(switch_node);
        let const0 = rvsdg.add_const_u32(branch0, 10);

        // Branch 0 results: both outputs are const0
        rvsdg.reconnect_region_result(
            branch0,
            0,
            ValueOrigin::Output {
                producer: const0,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            branch0,
            1,
            ValueOrigin::Output {
                producer: const0,
                output: 0,
            },
        );

        let branch1 = rvsdg.add_switch_branch(switch_node);
        let const1 = rvsdg.add_const_u32(branch1, 20);
        let const2 = rvsdg.add_const_u32(branch1, 30);

        // Branch 1 results: different origins (const1 and const2)
        rvsdg.reconnect_region_result(
            branch1,
            0,
            ValueOrigin::Output {
                producer: const1,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            branch1,
            1,
            ValueOrigin::Output {
                producer: const2,
                output: 0,
            },
        );

        // A user of the second output
        let user_of_1 = rvsdg.add_value_proxy(region, ValueInput::output(TY_U32, switch_node, 1));

        let mut eliminator = DuplicateSwitchOutputEliminator::new();
        let changed = eliminator.process_region(&mut rvsdg, region);

        assert!(!changed);

        // User of 1 should still be connected to output 1
        assert_eq!(
            rvsdg[user_of_1].value_inputs()[0].origin,
            ValueOrigin::Output {
                producer: switch_node,
                output: 1
            }
        );
    }
}
