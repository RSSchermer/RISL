use std::collections::VecDeque;

use crate::rvsdg::visit::region_nodes::RegionNodesVisitor;
use crate::rvsdg::{Connectivity, Node, NodeKind, Region, Rvsdg, ValueOrigin, visit};

struct SwitchNodeCollector<'a> {
    queue: &'a mut VecDeque<Node>,
}

impl RegionNodesVisitor for SwitchNodeCollector<'_> {
    fn visit_node(&mut self, rvsdg: &Rvsdg, node: Node) {
        if let NodeKind::Switch(_) = rvsdg[node].kind() {
            // Push the node into the queue first before visiting its nested regions. This ensures
            // that any potential nested switch nodes are processed from the outside in. This is
            // important because duplicate input elimination in the outer switch node can uncover
            // additional duplicate inputs for any inner switch nodes.
            self.queue.push_back(node);

            visit::region_nodes::visit_node(self, rvsdg, node);
        }
    }
}

pub struct DuplicateSwitchInputEliminator {
    switch_node_queue: VecDeque<Node>,
    did_eliminate_duplicate_inputs: bool,
}

impl DuplicateSwitchInputEliminator {
    pub fn new() -> Self {
        Self {
            switch_node_queue: VecDeque::new(),
            did_eliminate_duplicate_inputs: false,
        }
    }

    pub fn process_region(&mut self, rvsdg: &mut Rvsdg, region: Region) -> bool {
        self.switch_node_queue.clear();
        self.did_eliminate_duplicate_inputs = false;

        SwitchNodeCollector {
            queue: &mut self.switch_node_queue,
        }
        .visit_region(rvsdg, region);

        while let Some(switch_node) = self.switch_node_queue.pop_front() {
            self.process_switch_node(rvsdg, switch_node);
        }

        self.did_eliminate_duplicate_inputs
    }

    fn process_switch_node(&mut self, rvsdg: &mut Rvsdg, switch_node: Node) {
        let switch_data = rvsdg[switch_node].expect_switch();
        let input_count = switch_data.value_inputs().len();
        let arg_count = input_count - 1;
        let branch_count = switch_data.branches().len();

        // The first input is the branch-selector, we only look for duplicates in inputs that are
        // passed in as branch arguments.
        for i in 0..arg_count {
            let origin = rvsdg[switch_node].value_inputs()[i + 1].origin;

            // Search for duplicate inputs to the right; any duplicates to the left will have
            // already been adjusted.
            for j in i + 1..arg_count {
                if rvsdg[switch_node].value_inputs()[j + 1].origin == origin {
                    self.did_eliminate_duplicate_inputs = true;

                    for b in 0..branch_count {
                        let branch = rvsdg[switch_node].expect_switch().branches()[b];

                        rvsdg.reconnect_value_users(
                            branch,
                            ValueOrigin::Argument(j as u32),
                            ValueOrigin::Argument(i as u32),
                        )
                    }

                    self.did_eliminate_duplicate_inputs = true;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rvsdg::{ValueInput, ValueOrigin};
    use crate::ty::{TY_DUMMY, TY_PREDICATE, TY_U32};
    use crate::{FnArg, FnSig, Function, Module, Symbol};

    #[test]
    fn test_duplicate_switch_input_elimination() {
        let mut module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let function = Function {
            name: Symbol::from_ref(""),
            module: Symbol::from_ref(""),
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
                ],
                ret_ty: None,
            },
        );

        let (_, region) = rvsdg.register_function(&module, function, vec![]);

        let pred_input = rvsdg.add_const_predicate(region, 0);
        let pred_val = ValueInput::output(TY_PREDICATE, pred_input, 0);

        let input_a_val = ValueInput::argument(TY_U32, 0);
        let input_b_val = ValueInput::argument(TY_U32, 1);

        // Switch with 4 value-inputs:
        // 0: branch-selector (pred_val)
        // 1: input_a_val (Argument(0))
        // 2: input_b_val (Argument(1))
        // 3: input_a_val (Argument(0)) -> Duplicate of input 1
        let switch_node = rvsdg.add_switch(
            region,
            vec![pred_val, input_a_val, input_b_val, input_a_val],
            vec![],
            None,
        );

        let branch0 = rvsdg.add_switch_branch(switch_node);
        let branch1 = rvsdg.add_switch_branch(switch_node);

        let user0 = rvsdg.add_value_proxy(branch0, ValueInput::argument(TY_U32, 2));
        let user1 = rvsdg.add_value_proxy(branch1, ValueInput::argument(TY_U32, 2));
        let user2 = rvsdg.add_value_proxy(branch0, ValueInput::argument(TY_U32, 1));

        let mut eliminator = DuplicateSwitchInputEliminator::new();
        let changed = eliminator.process_region(&mut rvsdg, region);

        assert!(changed);

        // Check if users were reconnected
        assert_eq!(
            rvsdg[user0].value_inputs()[0].origin,
            ValueOrigin::Argument(0)
        );
        assert_eq!(
            rvsdg[user1].value_inputs()[0].origin,
            ValueOrigin::Argument(0)
        );

        // Check if user of Argument(1) remained unchanged
        assert_eq!(
            rvsdg[user2].value_inputs()[0].origin,
            ValueOrigin::Argument(1)
        );
    }

    #[test]
    fn test_duplicate_switch_input_elimination_multiple_duplicates() {
        let mut module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let function = Function {
            name: Symbol::from_ref(""),
            module: Symbol::from_ref(""),
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
                ],
                ret_ty: None,
            },
        );

        let (_, region) = rvsdg.register_function(&module, function, vec![]);

        let pred_input = rvsdg.add_const_predicate(region, 0);
        let pred_val = ValueInput::output(TY_PREDICATE, pred_input, 0);

        let input_a_val = ValueInput::argument(TY_U32, 0);
        let input_b_val = ValueInput::argument(TY_U32, 1);

        // Switch with 5 value-inputs:
        // 0: branch-selector (pred_val)
        // 1: input_a_val (Argument(0))
        // 2: input_b_val (Argument(1))
        // 3: input_a_val (Argument(0)) -> Duplicate of input 1
        // 4: input_a_val (Argument(0)) -> Another duplicate of input 1
        let switch_node = rvsdg.add_switch(
            region,
            vec![pred_val, input_a_val, input_b_val, input_a_val, input_a_val],
            vec![],
            None,
        );

        let branch0 = rvsdg.add_switch_branch(switch_node);

        let user0 = rvsdg.add_value_proxy(branch0, ValueInput::argument(TY_U32, 2)); // Argument(2) -> Input index 3
        let user1 = rvsdg.add_value_proxy(branch0, ValueInput::argument(TY_U32, 3)); // Argument(3) -> Input index 4
        let user2 = rvsdg.add_value_proxy(branch0, ValueInput::argument(TY_U32, 1)); // Argument(1) -> Input index 2

        let mut eliminator = DuplicateSwitchInputEliminator::new();
        let changed = eliminator.process_region(&mut rvsdg, region);

        assert!(changed);

        // Check if users of Argument(2) and Argument(3) were reconnected to Argument(0)
        assert_eq!(
            rvsdg[user0].value_inputs()[0].origin,
            ValueOrigin::Argument(0)
        );
        assert_eq!(
            rvsdg[user1].value_inputs()[0].origin,
            ValueOrigin::Argument(0)
        );

        // Check if user of Argument(1) remained unchanged
        assert_eq!(
            rvsdg[user2].value_inputs()[0].origin,
            ValueOrigin::Argument(1)
        );
    }

    #[test]
    fn test_duplicate_switch_input_elimination_nested() {
        let mut module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let function = Function {
            name: Symbol::from_ref(""),
            module: Symbol::from_ref(""),
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
                ],
                ret_ty: None,
            },
        );

        let (_, region) = rvsdg.register_function(&module, function, vec![]);

        let pred_input = rvsdg.add_const_predicate(region, 0);
        let pred_val = ValueInput::output(TY_PREDICATE, pred_input, 0);

        let input_a_val = ValueInput::argument(TY_U32, 0);
        let input_b_val = ValueInput::argument(TY_U32, 1);

        // Outer Switch with 4 value-inputs:
        // 0: branch-selector (pred_val)
        // 1: input_a_val (Argument(0))
        // 2: input_b_val (Argument(1))
        // 3: input_a_val (Argument(0)) -> Duplicate of input 1
        let outer_switch = rvsdg.add_switch(
            region,
            vec![pred_val, input_a_val, input_b_val, input_a_val],
            vec![],
            None,
        );

        let outer_branch0 = rvsdg.add_switch_branch(outer_switch);

        let pred_input_inner = rvsdg.add_const_predicate(outer_branch0, 0);
        let pred_val_inner = ValueInput::output(TY_PREDICATE, pred_input_inner, 0);

        // Inner Switch with 4 value-inputs:
        // 0: branch-selector (pred_val_inner)
        // 1: Argument(0) of outer branch (traces to input_a_val)
        // 2: Argument(1) of outer branch (traces to input_b_val)
        // 3: Argument(2) of outer branch (traces to input_a_val)
        let inner_switch = rvsdg.add_switch(
            outer_branch0,
            vec![
                pred_val_inner,
                ValueInput::argument(TY_U32, 0),
                ValueInput::argument(TY_U32, 1),
                ValueInput::argument(TY_U32, 2),
            ],
            vec![],
            None,
        );

        let inner_branch0 = rvsdg.add_switch_branch(inner_switch);
        let inner_user = rvsdg.add_value_proxy(inner_branch0, ValueInput::argument(TY_U32, 2));

        let mut eliminator = DuplicateSwitchInputEliminator::new();
        let changed = eliminator.process_region(&mut rvsdg, region);

        assert!(changed);

        // Reconnected the final input of the inner switch to Argument(0) of outer branch
        assert_eq!(
            rvsdg[inner_switch].value_inputs()[3].origin,
            ValueOrigin::Argument(0)
        );

        // Reconnected the user of Argument(2) of inner switch to Argument(0)
        assert_eq!(
            rvsdg[inner_user].value_inputs()[0].origin,
            ValueOrigin::Argument(0)
        );
    }
}
