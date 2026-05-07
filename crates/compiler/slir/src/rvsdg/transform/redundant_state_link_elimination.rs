//! Identifies and unlinks [SwitchNode]s and [LoopNode]s from the state chain when their internal
//! regions do not actually use state.
//!
//! After memory promotion (and similar transformations), it is common for a switch node's branch
//! regions or a loop node's loop region to no longer perform any stateful operations. In such
//! cases, leaving the node linked into its region's state chain creates an unnecessary reordering
//! constraint that may inhibit further optimizations.
//!
//! A region is said to "not use state" when its state result connects directly to its state
//! argument (without passing through any other nodes). A switch node is said to "not use state" if
//! none of its branches use state. A loop node is said to "not use state" if its loop region does
//! not use state.

use crate::rvsdg::visit::region_nodes::RegionNodesVisitor;
use crate::rvsdg::{Connectivity, Node, NodeKind, Rvsdg, StateOrigin, visit};
use crate::{Function, Module};

struct NodeCollector {
    nodes: Vec<Node>,
}

impl NodeCollector {
    fn new() -> Self {
        Self { nodes: Vec::new() }
    }
}

impl RegionNodesVisitor for NodeCollector {
    fn visit_node(&mut self, rvsdg: &Rvsdg, node: Node) {
        // Collect outside-in now, then later consume by popping from the stack, which results in
        // inside-out processing.
        match rvsdg[node].kind() {
            NodeKind::Switch(_) | NodeKind::Loop(_) => {
                if rvsdg[node].state().is_some() {
                    self.nodes.push(node);
                }
            }
            _ => {}
        }

        visit::region_nodes::visit_node(self, rvsdg, node);
    }
}

pub struct RedundantStateLinkEliminator {
    collector: NodeCollector,
}

impl RedundantStateLinkEliminator {
    pub fn new() -> Self {
        Self {
            collector: NodeCollector::new(),
        }
    }

    pub fn eliminate_in_fn(&mut self, rvsdg: &mut Rvsdg, function: Function) {
        let fn_node = rvsdg
            .get_function_node(function)
            .expect("function not registered");

        self.collector.nodes.clear();
        self.collector.visit_node(rvsdg, fn_node);

        while let Some(node) = self.collector.nodes.pop() {
            if rvsdg[node].state().is_none() {
                continue;
            }

            match rvsdg[node].kind() {
                NodeKind::Switch(switch_node) => {
                    let all_stateless = switch_node
                        .branches()
                        .iter()
                        .all(|branch| *rvsdg[*branch].state_result() == StateOrigin::Argument);

                    if all_stateless {
                        rvsdg.unlink_switch_state(node);
                    }
                }
                NodeKind::Loop(loop_node) => {
                    let loop_region = loop_node.loop_region();
                    let stateless = *rvsdg[loop_region].state_result() == StateOrigin::Argument;

                    if stateless {
                        rvsdg.unlink_loop_state(node);
                    }
                }
                _ => {}
            }
        }
    }
}

pub fn transform_entry_points(module: &Module, rvsdg: &mut Rvsdg) {
    let mut eliminator = RedundantStateLinkEliminator::new();

    for (entry_point, _) in module.entry_points.iter() {
        eliminator.eliminate_in_fn(rvsdg, entry_point);
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::rvsdg::{StateUser, ValueInput, ValueOrigin};
    use crate::ty::{TY_DUMMY, TY_PREDICATE, TY_PTR_U32, TY_U32};
    use crate::{FnSig, Symbol};

    #[test]
    fn test_unlink_redundantly_linked_switch() {
        let function = Function {
            name: Symbol::from_ref(""),
            module: Symbol::from_ref(""),
        };
        let mut module = Module::new(Symbol::from_ref(""));
        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
                ty: TY_DUMMY,
                args: vec![],
                ret_ty: None,
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let pred = rvsdg.add_const_predicate(region, 0);

        // A switch with two empty branches; linked into state chain.
        let switch = rvsdg.add_switch(
            region,
            vec![ValueInput::output(TY_PREDICATE, pred, 0)],
            vec![],
            Some(StateOrigin::Argument),
        );
        rvsdg.add_switch_branch(switch);
        rvsdg.add_switch_branch(switch);

        assert!(rvsdg[switch].state().is_some());

        let mut elim = RedundantStateLinkEliminator::new();
        elim.eliminate_in_fn(&mut rvsdg, function);

        assert!(
            rvsdg[switch].state().is_none(),
            "switch with stateless branches should be unlinked"
        );
        // Region's state chain should now connect argument directly to result.
        assert_eq!(*rvsdg[region].state_result(), StateOrigin::Argument);
        assert_eq!(*rvsdg[region].state_argument(), StateUser::Result);
    }

    #[test]
    fn test_unlink_redundantly_linked_loop() {
        let function = Function {
            name: Symbol::from_ref(""),
            module: Symbol::from_ref(""),
        };
        let mut module = Module::new(Symbol::from_ref(""));
        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
                ty: TY_DUMMY,
                args: vec![],
                ret_ty: None,
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let (loop_node, loop_region) = rvsdg.add_loop(region, vec![], Some(StateOrigin::Argument));
        let const_false = rvsdg.add_const_bool(loop_region, false);
        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: const_false,
                output: 0,
            },
        );

        assert!(rvsdg[loop_node].state().is_some());

        let mut elim = RedundantStateLinkEliminator::new();
        elim.eliminate_in_fn(&mut rvsdg, function);

        assert!(
            rvsdg[loop_node].state().is_none(),
            "loop with stateless body should be unlinked"
        );
    }

    #[test]
    fn test_preserve_switch_with_one_stateful_branch() {
        let function = Function {
            name: Symbol::from_ref(""),
            module: Symbol::from_ref(""),
        };
        let mut module = Module::new(Symbol::from_ref(""));
        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
                ty: TY_DUMMY,
                args: vec![],
                ret_ty: None,
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let pred = rvsdg.add_const_predicate(region, 0);

        let switch = rvsdg.add_switch(
            region,
            vec![ValueInput::output(TY_PREDICATE, pred, 0)],
            vec![],
            Some(StateOrigin::Argument),
        );

        // Empty branch (does not use state).
        let _branch_0 = rvsdg.add_switch_branch(switch);

        // Stateful branch: contains an op_store, which uses state.
        let branch_1 = rvsdg.add_switch_branch(switch);
        let alloca = rvsdg.add_op_alloca(branch_1, TY_U32);
        let value = rvsdg.add_const_u32(branch_1, 0);
        let _store = rvsdg.add_op_store(
            branch_1,
            ValueInput::output(TY_PTR_U32, alloca, 0),
            ValueInput::output(TY_U32, value, 0),
            StateOrigin::Argument,
        );

        assert!(rvsdg[switch].state().is_some());
        assert_ne!(*rvsdg[branch_1].state_result(), StateOrigin::Argument);

        let mut elim = RedundantStateLinkEliminator::new();
        elim.eliminate_in_fn(&mut rvsdg, function);

        assert!(
            rvsdg[switch].state().is_some(),
            "switch with at least one stateful branch must remain linked"
        );
    }

    #[test]
    fn test_preserve_stateful_loop() {
        let function = Function {
            name: Symbol::from_ref(""),
            module: Symbol::from_ref(""),
        };
        let mut module = Module::new(Symbol::from_ref(""));
        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
                ty: TY_DUMMY,
                args: vec![],
                ret_ty: None,
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let (loop_node, loop_region) = rvsdg.add_loop(region, vec![], Some(StateOrigin::Argument));

        let alloca = rvsdg.add_op_alloca(loop_region, TY_U32);
        let value = rvsdg.add_const_u32(loop_region, 0);
        let _store = rvsdg.add_op_store(
            loop_region,
            ValueInput::output(TY_PTR_U32, alloca, 0),
            ValueInput::output(TY_U32, value, 0),
            StateOrigin::Argument,
        );

        let const_false = rvsdg.add_const_bool(loop_region, false);
        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: const_false,
                output: 0,
            },
        );

        assert!(rvsdg[loop_node].state().is_some());
        assert_ne!(*rvsdg[loop_region].state_result(), StateOrigin::Argument);

        let mut elim = RedundantStateLinkEliminator::new();
        elim.eliminate_in_fn(&mut rvsdg, function);

        assert!(
            rvsdg[loop_node].state().is_some(),
            "loop with stateful body must remain linked"
        );
    }

    #[test]
    fn test_nested_unlinking() {
        let function = Function {
            name: Symbol::from_ref(""),
            module: Symbol::from_ref(""),
        };
        let mut module = Module::new(Symbol::from_ref(""));
        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
                ty: TY_DUMMY,
                args: vec![],
                ret_ty: None,
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        // Outer switch, linked into the function-region's state chain.
        let outer_pred = rvsdg.add_const_predicate(region, 0);
        let outer = rvsdg.add_switch(
            region,
            vec![ValueInput::output(TY_PREDICATE, outer_pred, 0)],
            vec![],
            Some(StateOrigin::Argument),
        );
        let outer_branch = rvsdg.add_switch_branch(outer);

        // Inner switch inside the outer branch, linked into the branch's state chain.
        let inner_pred = rvsdg.add_const_predicate(outer_branch, 0);
        let inner = rvsdg.add_switch(
            outer_branch,
            vec![ValueInput::output(TY_PREDICATE, inner_pred, 0)],
            vec![],
            Some(StateOrigin::Argument),
        );
        rvsdg.add_switch_branch(inner);

        assert!(rvsdg[outer].state().is_some());
        assert!(rvsdg[inner].state().is_some());

        let mut elim = RedundantStateLinkEliminator::new();
        elim.eliminate_in_fn(&mut rvsdg, function);

        assert!(rvsdg[inner].state().is_none(), "inner should be unlinked");
        assert!(rvsdg[outer].state().is_none(), "outer should be unlinked");
    }
}
