use rustc_hash::FxHashSet;

use crate::rvsdg::transform::region_replication::inline_switch_branch;
use crate::rvsdg::visit::region_nodes::RegionNodesVisitor;
use crate::rvsdg::visit::reverse_value_flow::ReverseValueFlowVisitor;
use crate::rvsdg::{Node, NodeKind, Region, Rvsdg, SimpleNode, ValueOrigin, visit};
use crate::{Function, Module};

struct NodeCollector<'a> {
    candidate_stack: &'a mut Vec<Node>,
}

impl RegionNodesVisitor for NodeCollector<'_> {
    fn visit_node(&mut self, rvsdg: &Rvsdg, node: Node) {
        use NodeKind::*;

        match rvsdg[node].kind() {
            Switch(_) => self.candidate_stack.push(node),
            _ => (),
        }

        visit::region_nodes::visit_node(self, rvsdg, node);
    }
}

pub struct ConstPredicateFinder<'a> {
    visited: &'a mut FxHashSet<(Region, ValueOrigin)>,
    predicate: Option<u32>,
}

impl<'a> ConstPredicateFinder<'a> {
    pub fn new(visited: &'a mut FxHashSet<(Region, ValueOrigin)>) -> Self {
        Self {
            visited,
            predicate: None,
        }
    }

    pub fn find(&mut self, rvsdg: &Rvsdg, switch_node: Node) -> Option<u32> {
        self.predicate = None;
        self.visited.clear();

        self.visit_value_input(rvsdg, switch_node, 0);

        self.predicate
    }
}

impl ReverseValueFlowVisitor for ConstPredicateFinder<'_> {
    fn should_visit(&mut self, region: Region, origin: ValueOrigin) -> bool {
        self.visited.insert((region, origin))
    }

    fn visit_value_output(&mut self, rvsdg: &Rvsdg, node: Node, output: u32) {
        use NodeKind::*;
        use SimpleNode::*;

        if let Simple(ConstPredicate(n)) = rvsdg[node].kind() {
            assert_eq!(output, 0);

            self.predicate = Some(n.value());
        }

        // Don't continue visiting for any node kind, regardless of if we found a constant
        // predicate; we only traverse up regions, and for all other instances of a predicate being
        // routed through some other node, we assume it is not constant.
    }

    fn visit_region_argument(&mut self, rvsdg: &Rvsdg, region: Region, argument: u32) {
        use NodeKind::*;

        let owner = rvsdg[region].owner();

        match rvsdg[owner].kind() {
            Loop(n) => {
                let result = argument + 1;
                let loop_region = n.loop_region();
                let result_origin = rvsdg[loop_region].value_results()[result as usize].origin;

                // Only continue visiting if the loop value stays the same for each iteration, that
                // is: the loop result that corresponds to the argument connects directly to the
                // argument.
                if result_origin == ValueOrigin::Argument(argument) {
                    visit::reverse_value_flow::visit_region_argument(self, rvsdg, region, argument)
                }
            }
            _ => visit::reverse_value_flow::visit_region_argument(self, rvsdg, region, argument),
        }
    }
}

pub struct ConstSwitchInliner {
    // We visit nodes from the "outside in", but we want to inline nodes from the "inside out"
    // (because if we inline from the outside in, inlining a node that contains more candidates will
    // invalidate our candidate list). Therefore, we must explicitly use a stack rather than a
    // queue.
    candidate_stack: Vec<Node>,
    visited: FxHashSet<(Region, ValueOrigin)>,
}

impl ConstSwitchInliner {
    pub fn new() -> Self {
        Self {
            candidate_stack: Vec::new(),
            visited: FxHashSet::default(),
        }
    }

    pub fn inline_in_fn(&mut self, module: &mut Module, rvsdg: &mut Rvsdg, function: Function) {
        let fn_node = rvsdg
            .get_function_node(function)
            .expect("function not registered");
        let mut collector = NodeCollector {
            candidate_stack: &mut self.candidate_stack,
        };

        collector.visit_node(rvsdg, fn_node);

        while let Some(node) = self.candidate_stack.pop() {
            self.process_switch_node(module, rvsdg, node);
        }
    }

    fn process_switch_node(&mut self, module: &mut Module, rvsdg: &mut Rvsdg, switch_node: Node) {
        let mut finder = ConstPredicateFinder::new(&mut self.visited);

        if let Some(predicate) = finder.find(rvsdg, switch_node) {
            inline_switch_branch(module, rvsdg, switch_node, predicate as usize);
        }
    }
}

pub fn transform_entry_points(module: &mut Module, rvsdg: &mut Rvsdg) {
    let mut inliner = ConstSwitchInliner::new();
    let entry_points = module.entry_points.iter().map(|e| e.0).collect::<Vec<_>>();

    for entry_point in entry_points {
        inliner.inline_in_fn(module, rvsdg, entry_point);
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::rvsdg::{ValueInput, ValueOutput};
    use crate::ty::{TY_DUMMY, TY_PREDICATE, TY_U32};
    use crate::{BinaryOperator, FnSig, Symbol};

    #[test]
    fn direct_const_predicate_input() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref(""),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Default::default(),
                ty: TY_DUMMY,
                args: vec![],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let pred_node = rvsdg.add_const_predicate(region, 0);
        let lhs_node = rvsdg.add_const_u32(region, 1);
        let rhs_node = rvsdg.add_const_u32(region, 2);
        let switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::output(TY_PREDICATE, pred_node, 0),
                ValueInput::output(TY_U32, lhs_node, 0),
                ValueInput::output(TY_U32, rhs_node, 0),
            ],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);

        let add_node = rvsdg.add_op_binary(
            branch_0,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 0),
            ValueInput::argument(TY_U32, 1),
        );

        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: add_node,
                output: 0,
            },
        );

        let branch_1 = rvsdg.add_switch_branch(switch_node);

        let fallback_node = rvsdg.add_const_u32(branch_1, 0);

        rvsdg.reconnect_region_result(
            branch_1,
            0,
            ValueOrigin::Output {
                producer: fallback_node,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: switch_node,
                output: 0,
            },
        );

        let mut inliner = ConstSwitchInliner::new();

        inliner.inline_in_fn(&mut module, &mut rvsdg, function);

        let ValueOrigin::Output {
            producer: add_node,
            output: 0,
        } = rvsdg[region].value_results()[0].origin
        else {
            panic!("expected result to be connected to a node's first output")
        };

        let add_data = rvsdg[add_node].expect_op_binary();

        assert_eq!(add_data.operator(), BinaryOperator::Add);
        assert_eq!(
            add_data.lhs_input().origin,
            ValueOrigin::Output {
                producer: lhs_node,
                output: 0
            }
        );
        assert_eq!(
            add_data.rhs_input().origin,
            ValueOrigin::Output {
                producer: rhs_node,
                output: 0
            }
        );
        assert!(!rvsdg.is_live_node(switch_node));
    }

    #[test]
    fn nested_const_switches() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref(""),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Default::default(),
                ty: TY_DUMMY,
                args: vec![],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let pred_0_node = rvsdg.add_const_predicate(region, 0);
        let pred_1_node = rvsdg.add_const_predicate(region, 0);
        let value_0_node = rvsdg.add_const_u32(region, 1);
        let value_1_node = rvsdg.add_const_u32(region, 2);
        let value_2_node = rvsdg.add_const_u32(region, 3);
        let outer_switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::output(TY_PREDICATE, pred_0_node, 0),
                ValueInput::output(TY_PREDICATE, pred_1_node, 0),
                ValueInput::output(TY_U32, value_0_node, 0),
                ValueInput::output(TY_U32, value_1_node, 0),
                ValueInput::output(TY_U32, value_2_node, 0),
            ],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let outer_branch_0 = rvsdg.add_switch_branch(outer_switch_node);

        let add_0_node = rvsdg.add_op_binary(
            outer_branch_0,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 1),
            ValueInput::argument(TY_U32, 2),
        );
        let inner_switch_node = rvsdg.add_switch(
            outer_branch_0,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(TY_U32, add_0_node, 0),
                ValueInput::argument(TY_U32, 3),
            ],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let inner_branch_0 = rvsdg.add_switch_branch(inner_switch_node);
        let add_1_node = rvsdg.add_op_binary(
            inner_branch_0,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 0),
            ValueInput::argument(TY_U32, 1),
        );

        rvsdg.reconnect_region_result(
            inner_branch_0,
            0,
            ValueOrigin::Output {
                producer: add_1_node,
                output: 0,
            },
        );

        let inner_branch_1 = rvsdg.add_switch_branch(inner_switch_node);

        let inner_fallback_node = rvsdg.add_const_u32(inner_branch_1, 0);

        rvsdg.reconnect_region_result(
            inner_branch_1,
            0,
            ValueOrigin::Output {
                producer: inner_fallback_node,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(
            outer_branch_0,
            0,
            ValueOrigin::Output {
                producer: inner_switch_node,
                output: 0,
            },
        );

        let outer_branch_1 = rvsdg.add_switch_branch(outer_switch_node);

        let outer_fallback_node = rvsdg.add_const_u32(outer_branch_1, 0);

        rvsdg.reconnect_region_result(
            outer_branch_1,
            0,
            ValueOrigin::Output {
                producer: outer_fallback_node,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: outer_switch_node,
                output: 0,
            },
        );

        let mut inliner = ConstSwitchInliner::new();

        inliner.inline_in_fn(&mut module, &mut rvsdg, function);

        let ValueOrigin::Output {
            producer: add_1_repl_node,
            output: 0,
        } = rvsdg[region].value_results()[0].origin
        else {
            panic!("expected result to be connected to a node's first output")
        };

        let add_1_repl_data = rvsdg[add_1_repl_node].expect_op_binary();

        assert_eq!(add_1_repl_data.operator(), BinaryOperator::Add);
        assert_eq!(
            add_1_repl_data.rhs_input().origin,
            ValueOrigin::Output {
                producer: value_2_node,
                output: 0
            }
        );

        let ValueOrigin::Output {
            producer: add_0_repl_node,
            output: 0,
        } = add_1_repl_data.lhs_input().origin
        else {
            panic!("expected result to be connected to a node's first output")
        };

        let add_0_repl_data = rvsdg[add_0_repl_node].expect_op_binary();

        assert_eq!(add_0_repl_data.operator(), BinaryOperator::Add);
        assert_eq!(
            add_0_repl_data.lhs_input().origin,
            ValueOrigin::Output {
                producer: value_0_node,
                output: 0
            }
        );
        assert_eq!(
            add_0_repl_data.rhs_input().origin,
            ValueOrigin::Output {
                producer: value_1_node,
                output: 0
            }
        );

        assert!(!rvsdg.is_live_node(inner_switch_node));
        assert!(!rvsdg.is_live_node(outer_switch_node));
    }
}
