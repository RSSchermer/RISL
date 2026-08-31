//! Identifies and eliminates pass-through values of [Loop] and [Switch] nodes.
//!
//! A loop-value is a pass-through value when the corresponding loop output is used, and the
//! loop-value is loop-invariant (the loop-value's loop-region result connects directly to the
//! loop-value's loop-region argument).
//!
//! A switch output represents a pass-through value if the output is used, and the corresponding
//! branch result connects to the same branch argument, for every branch in the switch node.
//!
//! Nodes are processed in post-order ("inside out") so that pass-through values eliminated in
//! inner nodes can expose a pass-through value in an enclosing node.

use crate::rvsdg::visit::region_nodes::RegionNodesVisitor;
use crate::rvsdg::{Connectivity, Node, NodeKind, Rvsdg, ValueOrigin, visit};
use crate::{Function, Module};

struct RegionOwnerCollector {
    nodes: Vec<Node>,
}

impl RegionOwnerCollector {
    fn new() -> Self {
        Self { nodes: Vec::new() }
    }
}

impl RegionNodesVisitor for RegionOwnerCollector {
    fn visit_node(&mut self, rvsdg: &Rvsdg, node: Node) {
        if matches!(rvsdg[node].kind(), NodeKind::Loop(_) | NodeKind::Switch(_)) {
            self.nodes.push(node);
        }

        visit::region_nodes::visit_node(self, rvsdg, node);
    }
}

pub struct PassthroughEliminator {
    collector: RegionOwnerCollector,
}

impl PassthroughEliminator {
    pub fn new() -> Self {
        Self {
            collector: RegionOwnerCollector::new(),
        }
    }

    pub fn eliminate_in_fn(&mut self, rvsdg: &mut Rvsdg, function: Function) -> bool {
        let fn_node = rvsdg
            .get_function_node(function)
            .expect("function not registered");

        self.collector.visit_node(rvsdg, fn_node);
        let mut changed = false;

        while let Some(node) = self.collector.nodes.pop() {
            changed |= match rvsdg[node].kind() {
                NodeKind::Loop(_) => eliminate_loop_passthroughs(rvsdg, node),
                NodeKind::Switch(_) => eliminate_switch_passthroughs(rvsdg, node),
                _ => unreachable!(),
            };
        }

        changed
    }
}

fn eliminate_loop_passthroughs(rvsdg: &mut Rvsdg, loop_node: Node) -> bool {
    let outer_region = rvsdg[loop_node].region();
    let loop_data = rvsdg[loop_node].expect_loop();
    let loop_region = loop_data.loop_region();
    let num_outputs = loop_data.value_outputs().len();
    let mut changed = false;

    for output in 0..num_outputs {
        // Loop region result 0 is the reentry condition; results 1..N+1 are loop values.
        let result_origin = rvsdg[loop_region].value_results()[output + 1].origin;

        if result_origin == ValueOrigin::Argument(output as u32)
            && !rvsdg[loop_node].value_outputs()[output].users.is_empty()
        {
            let input_origin = rvsdg[loop_node].value_inputs()[output].origin;

            rvsdg.reconnect_value_users(
                outer_region,
                ValueOrigin::Output {
                    producer: loop_node,
                    output: output as u32,
                },
                input_origin,
            );
            changed = true;
        }
    }

    changed
}

fn eliminate_switch_passthroughs(rvsdg: &mut Rvsdg, switch_node: Node) -> bool {
    let num_outputs = rvsdg[switch_node].expect_switch().value_outputs().len();
    let mut changed = false;

    for output in (0..num_outputs).rev() {
        let switch_data = rvsdg[switch_node].expect_switch();
        let mut passthrough_input = None;

        for branch_region in switch_data.branches() {
            let ValueOrigin::Argument(input) = rvsdg[*branch_region].value_results()[output].origin
            else {
                passthrough_input = None;

                break;
            };

            match passthrough_input {
                Some(existing) if existing != input => {
                    passthrough_input = None;

                    break;
                }
                None => passthrough_input = Some(input),
                _ => {}
            }
        }

        if let Some(entry_input) = passthrough_input {
            let entry_input_origin =
                rvsdg[switch_node].expect_switch().entry_inputs()[entry_input as usize].origin;

            rvsdg.reconnect_value_users(
                rvsdg[switch_node].region(),
                ValueOrigin::Output {
                    producer: switch_node,
                    output: output as u32,
                },
                entry_input_origin,
            );

            rvsdg.remove_switch_output(switch_node, output as u32);
            changed = true;
        }
    }

    changed
}

pub fn transform_entry_points(module: &Module, rvsdg: &mut Rvsdg) -> bool {
    let mut eliminator = PassthroughEliminator::new();
    let mut changed = false;

    for (entry_point, _) in module.entry_points.iter() {
        changed |= eliminator.eliminate_in_fn(rvsdg, entry_point);
    }

    changed
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::rvsdg::{ValueInput, ValueOutput};
    use crate::ty::{TY_DUMMY, TY_PREDICATE, TY_U32};
    use crate::{FnArg, FnSig, Symbol};

    #[test]
    fn eliminates_loop_passthrough() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref("test"),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
                ty: TY_DUMMY,
                args: vec![FnArg {
                    ty: TY_U32,
                    shader_io_binding: None,
                }],
                ret_ty: None,
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let const_0 = rvsdg.add_const_u32(region, 10);
        let const_1 = rvsdg.add_const_u32(region, 20);
        let (loop_node, loop_region) = rvsdg.add_loop(
            region,
            vec![
                ValueInput::output(TY_U32, const_0, 0),
                ValueInput::output(TY_U32, const_1, 0),
            ],
            None,
        );

        let condition = rvsdg.add_const_bool(loop_region, false);
        let proxy = rvsdg.add_value_proxy(loop_region, ValueInput::argument(TY_U32, 1));

        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: condition,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(loop_region, 1, ValueOrigin::Argument(0));
        rvsdg.reconnect_region_result(
            loop_region,
            2,
            ValueOrigin::Output {
                producer: proxy,
                output: 0,
            },
        );

        let user_0 = rvsdg.add_value_proxy(region, ValueInput::output(TY_U32, loop_node, 0));
        let user_1 = rvsdg.add_value_proxy(region, ValueInput::output(TY_U32, loop_node, 1));

        let mut eliminator = PassthroughEliminator::new();

        assert!(eliminator.eliminate_in_fn(&mut rvsdg, function));
        assert!(!eliminator.eliminate_in_fn(&mut rvsdg, function));

        assert_eq!(
            rvsdg[user_0].value_inputs()[0].origin,
            ValueOrigin::Output {
                producer: const_0,
                output: 0,
            }
        );
        assert_eq!(
            rvsdg[user_1].value_inputs()[0].origin,
            ValueOrigin::Output {
                producer: loop_node,
                output: 1,
            }
        );
    }

    #[test]
    fn eliminates_switch_passthrough() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref("test"),
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

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let predicate = rvsdg.add_const_predicate(region, 0);
        let switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::output(TY_PREDICATE, predicate, 0),
                ValueInput::argument(TY_U32, 0),
                ValueInput::argument(TY_U32, 1),
            ],
            vec![
                ValueOutput::new(TY_U32),
                ValueOutput::new(TY_U32),
                ValueOutput::new(TY_U32),
            ],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);

        rvsdg.reconnect_region_result(branch_0, 0, ValueOrigin::Argument(0));
        rvsdg.reconnect_region_result(branch_0, 1, ValueOrigin::Argument(1));
        rvsdg.reconnect_region_result(branch_0, 2, ValueOrigin::Argument(1));

        let branch_1 = rvsdg.add_switch_branch(switch_node);

        rvsdg.reconnect_region_result(branch_1, 0, ValueOrigin::Argument(0));
        rvsdg.reconnect_region_result(branch_1, 1, ValueOrigin::Argument(1));
        rvsdg.reconnect_region_result(branch_1, 2, ValueOrigin::Argument(0));

        let user_0 = rvsdg.add_value_proxy(region, ValueInput::output(TY_U32, switch_node, 0));
        let user_1 = rvsdg.add_value_proxy(region, ValueInput::output(TY_U32, switch_node, 1));
        let user_2 = rvsdg.add_value_proxy(region, ValueInput::output(TY_U32, switch_node, 2));

        let mut eliminator = PassthroughEliminator::new();

        assert!(eliminator.eliminate_in_fn(&mut rvsdg, function));
        assert!(!eliminator.eliminate_in_fn(&mut rvsdg, function));

        assert_eq!(
            rvsdg[user_0].value_inputs()[0].origin,
            ValueOrigin::Argument(0)
        );
        assert_eq!(
            rvsdg[user_1].value_inputs()[0].origin,
            ValueOrigin::Argument(1)
        );
        assert_eq!(
            rvsdg[user_2].value_inputs()[0].origin,
            ValueOrigin::Output {
                producer: switch_node,
                output: 0,
            }
        );
        assert_eq!(rvsdg[switch_node].value_outputs().len(), 1);
    }

    #[test]
    fn eliminates_alternating_passthrough_chain_in_one_run() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref("test"),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
                ty: TY_DUMMY,
                args: vec![FnArg {
                    ty: TY_U32,
                    shader_io_binding: None,
                }],
                ret_ty: None,
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let outer_predicate = rvsdg.add_const_predicate(region, 0);
        let outer_switch = rvsdg.add_switch(
            region,
            vec![
                ValueInput::output(TY_PREDICATE, outer_predicate, 0),
                ValueInput::argument(TY_U32, 0),
            ],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let outer_branch_0 = rvsdg.add_switch_branch(outer_switch);

        rvsdg.reconnect_region_result(outer_branch_0, 0, ValueOrigin::Argument(0));

        let outer_branch_1 = rvsdg.add_switch_branch(outer_switch);
        let (loop_node, loop_region) =
            rvsdg.add_loop(outer_branch_1, vec![ValueInput::argument(TY_U32, 0)], None);

        let inner_predicate = rvsdg.add_const_predicate(loop_region, 0);
        let inner_switch = rvsdg.add_switch(
            loop_region,
            vec![
                ValueInput::output(TY_PREDICATE, inner_predicate, 0),
                ValueInput::argument(TY_U32, 0),
            ],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let inner_branch_0 = rvsdg.add_switch_branch(inner_switch);

        rvsdg.reconnect_region_result(inner_branch_0, 0, ValueOrigin::Argument(0));

        let inner_branch_1 = rvsdg.add_switch_branch(inner_switch);

        rvsdg.reconnect_region_result(inner_branch_1, 0, ValueOrigin::Argument(0));

        let condition = rvsdg.add_const_bool(loop_region, false);

        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: condition,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            loop_region,
            1,
            ValueOrigin::Output {
                producer: inner_switch,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(
            outer_branch_1,
            0,
            ValueOrigin::Output {
                producer: loop_node,
                output: 0,
            },
        );

        let user = rvsdg.add_value_proxy(region, ValueInput::output(TY_U32, outer_switch, 0));

        let mut eliminator = PassthroughEliminator::new();

        assert!(eliminator.eliminate_in_fn(&mut rvsdg, function));
        assert!(!eliminator.eliminate_in_fn(&mut rvsdg, function));

        assert_eq!(
            rvsdg[user].value_inputs()[0].origin,
            ValueOrigin::Argument(0)
        );
        assert_eq!(rvsdg[outer_switch].value_outputs().len(), 0);
        assert_eq!(rvsdg[inner_switch].value_outputs().len(), 0);
    }
}
