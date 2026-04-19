//! Identifies and eliminates "pass-through" outputs of [Loop] nodes.
//!
//! A loop output is considered a "pass-through" if the corresponding loop-value is "loop constant":
//! the loop-region's result for that value connects directly back to the loop-region's argument
//! for that same value. When such an output is found, all users of the loop output are redirected
//! to the origin of the corresponding loop input.

use crate::rvsdg::visit::region_nodes::RegionNodesVisitor;
use crate::rvsdg::{Connectivity, Node, NodeKind, Rvsdg, ValueOrigin, visit};
use crate::{Function, Module};

struct LoopNodeCollector {
    nodes: Vec<Node>,
}

impl LoopNodeCollector {
    fn new() -> Self {
        Self { nodes: Vec::new() }
    }
}

impl RegionNodesVisitor for LoopNodeCollector {
    fn visit_node(&mut self, rvsdg: &Rvsdg, node: Node) {
        // We collect the nodes in outside-in order, the later tread the `nodes` vector as a stack
        // to effectively reverse the order and ensure inside-out processing. We do this because
        // a passthrough elimination for a nested loop node may uncover more passthroughs for an
        // outer loop node.

        if let NodeKind::Loop(_) = rvsdg[node].kind() {
            self.nodes.push(node);
        }

        visit::region_nodes::visit_node(self, rvsdg, node);
    }
}

pub struct LoopPassthroughEliminator {
    collector: LoopNodeCollector,
}

impl LoopPassthroughEliminator {
    pub fn new() -> Self {
        Self {
            collector: LoopNodeCollector::new(),
        }
    }

    pub fn eliminate_in_fn(&mut self, rvsdg: &mut Rvsdg, function: Function) {
        let fn_node = rvsdg
            .get_function_node(function)
            .expect("function not registered");

        self.collector.visit_node(rvsdg, fn_node);

        while let Some(loop_node) = self.collector.nodes.pop() {
            let outer_region = rvsdg[loop_node].region();
            let loop_data = rvsdg[loop_node].expect_loop();
            let loop_region = loop_data.loop_region();
            let num_outputs = loop_data.value_outputs().len();

            for i in 0..num_outputs {
                // Loop region results: index 0 is reentry condition, indices 1..N+1 are
                // loop-values.
                let result_index = i + 1;
                let result_origin = rvsdg[loop_region].value_results()[result_index].origin;

                if let ValueOrigin::Argument(arg_index) = result_origin {
                    if arg_index == i as u32 {
                        let input_origin = rvsdg[loop_node].value_inputs()[i].origin;

                        rvsdg.reconnect_value_users(
                            outer_region,
                            ValueOrigin::Output {
                                producer: loop_node,
                                output: i as u32,
                            },
                            input_origin,
                        );
                    }
                }
            }
        }
    }
}

pub fn transform_entry_points(module: &Module, rvsdg: &mut Rvsdg) {
    let mut eliminator = LoopPassthroughEliminator::new();

    for (entry_point, _) in module.entry_points.iter() {
        eliminator.eliminate_in_fn(rvsdg, entry_point);
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::rvsdg::ValueInput;
    use crate::ty::{TY_DUMMY, TY_U32, TypeRegistry};
    use crate::{FnSig, Symbol};

    #[test]
    fn test_eliminate_loop_passthrough() {
        let ty_registry = TypeRegistry::default();
        let mut rvsdg = Rvsdg::new(ty_registry.clone());
        let mut module = Module::new(Symbol::from_ref(""));

        let function = Function {
            name: Symbol::from_ref(""),
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

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let const_0 = rvsdg.add_const_u32(region, 10);
        let const_1 = rvsdg.add_const_u32(region, 20);

        let input_0 = ValueInput::output(TY_U32, const_0, 0);
        let input_1 = ValueInput::output(TY_U32, const_1, 0);

        // Create a loop with 2 loop-values
        let (loop_node, loop_region) = rvsdg.add_loop(region, vec![input_0, input_1], None);

        // Reentry condition: just a constant false
        let const_false = rvsdg.add_const_bool(loop_region, false);
        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: const_false,
                output: 0,
            },
        );

        // Loop value 0: Passthrough (Result 1 -> Argument 0)
        rvsdg.reconnect_region_result(loop_region, 1, ValueOrigin::Argument(0));

        // Loop value 1: Not passthrough (Result 2 -> some other node)
        let proxy = rvsdg.add_value_proxy(loop_region, ValueInput::argument(TY_U32, 1));
        rvsdg.reconnect_region_result(
            loop_region,
            2,
            ValueOrigin::Output {
                producer: proxy,
                output: 0,
            },
        );

        // Add users for loop outputs
        let user_0 = rvsdg.add_value_proxy(region, ValueInput::output(TY_U32, loop_node, 0));
        let user_1 = rvsdg.add_value_proxy(region, ValueInput::output(TY_U32, loop_node, 1));

        let mut eliminator = LoopPassthroughEliminator::new();
        eliminator.eliminate_in_fn(&mut rvsdg, function);

        // user_0 should be reconnected to input_0's origin
        assert_eq!(rvsdg[user_0].value_inputs()[0].origin, input_0.origin);

        // user_1 should still point to loop_node output 1
        assert_eq!(
            rvsdg[user_1].value_inputs()[0].origin,
            ValueOrigin::Output {
                producer: loop_node,
                output: 1,
            }
        );
    }
}
