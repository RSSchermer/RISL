//! Identifies and eliminates "pass-through" outputs of [Switch] nodes.
//!
//! A switch output is considered a "pass-through" if, for every branch of the switch node, the
//! value returned for that output is exactly the same entry input argument.
//!
//! When such an output is found:
//!
//! 1. All users of the switch output are redirected to the origin of the corresponding entry input.
//! 2. The redundant output is removed from the switch node.
//!
//! This transformation is performed using a post-order traversal to ensure that nested switch nodes
//! are simplified from the inside out, potentially revealing further pass-through elimination
//! opportunities in outer switch nodes.

use crate::rvsdg::visit::region_nodes::RegionNodesVisitor;
use crate::rvsdg::{Connectivity, Node, NodeKind, Rvsdg, ValueOrigin};
use crate::{Function, Module};

struct Job {
    switch_node: Node,
    output_to_input: Vec<(u32, u32)>,
}

struct JobCollector {
    jobs: Vec<Job>,
}

impl JobCollector {
    fn new() -> Self {
        Self { jobs: Vec::new() }
    }
}

impl RegionNodesVisitor for JobCollector {
    fn visit_node(&mut self, rvsdg: &Rvsdg, node: Node) {
        // We use post-order traversal (visiting sub-regions first) to ensure that nested switches
        // are processed from the inside out. This allows the pass to catch chains of
        // pass-throughs across nested switch boundaries in a single execution: if an inner
        // switch has a pass-through that is eliminated, it might reveal that the outer
        // switch now also contains a pass-through.
        match rvsdg[node].kind() {
            NodeKind::Switch(switch) => {
                for branch in switch.branches() {
                    self.visit_region(rvsdg, *branch);
                }
            }
            NodeKind::Loop(loop_node) => {
                self.visit_region(rvsdg, loop_node.loop_region());
            }
            NodeKind::Function(func) => {
                self.visit_region(rvsdg, func.body_region());
            }
            _ => (),
        }

        // Now process the current node
        if let NodeKind::Switch(switch_node) = rvsdg[node].kind() {
            let mut output_to_input = Vec::new();

            for (output, _) in switch_node.value_outputs().iter().enumerate() {
                let mut passthrough_input: Option<u32> = None;
                let mut is_passthrough = true;

                for branch_region in switch_node.branches() {
                    let result = &rvsdg[*branch_region].value_results()[output];

                    if let ValueOrigin::Argument(arg) = result.origin {
                        if let Some(existing) = passthrough_input {
                            if existing != arg {
                                is_passthrough = false;
                                break;
                            }
                        } else {
                            passthrough_input = Some(arg);
                        }
                    } else {
                        is_passthrough = false;
                        break;
                    }
                }

                if is_passthrough {
                    if let Some(input) = passthrough_input {
                        output_to_input.push((output as u32, input));
                    }
                }
            }

            if !output_to_input.is_empty() {
                self.jobs.push(Job {
                    switch_node: node,
                    output_to_input,
                });
            }
        }
    }
}

pub struct SwitchPassthroughEliminator {
    collector: JobCollector,
}

impl SwitchPassthroughEliminator {
    pub fn new() -> Self {
        Self {
            collector: JobCollector::new(),
        }
    }

    pub fn eliminate_in_fn(&mut self, rvsdg: &mut Rvsdg, function: Function) {
        let fn_node = rvsdg
            .get_function_node(function)
            .expect("function not registered");

        self.collector.visit_node(rvsdg, fn_node);

        for job in self.collector.jobs.drain(..) {
            // Process outputs in reverse order to avoid index-shifting issues when removing them.
            // Since the collector visits and pushes outputs to `output_to_input` in ascending
            // order, we can simply iterate the vector backwards.
            for (output, entry_input) in job.output_to_input.into_iter().rev() {
                let entry_input_origin = rvsdg[job.switch_node].expect_switch().entry_inputs()
                    [entry_input as usize]
                    .origin;

                rvsdg.reconnect_value_users(
                    rvsdg[job.switch_node].region(),
                    ValueOrigin::Output {
                        producer: job.switch_node,
                        output,
                    },
                    entry_input_origin,
                );

                rvsdg.remove_switch_output(job.switch_node, output);
            }
        }
    }
}

pub fn transform_entry_points(module: &Module, rvsdg: &mut Rvsdg) {
    let mut eliminator = SwitchPassthroughEliminator::new();

    for (entry_point, _) in module.entry_points.iter() {
        eliminator.eliminate_in_fn(rvsdg, entry_point);
    }
}

#[cfg(test)]
mod tests {
    use internment::Intern;

    use super::*;
    use crate::rvsdg::{Rvsdg, ValueInput};
    use crate::ty::{TY_U32, TypeRegistry};

    #[test]
    fn test_eliminate_passthrough() {
        let ty_registry = TypeRegistry::default();
        let mut rvsdg = Rvsdg::new(ty_registry.clone());
        let mut module = Module::new(Intern::new("test".to_string()));

        let func = Function {
            name: Intern::new("main".to_string()),
            module: Intern::new("test".to_string()),
        };

        let fn_ty = ty_registry.register(crate::ty::TypeKind::Function(func));
        module.fn_sigs.register(
            func,
            crate::FnSig {
                name: func.name,
                ty: fn_ty,
                args: vec![crate::FnArg {
                    ty: TY_U32,
                    shader_io_binding: None,
                }],
                ret_ty: None,
            },
        );

        let (_, region) = rvsdg.register_function(&module, func, vec![]);

        // Add a switch with one passthrough output
        let pred_input = rvsdg.add_const_predicate(region, 0);
        let pred_val = ValueInput::output(crate::ty::TY_PREDICATE, pred_input, 0);
        let input_val = ValueInput::argument(TY_U32, 0);

        let switch_node = rvsdg.add_switch(region, vec![pred_val, input_val], vec![], None);
        let branch0 = rvsdg.add_switch_branch(switch_node);
        let branch1 = rvsdg.add_switch_branch(switch_node);
        let output = rvsdg.add_switch_output(switch_node, TY_U32);

        // Connect both branches to return argument 0 (which corresponds to input_val, the second entry input)
        rvsdg.reconnect_region_result(branch0, output, ValueOrigin::Argument(0));
        rvsdg.reconnect_region_result(branch1, output, ValueOrigin::Argument(0));

        let user_node =
            rvsdg.add_value_proxy(region, ValueInput::output(TY_U32, switch_node, output));

        let mut eliminator = SwitchPassthroughEliminator::new();

        eliminator.eliminate_in_fn(&mut rvsdg, func);

        // Check if user_node now points to input_val's origin
        let user_input = &rvsdg[user_node].value_inputs()[0];
        assert_eq!(user_input.origin, input_val.origin);

        // Check if switch output was removed
        assert_eq!(rvsdg[switch_node].value_outputs().len(), 0);
    }
}
