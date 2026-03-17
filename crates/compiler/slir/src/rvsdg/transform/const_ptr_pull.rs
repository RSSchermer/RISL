//! Transform that attempts to "pull" [ConstPtr] nodes out of nested regions into the region that
//! contains its originating value.
//!
//! If a [ConstPtr] node inside a nested region ([Switch] branch or [Loop] region) uses a region
//! argument as its input value, then the [ConstPtr] node is moved into the outer region and its
//! output value is passed into the nested region. This may be done repeatly until the pointer
//! originates from the same region as its originating value.
//!
//! This is a preparatory transform for variable-pointer-emulation, which expects the "originating
//! pointer" of a [Switch] or [Loop] node variable pointer output to already be available in the
//! region that contains the [Switch]/[Loop] node. This pass must therefore be run before
//! variable-pointer-emulation (which is part of the [memory_transform](super::memory_transform)
//! pass.

use crate::rvsdg::visit::region_nodes::RegionNodesVisitor;
use crate::rvsdg::{
    Connectivity, Node, NodeKind, Region, Rvsdg, SimpleNode, ValueInput, ValueOrigin, visit,
};
use crate::{Function, Module};

struct CandidateCollector<'a> {
    body_region: Region,
    queue: &'a mut Vec<Node>,
}

impl RegionNodesVisitor for CandidateCollector<'_> {
    fn visit_node(&mut self, rvsdg: &Rvsdg, node: Node) {
        if let NodeKind::Simple(SimpleNode::ConstPtr(const_ptr)) = rvsdg[node].kind() {
            let region = rvsdg[node].region();

            if region != self.body_region {
                let base_origin = const_ptr.base().origin;

                // Only nodes with an Argument origin are candidates for pulling out of a nested
                // region.
                if let ValueOrigin::Argument(_) = base_origin {
                    self.queue.push(node);
                }
            }
        }
        visit::region_nodes::visit_node(self, rvsdg, node);
    }
}

pub struct ConstPtrPuller {
    queue: Vec<Node>,
}

impl ConstPtrPuller {
    pub fn new() -> Self {
        Self { queue: Vec::new() }
    }

    pub fn pull_in_fn(&mut self, rvsdg: &mut Rvsdg, function: Function) {
        let fn_node = rvsdg
            .get_function_node(function)
            .expect("function not registered");
        let body_region = rvsdg[fn_node].expect_function().body_region();

        self.queue.clear();
        let mut collector = CandidateCollector {
            body_region,
            queue: &mut self.queue,
        };
        collector.visit_node(rvsdg, fn_node);

        while let Some(node) = self.queue.pop() {
            self.pull_node(rvsdg, node, body_region);
        }
    }

    fn pull_node(&mut self, rvsdg: &mut Rvsdg, node: Node, body_region: Region) {
        let region = rvsdg[node].region();

        if region == body_region {
            return;
        }

        let owner_node = rvsdg[region].owner();
        let parent_node_data = &rvsdg[owner_node];
        let outer_region = parent_node_data.region();

        let const_ptr = rvsdg[node].expect_const_ptr();
        let pointee_ty = const_ptr.pointee_ty();
        let base_origin = const_ptr.base().origin;

        // Resolve the origin of the base in the outer region.
        let ValueOrigin::Argument(arg) = base_origin else {
            // Ff the base origin is a node output, then it does not need to be pulled into the
            // outer region.
            return;
        };

        let outer_base_origin = match parent_node_data.kind() {
            NodeKind::Switch(_) => {
                // Switch input 0 is predicate, input i corresponds to argument i-1.
                let input = arg + 1;

                parent_node_data.value_inputs()[input as usize].origin
            }
            NodeKind::Loop(_) => {
                // Loop inputs map 1-to-1 to arguments.
                parent_node_data.value_inputs()[arg as usize].origin
            }
            _ => panic!("parent node of a nested region must be a Switch or Loop node"),
        };

        // Create a new ConstPtr in the outer region
        let base_ty = rvsdg.value_origin_ty(outer_region, outer_base_origin);
        let new_const_ptr = rvsdg.add_const_ptr(
            outer_region,
            pointee_ty,
            ValueInput {
                ty: base_ty,
                origin: outer_base_origin,
            },
        );

        // Add the output of the new ConstPtr as a new input to the owner node of the nested region
        // and find the corresponding region argument.
        let input_val =
            ValueInput::output(rvsdg[new_const_ptr].value_outputs()[0].ty, new_const_ptr, 0);
        let nested_arg = match rvsdg[owner_node].kind() {
            NodeKind::Switch(_) => {
                let parent_input = rvsdg.add_switch_input(owner_node, input_val);

                parent_input - 1
            }
            NodeKind::Loop(_) => {
                let parent_input = rvsdg.add_loop_input(owner_node, input_val);
                let loop_region = rvsdg[owner_node].expect_loop().loop_region();

                // Reconnect the new loop result to the new loop argument to ensure the value
                // is passed forward to subsequent loop iterations.
                rvsdg.reconnect_region_result(
                    loop_region,
                    parent_input + 1,
                    ValueOrigin::Argument(parent_input),
                );

                parent_input
            }
            _ => panic!("parent node of a nested region must be a Switch or Loop node"),
        };

        // Reconnect users of the old ConstPtr to the new region argument
        rvsdg.reconnect_value_users(
            region,
            ValueOrigin::Output {
                producer: node,
                output: 0,
            },
            ValueOrigin::Argument(nested_arg),
        );

        // Now that the old ConstPtr no longer has any users, we should be able to remove it.
        rvsdg.remove_node(node);

        // Push the new node to the queue to continue pulling it into further outer regions if
        // necessary
        self.queue.push(new_const_ptr);
    }
}

pub fn transform_entry_points(module: &Module, rvsdg: &mut Rvsdg) {
    let mut puller = ConstPtrPuller::new();

    for (entry_point, _) in module.entry_points.iter() {
        puller.pull_in_fn(rvsdg, entry_point.clone());
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::rvsdg::{Rvsdg, StateOrigin, ValueInput, ValueOrigin, ValueOutput};
    use crate::ty::{TY_DUMMY, TY_PREDICATE, TY_PTR_U32, TY_U32};
    use crate::{FnArg, FnSig, Function, Module, Symbol};

    #[test]
    fn test_const_ptr_pull_switch() {
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
                args: vec![
                    FnArg {
                        ty: TY_PREDICATE,
                        shader_io_binding: None,
                    },
                    FnArg {
                        ty: TY_U32,
                        shader_io_binding: None,
                    },
                ],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::argument(TY_U32, 1),
            ],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);

        rvsdg.reconnect_region_result(branch_0, 0, ValueOrigin::Argument(0));

        let branch_1 = rvsdg.add_switch_branch(switch_node);

        // The ConstPtr node
        let const_ptr_node = rvsdg.add_const_ptr(branch_1, TY_U32, ValueInput::argument(TY_U32, 0));

        // The ConstPtr node's user
        let load_node = rvsdg.add_op_load(
            branch_1,
            ValueInput::output(TY_PTR_U32, const_ptr_node, 0),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            branch_1,
            0,
            ValueOrigin::Output {
                producer: load_node,
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

        let mut puller = ConstPtrPuller::new();
        puller.pull_in_fn(&mut rvsdg, function);

        // Verification: Trace from the load node's pointer input.
        let load_ptr_input = rvsdg[load_node].value_inputs()[0];
        let ValueOrigin::Argument(arg) = load_ptr_input.origin else {
            panic!("load node should now use a region argument");
        };

        // Trace the switch input corresponding to this argument.
        // Argument N corresponds to input N+1.
        let switch_input = rvsdg[switch_node].value_inputs()[arg as usize + 1];
        let ValueOrigin::Output {
            producer: new_const_ptr_node,
            output: 0,
        } = switch_input.origin
        else {
            panic!("switch node input should connect to a new ConstPtr node");
        };

        assert!(rvsdg[new_const_ptr_node].is_const_ptr());
        assert_eq!(rvsdg[new_const_ptr_node].region(), region);
        assert_eq!(
            rvsdg[new_const_ptr_node].expect_const_ptr().base().origin,
            ValueOrigin::Argument(1)
        );
    }

    #[test]
    fn test_const_ptr_pull_loop() {
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
                args: vec![FnArg {
                    ty: TY_U32,
                    shader_io_binding: None,
                }],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let (loop_node, loop_region) =
            rvsdg.add_loop(region, vec![ValueInput::argument(TY_U32, 0)], None);

        // The ConstPtr node
        let const_ptr_node =
            rvsdg.add_const_ptr(loop_region, TY_U32, ValueInput::argument(TY_U32, 0));

        // Use ConstPtr node's user
        let load_node = rvsdg.add_op_load(
            loop_region,
            ValueInput::output(TY_PTR_U32, const_ptr_node, 0),
            StateOrigin::Argument,
        );

        let reentry_condition_node = rvsdg.add_const_bool(loop_region, true);
        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: reentry_condition_node,
                output: 0,
            },
        );

        // Pass something through loop result 1
        let dummy_val = rvsdg.add_const_u32(loop_region, 42);
        rvsdg.reconnect_region_result(
            loop_region,
            1,
            ValueOrigin::Output {
                producer: dummy_val,
                output: 0,
            },
        );

        let mut puller = ConstPtrPuller::new();
        puller.pull_in_fn(&mut rvsdg, function);

        // Load in loop should now use a region argument.
        let load_ptr_input = rvsdg[load_node].value_inputs()[0];
        let ValueOrigin::Argument(new_arg_idx) = load_ptr_input.origin else {
            panic!("loda node should use a region argument");
        };

        // This argument should correspond to a new loop input.
        let loop_input = rvsdg[loop_node].value_inputs()[new_arg_idx as usize];
        let ValueOrigin::Output {
            producer: new_const_ptr_node,
            output: 0,
        } = loop_input.origin
        else {
            panic!("loop input should connect to new ConstPtr");
        };

        assert!(rvsdg[new_const_ptr_node].is_const_ptr());
        assert_eq!(rvsdg[new_const_ptr_node].region(), region);
        assert_eq!(
            rvsdg[new_const_ptr_node].expect_const_ptr().base().origin,
            ValueOrigin::Argument(0)
        );
    }

    #[test]
    fn test_const_ptr_pull_nested() {
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
                args: vec![
                    FnArg {
                        ty: TY_PREDICATE,
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

        let switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::argument(TY_U32, 1),
            ],
            vec![],
            None,
        );

        let _branch_0 = rvsdg.add_switch_branch(switch_node);
        let branch_1 = rvsdg.add_switch_branch(switch_node);

        // Nested loop in branch_1
        let (loop_node, loop_region) =
            rvsdg.add_loop(branch_1, vec![ValueInput::argument(TY_U32, 0)], None);

        // The ConstPtr node.
        let const_ptr_node =
            rvsdg.add_const_ptr(loop_region, TY_U32, ValueInput::argument(TY_U32, 0));
        // The ConstPtr node's user
        let load_node = rvsdg.add_op_load(
            loop_region,
            ValueInput::output(TY_PTR_U32, const_ptr_node, 0),
            StateOrigin::Argument,
        );

        let reentry_condition_node = rvsdg.add_const_bool(loop_region, true);
        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: reentry_condition_node,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(loop_region, 1, ValueOrigin::Argument(0));

        let mut puller = ConstPtrPuller::new();
        puller.pull_in_fn(&mut rvsdg, function);

        let load_ptr_input = rvsdg[load_node].value_inputs()[0];
        let ValueOrigin::Argument(loop_arg) = load_ptr_input.origin else {
            panic!("load node should now use a loop-region argument");
        };

        // Trace loop input.
        let loop_input = rvsdg[loop_node].value_inputs()[loop_arg as usize];
        let ValueOrigin::Argument(arg) = loop_input.origin else {
            panic!("loop node input should now use a branch_1 argument");
        };

        // Trace switch input.
        let switch_input = rvsdg[switch_node].value_inputs()[arg as usize + 1];
        let ValueOrigin::Output {
            producer: new_const_ptr_node,
            output: 0,
        } = switch_input.origin
        else {
            panic!("switch node input should connect to a new ConstPtr node");
        };

        assert!(rvsdg[new_const_ptr_node].is_const_ptr());
        assert_eq!(
            rvsdg[new_const_ptr_node].region(),
            region,
            "should be pulled into function body"
        );
        assert_eq!(
            rvsdg[new_const_ptr_node].expect_const_ptr().base().origin,
            ValueOrigin::Argument(1)
        );
    }
}
