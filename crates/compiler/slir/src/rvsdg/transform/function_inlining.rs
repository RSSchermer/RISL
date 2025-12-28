use std::collections::VecDeque;

use rustc_hash::FxHashSet;

use crate::Module;
use crate::rvsdg::transform::region_replication::replicate_region;
use crate::rvsdg::visit::bottom_up::{BottomUpVisitor, visit_node_bottom_up};
use crate::rvsdg::{
    Connectivity, Node, NodeKind, Region, Rvsdg, StateOrigin, ValueInput, ValueOrigin, ValueUser,
};

/// Adds missing dependencies to the function we're inlining into.
///
/// Call [route] to ensure a dependency is made available to a specific region, either by finding
/// a pre-existing argument that represents the dependency, or by creating a new input and routing
/// the dependency from the function's root region.
struct DependencyResolver<'a> {
    rvsdg: &'a mut Rvsdg,
    dependencies: Vec<ValueInput>,
    region_owner_stack: Vec<Node>,
}

impl<'a> DependencyResolver<'a> {
    fn new(rvsdg: &'a mut Rvsdg, function_node: Node) -> Self {
        let dependencies = rvsdg[function_node]
            .expect_function()
            .dependencies()
            .to_vec();

        Self {
            rvsdg,
            dependencies,
            region_owner_stack: Vec::new(),
        }
    }

    fn route(mut self, region: Region) -> Vec<ValueInput> {
        self.visit_region_owner(region);

        self.dependencies
    }

    fn visit_region_owner(&mut self, region: Region) {
        let owner = self.rvsdg[region].owner();

        match self.rvsdg[owner].kind() {
            NodeKind::Switch(_) | NodeKind::Loop(_) => {
                self.region_owner_stack.push(owner);
                self.visit_region_owner(self.rvsdg[owner].region());
            }
            NodeKind::Function(_) => {
                self.resolve_global_dependencies(owner);
                self.consume_owner_stack();
            }
            _ => unreachable!("node kind cannot own a region"),
        }
    }

    fn resolve_global_dependencies(&mut self, dest_function_node: Node) {
        for dependency in &mut self.dependencies {
            let ValueOrigin::Output {
                producer,
                output: 0,
            } = dependency.origin
            else {
                panic!("expect dependency input to connect to output `0` or a producer node");
            };

            // Note that `add_function_dependency` will only insert a dependency if the
            // `dest_function_node` (the function we're inlining into) does not already have that
            // dependency; if it does not yet have the dependency, then the dependency is inserted
            // at an argument index that is greater than the index of the pre-existing dependencies.
            // This is important because otherwise the argument indices in the mapping we are
            // building might get invalidated by the argument indices shifting as a result of
            // insertions.
            let arg_index = self
                .rvsdg
                .add_function_dependency(dest_function_node, producer);

            dependency.origin = ValueOrigin::Argument(arg_index);
        }
    }

    fn consume_owner_stack(&mut self) {
        if let Some(owner) = self.region_owner_stack.pop() {
            for dependency in &mut self.dependencies {
                let input = self
                    .rvsdg
                    .get_input_index(owner, dependency.origin)
                    .unwrap_or_else(|| match self.rvsdg[owner].kind() {
                        NodeKind::Switch(_) => self.rvsdg.add_switch_input(owner, *dependency),
                        NodeKind::Loop(_) => self.rvsdg.add_loop_input(owner, *dependency),
                        _ => unreachable!("only switch and loop have been pushed to the stack"),
                    });

                let arg = match self.rvsdg[owner].kind() {
                    NodeKind::Switch(_) => input - 1,
                    NodeKind::Loop(_) => input,
                    _ => unreachable!("only switch and loop have been pushed to the stack"),
                };

                dependency.origin = ValueOrigin::Argument(arg);
            }

            self.consume_owner_stack();
        }
    }
}

fn resolve_dependencies(rvsdg: &mut Rvsdg, function_node: Node, region: Region) -> Vec<ValueInput> {
    let mut resolver = DependencyResolver::new(rvsdg, function_node);

    resolver.route(region)
}

pub fn inline_function(module: &mut Module, rvsdg: &mut Rvsdg, call_node: Node) {
    let node_data = &rvsdg[call_node];

    let dst_region = node_data.region();

    let value_output_count = node_data.value_outputs().len();
    let call_data = node_data.expect_op_call();

    let function = call_data.resolve_fn(module);
    let function_node = rvsdg
        .get_function_node(function)
        .expect("cannot call an unregistered function");
    let function_node_data = rvsdg[function_node].expect_function();
    let src_region = function_node_data.body_region();
    let dependency_count = function_node_data.dependencies().len();
    let function_argument_count = call_data.argument_inputs().len();
    let region_argument_count = dependency_count + function_argument_count;

    // The state origin to which the inlined region's state argument maps in the destination region.
    let state_argument_mapping = call_data.state().map(|state| state.origin);

    // We also have to construct a mapping that maps each of the non-state arguments to origins in
    // the destination region. These will first map the inlined function's dependencies (if any),
    // followed by the mappings for the call arguments (as that is how a function region's arguments
    // are organized).
    let mut argument_mapping = Vec::with_capacity(region_argument_count);

    // Resolve and add the dependencies.
    argument_mapping.extend(
        resolve_dependencies(rvsdg, function_node, dst_region)
            .iter()
            .map(|input| input.origin),
    );

    // Then add mappings for the call arguments.
    argument_mapping.extend(
        rvsdg[call_node]
            .expect_op_call()
            .argument_inputs()
            .iter()
            .map(|input| input.origin),
    );

    let result_mapping = replicate_region(
        module,
        rvsdg,
        src_region,
        dst_region,
        argument_mapping,
        state_argument_mapping,
    );

    // Region replication does not connect the outputs used by the results of the original function
    // body, so we do that now using the mapping returned by region replication.
    for i in 0..value_output_count {
        rvsdg.reconnect_value_users(
            dst_region,
            ValueOrigin::Output {
                producer: call_node,
                output: i as u32,
            },
            result_mapping[i],
        );
    }

    // Now that we've disconnected the users of all of the call node's value outputs, the call node
    // should be "dead", so we can remove it from the graph.
    rvsdg.remove_node(call_node);
}

struct CallNodeCollector {
    seen: FxHashSet<Node>,
    queue: VecDeque<Node>,
}

impl CallNodeCollector {
    fn new() -> Self {
        Self {
            seen: Default::default(),
            queue: VecDeque::new(),
        }
    }
}

impl BottomUpVisitor for CallNodeCollector {
    fn visit_node(&mut self, rvsdg: &Rvsdg, node: Node) {
        if rvsdg[node].is_op_call() && self.seen.insert(node) {
            self.queue.push_back(node);
        }

        visit_node_bottom_up(self, rvsdg, node);
    }
}

/// For all entry points in the given `module`, finds all call nodes and inlines the called
/// function, iteratively inlining any new call operations amongst the inlined nodes until the entry
/// points no longer contain any call operations for user-defined functions.
pub fn transform_entry_points(module: &mut Module, rvsdg: &mut Rvsdg) {
    let entry_points = module
        .entry_points
        .iter()
        .map(|(f, _)| f)
        .collect::<Vec<_>>();

    let mut collector = CallNodeCollector::new();

    for entry_point in entry_points {
        if let Some(function_node) = rvsdg.get_function_node(entry_point) {
            let body_region = rvsdg[function_node].expect_function().body_region();

            collector.visit_region(rvsdg, body_region);

            // Inline the nodes currently in the queue, then check for new call nodes and add them
            // to the queue; keep iterating until the queue is empty.
            while !collector.queue.is_empty() {
                while let Some(node) = collector.queue.pop_front() {
                    inline_function(module, rvsdg, node);
                }

                // Inlining may have added more call nodes to the function, so search the body
                // again.
                // TODO: revisiting every single node in the body region again is not exactly the
                // most efficient way to find new call nodes, though I'm also not sure how much this
                // contributes to the overall compile time; doing the simple thing for now, but we
                // may want to measure this at some point and perhaps find a way to restrict the
                // search to only the newly inlined nodes.
                collector.visit_region(rvsdg, body_region);
            }

            // Any function dependencies that were inlined will now be unused, so clean up the entry
            // point's dependencies.
            rvsdg.remove_unused_dependencies(function_node);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::rvsdg::StateUser;
    use crate::ty::{TY_DUMMY, TY_U32, TypeKind};
    use crate::{BinaryOperator, EntryPoint, EntryPointKind, FnArg, FnSig, Function, Symbol};

    #[test]
    fn test_inline_function() {
        let mut module = Module::new(Symbol::from_ref(""));

        let inline_target = Function {
            name: Symbol::from_ref("inline_target"),
            module: Symbol::from_ref(""),
        };
        let inline_target_ty = module.ty.register(TypeKind::Function(inline_target));

        module.fn_sigs.register(
            inline_target,
            FnSig {
                name: Default::default(),
                ty: inline_target_ty,
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
                ret_ty: Some(TY_U32),
            },
        );

        let inline_dst = Function {
            name: Symbol::from_ref("inline_dst"),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            inline_dst,
            FnSig {
                name: Default::default(),
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
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        // Build inline_target
        let (inline_target_node, src_region) =
            rvsdg.register_function(&module, inline_target, iter::empty());

        let inline_target_node_0 = rvsdg.add_op_binary(
            src_region,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 0),
            ValueInput::argument(TY_U32, 1),
        );

        rvsdg.reconnect_region_result(
            src_region,
            0,
            ValueOrigin::Output {
                producer: inline_target_node_0,
                output: 0,
            },
        );

        // Build inline_dst
        let (_, dst_region) = rvsdg.register_function(&module, inline_dst, [inline_target_node]);

        let inline_dst_node_0 = rvsdg.add_const_u32(dst_region, 5);
        let inline_dst_node_1 = rvsdg.add_op_call(
            &module,
            dst_region,
            ValueInput::argument(inline_target_ty, 0),
            [
                ValueInput::argument(TY_U32, 1),
                ValueInput::output(TY_U32, inline_dst_node_0, 0),
            ],
            StateOrigin::Argument,
        );
        let inline_dst_node_2 = rvsdg.add_op_binary(
            dst_region,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 2),
            ValueInput::output(TY_U32, inline_dst_node_1, 0),
        );

        rvsdg.reconnect_region_result(
            dst_region,
            0,
            ValueOrigin::Output {
                producer: inline_dst_node_2,
                output: 0,
            },
        );

        inline_function(&mut module, &mut rvsdg, inline_dst_node_1);

        // The first argument to the region is the function dependency, which should no longer be
        // in use after inlining
        assert!(rvsdg[dst_region].value_arguments()[0].users.is_empty());

        // The second argument should now connect to input `0` of the inlined "add" node.
        let arg_1_users = &rvsdg[dst_region].value_arguments()[1].users;

        assert_eq!(arg_1_users.len(), 1);

        let arg_1_user = arg_1_users[0];

        let ValueUser::Input {
            consumer: inlined_node,
            input,
        } = arg_1_user
        else {
            panic!("expected user to be a node input");
        };

        assert_eq!(
            rvsdg[inlined_node].expect_op_binary().operator(),
            BinaryOperator::Add
        );
        assert_eq!(input, 0);

        // Node `0` in the destination region should now connect to input `1` of the inlined "add"
        // node.
        let inline_dst_node_0_users = &rvsdg[inline_dst_node_0].expect_const_u32().output().users;

        assert_eq!(inline_dst_node_0_users.len(), 1);
        assert_eq!(
            inline_dst_node_0_users[0],
            ValueUser::Input {
                consumer: inlined_node,
                input: 1
            }
        );

        // The second input of node `2` is the destination region should now connect to the output
        // of the inlined "add" node.
        assert_eq!(
            rvsdg[inline_dst_node_2].value_inputs()[1].origin,
            ValueOrigin::Output {
                producer: inlined_node,
                output: 0
            }
        );

        // The first input of the inlined node should now connect to the second argument of the
        // destination region.
        assert_eq!(
            rvsdg[inlined_node].value_inputs()[0].origin,
            ValueOrigin::Argument(1)
        );

        // The second input of the inlined node should now connect to the output of node `0` in the
        // destination region.
        assert_eq!(
            rvsdg[inlined_node].value_inputs()[1].origin,
            ValueOrigin::Output {
                producer: inline_dst_node_0,
                output: 0,
            }
        );

        // The output of the inlined node should now connect to the second input of node `2` in the
        // destination region.
        let inlined_node_users = &rvsdg[inlined_node].value_outputs()[0].users;

        assert_eq!(inlined_node_users.len(), 1);
        assert_eq!(
            inlined_node_users[0],
            ValueUser::Input {
                consumer: inline_dst_node_2,
                input: 1
            }
        );

        // After inlining the only call node in the region, the destination region should no longer
        // contain any call nodes.
        assert!(
            !rvsdg[dst_region]
                .nodes()
                .into_iter()
                .copied()
                .any(|n| rvsdg[n].is_op_call())
        );
    }

    #[test]
    fn test_inline_function_stateful() {
        let mut module = Module::new(Symbol::from_ref(""));

        let inline_target = Function {
            name: Symbol::from_ref("inline_target"),
            module: Symbol::from_ref(""),
        };
        let inline_target_ty = module.ty.register(TypeKind::Function(inline_target));
        let ty_ptr_u32 = module.ty.register(TypeKind::Ptr(TY_U32));

        module.fn_sigs.register(
            inline_target,
            FnSig {
                name: Default::default(),
                ty: inline_target_ty,
                args: vec![FnArg {
                    ty: ty_ptr_u32,
                    shader_io_binding: None,
                }],
                ret_ty: None,
            },
        );

        let inline_dst = Function {
            name: Symbol::from_ref("inline_dst"),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            inline_dst,
            FnSig {
                name: Default::default(),
                ty: TY_DUMMY,
                args: vec![],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        // Build inline_target
        let (inline_target_node, src_region) =
            rvsdg.register_function(&module, inline_target, iter::empty());

        let inline_target_node_0 = rvsdg.add_const_u32(src_region, 5);

        rvsdg.add_op_store(
            src_region,
            ValueInput::argument(ty_ptr_u32, 0),
            ValueInput::output(TY_U32, inline_target_node_0, 0),
            StateOrigin::Argument,
        );

        // Build inline_dst
        let (_, dst_region) = rvsdg.register_function(&module, inline_dst, [inline_target_node]);

        let inline_dst_node_0 = rvsdg.add_op_alloca(dst_region, TY_U32);
        let inline_dst_node_1 = rvsdg.add_op_call(
            &module,
            dst_region,
            ValueInput::argument(inline_target_ty, 0),
            [ValueInput::output(ty_ptr_u32, inline_dst_node_0, 0)],
            StateOrigin::Argument,
        );

        inline_function(&mut module, &mut rvsdg, inline_dst_node_1);

        // The destination region's state argument should now be connected to an inlined "store"
        // node.
        let StateUser::Node(inlined_node) = *rvsdg[dst_region].state_argument() else {
            panic!("expected the user of the state argument to be a node");
        };

        rvsdg[inlined_node].expect_op_store();

        assert_eq!(
            *rvsdg[dst_region].state_result(),
            StateOrigin::Node(inlined_node),
            "the destination region's state result should now be connected to the inlined node"
        );
    }

    #[test]
    fn test_inline_entry_points_exhaustive() {
        let mut module = Module::new(Symbol::from_ref(""));

        let add_1 = Function {
            name: Symbol::from_ref("add_1"),
            module: Symbol::from_ref(""),
        };
        let add_1_ty = module.ty.register(TypeKind::Function(add_1));

        module.fn_sigs.register(
            add_1,
            FnSig {
                name: Default::default(),
                ty: add_1_ty,
                args: vec![FnArg {
                    ty: TY_U32,
                    shader_io_binding: None,
                }],
                ret_ty: Some(TY_U32),
            },
        );

        let add_2 = Function {
            name: Symbol::from_ref("add_2"),
            module: Symbol::from_ref(""),
        };
        let add_2_ty = module.ty.register(TypeKind::Function(add_2));

        module.fn_sigs.register(
            add_2,
            FnSig {
                name: Default::default(),
                ty: add_2_ty,
                args: vec![FnArg {
                    ty: TY_U32,
                    shader_io_binding: None,
                }],
                ret_ty: Some(TY_U32),
            },
        );

        let entry_point = Function {
            name: Symbol::from_ref("entry_point"),
            module: Symbol::from_ref(""),
        };
        let entry_point_ty = module.ty.register(TypeKind::Function(entry_point));

        module.fn_sigs.register(
            entry_point,
            FnSig {
                name: Default::default(),
                ty: entry_point_ty,
                args: vec![],
                ret_ty: Some(TY_U32),
            },
        );
        module.entry_points.register(
            entry_point,
            EntryPoint {
                name: Symbol::from_ref("main"),
                kind: EntryPointKind::Compute(1, 1, 1),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        // Build add_1
        let (add_1_node, add_1_region) = rvsdg.register_function(&module, add_1, iter::empty());

        let add_1_node_0 = rvsdg.add_const_u32(add_1_region, 1);
        let add_1_node_1 = rvsdg.add_op_binary(
            add_1_region,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 0),
            ValueInput::output(TY_U32, add_1_node_0, 0),
        );

        rvsdg.reconnect_region_result(
            add_1_region,
            0,
            ValueOrigin::Output {
                producer: add_1_node_1,
                output: 0,
            },
        );

        // Build add_2
        let (add_2_node, add_2_region) = rvsdg.register_function(&module, add_2, [add_1_node]);

        let add_2_node_0 = rvsdg.add_op_call(
            &module,
            add_2_region,
            ValueInput::argument(add_1_ty, 0),
            [ValueInput::argument(TY_U32, 1)],
            StateOrigin::Argument,
        );
        let add_2_node_1 = rvsdg.add_op_call(
            &module,
            add_2_region,
            ValueInput::argument(add_1_ty, 0),
            [ValueInput::output(TY_U32, add_2_node_0, 0)],
            StateOrigin::Node(add_2_node_0),
        );

        rvsdg.reconnect_region_result(
            add_2_region,
            0,
            ValueOrigin::Output {
                producer: add_2_node_1,
                output: 0,
            },
        );

        // Build entry_point
        let (entry_point_node, entry_point_region) =
            rvsdg.register_function(&module, entry_point, [add_2_node]);

        let entry_point_node_0 = rvsdg.add_const_u32(entry_point_region, 10);
        let entry_point_node_1 = rvsdg.add_op_call(
            &module,
            entry_point_region,
            ValueInput::argument(add_2_ty, 0),
            [ValueInput::output(TY_U32, entry_point_node_0, 0)],
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            entry_point_region,
            0,
            ValueOrigin::Output {
                producer: entry_point_node_1,
                output: 0,
            },
        );

        transform_entry_points(&mut module, &mut rvsdg);

        assert_eq!(
            rvsdg[entry_point_region]
                .nodes()
                .into_iter()
                .filter(|n| rvsdg[**n].is_op_call())
                .count(),
            0,
            "entry_point function should no longer contain any call nodes"
        );

        assert!(
            rvsdg[entry_point_node]
                .expect_function()
                .dependencies()
                .is_empty(),
            "entry_point function should no longer have any dependencies"
        );
    }
}
