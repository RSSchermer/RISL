use rustc_hash::FxHashSet;

use crate::rvsdg::visit::value_flow::ValueFlowVisitor;
use crate::rvsdg::{Node, NodeKind, Region, Rvsdg, SimpleNode, ValueOrigin, ValueUser};
use crate::ty::{TY_BOOL, TY_F32, TY_I32, TY_U32};
use crate::{AllocId, ConstantKind, Function, Module};

struct Job {
    node: Node,
    alloc_id: AllocId,
    offset: usize,
}

pub struct ConstDependencyInliner {
    visited: FxHashSet<(Region, ValueUser)>,
    queue: Vec<Job>,
}

impl ConstDependencyInliner {
    pub fn new() -> Self {
        Self {
            visited: FxHashSet::default(),
            queue: Vec::new(),
        }
    }

    pub fn inline_in_fn(&mut self, module: &Module, rvsdg: &mut Rvsdg, function: Function) {
        let fn_node = rvsdg
            .get_function_node(function)
            .expect("function not registered");
        let fn_data = rvsdg[fn_node].expect_function();
        let body_region = fn_data.body_region();
        let dependencies = fn_data.dependencies().to_vec();

        for (i, dependency) in dependencies.iter().enumerate() {
            if let ValueOrigin::Output { producer, .. } = dependency.origin {
                if let NodeKind::Constant(constant_node) = rvsdg[producer].kind() {
                    let constant = constant_node.constant();
                    let constant_data = &module.constants[constant];

                    if let ConstantKind::ByteData(alloc_id, offset) = constant_data.kind() {
                        let ty = constant_data.ty();

                        if ty == TY_I32 || ty == TY_U32 || ty == TY_F32 || ty == TY_BOOL {
                            self.visited.clear();
                            self.queue.clear();

                            let mut visitor = JobCollector {
                                inliner: self,
                                alloc_id: *alloc_id,
                                offset: *offset,
                            };

                            visitor.visit_region_argument(rvsdg, body_region, i as u32);

                            self.inline_op_loads(module, rvsdg);
                        }
                    }
                }
            }
        }
    }

    fn inline_op_loads(&mut self, module: &Module, rvsdg: &mut Rvsdg) {
        for job in self.queue.drain(..) {
            let alloc_id = job.alloc_id;
            let offset = job.offset;
            let op_load_node = job.node;

            let allocation = &module.allocations[alloc_id];
            let region = rvsdg[op_load_node].region();

            let op_load = rvsdg[op_load_node].expect_op_load();
            let ty = op_load.value_output().ty;

            let inlined_node = if ty == TY_I32 {
                let val =
                    i32::from_le_bytes(allocation.bytes[offset..offset + 4].try_into().unwrap());

                rvsdg.add_const_i32(region, val)
            } else if ty == TY_U32 {
                let val =
                    u32::from_le_bytes(allocation.bytes[offset..offset + 4].try_into().unwrap());

                rvsdg.add_const_u32(region, val)
            } else if ty == TY_F32 {
                let val =
                    f32::from_le_bytes(allocation.bytes[offset..offset + 4].try_into().unwrap());

                rvsdg.add_const_f32(region, val)
            } else if ty == TY_BOOL {
                let val = allocation.bytes[offset] != 0;

                rvsdg.add_const_bool(region, val)
            } else {
                unreachable!("unsupported scalar type: {:?}", ty);
            };

            rvsdg.reconnect_value_users(
                region,
                ValueOrigin::Output {
                    producer: op_load_node,
                    output: 0,
                },
                ValueOrigin::Output {
                    producer: inlined_node,
                    output: 0,
                },
            );

            rvsdg.remove_node(op_load_node);
        }
    }
}

struct JobCollector<'a> {
    inliner: &'a mut ConstDependencyInliner,
    alloc_id: AllocId,
    offset: usize,
}

impl ValueFlowVisitor for JobCollector<'_> {
    fn should_visit(&mut self, region: Region, user: ValueUser) -> bool {
        self.inliner.visited.insert((region, user))
    }

    fn visit_region_result(&mut self, _rvsdg: &Rvsdg, _region: Region, _result: u32) {
        // We don't try to track the pointer past region results.
    }

    fn visit_value_input(&mut self, rvsdg: &Rvsdg, node: Node, input: u32) {
        use NodeKind::*;
        use SimpleNode::*;

        match rvsdg[node].kind() {
            Switch(n) => {
                if input > 0 {
                    let argument = input - 1;
                    for branch in n.branches() {
                        self.visit_region_argument(rvsdg, *branch, argument);
                    }
                }
            }
            Loop(n) => {
                let loop_region = n.loop_region();
                let loop_result_origin =
                    rvsdg[loop_region].value_results()[input as usize + 1].origin;

                // We only follow the value-flow into the loop if the loop-value is loop-invariant,
                // that is, the loop-result connects directly to its correponding argument.
                if loop_result_origin == ValueOrigin::Argument(input) {
                    self.visit_region_argument(rvsdg, loop_region, input);
                    self.visit_value_output(rvsdg, node, input);
                }
            }
            Simple(ValueProxy(_)) => {
                self.visit_value_output(rvsdg, node, 0);
            }
            Simple(OpLoad(_)) => {
                self.inliner.queue.push(Job {
                    node,
                    alloc_id: self.alloc_id,
                    offset: self.offset,
                });
            }
            _ => {
                // All other nodes act as terminators for the search; we don't try to track the
                // pointer through other node kinds.
            }
        }
    }
}

pub fn transform_entry_points(module: &Module, rvsdg: &mut Rvsdg) {
    let mut inliner = ConstDependencyInliner::new();

    let entry_points = module.entry_points.iter().map(|e| e.0).collect::<Vec<_>>();

    for entry_point in entry_points {
        inliner.inline_in_fn(module, rvsdg, entry_point);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rvsdg::{Connectivity, StateOrigin, ValueInput, ValueOutput};
    use crate::ty::{TY_DUMMY, TY_PREDICATE, TypeKind};
    use crate::{Allocation, Constant, FnSig, Symbol};

    #[test]
    fn load_in_body_region() {
        let mut module = Module::new(Symbol::from_ref("m"));
        let function = Function {
            name: Symbol::from_ref("f"),
            module: Symbol::from_ref("m"),
        };

        let alloc_id = module.allocations.register(Allocation {
            bytes: vec![
                42, 0, 0, 0, // 42 (i32)
                255, 255, 255, 255, // u32::MAX
                0, 0, 128, 63, // 1.0 (f32)
                1,  // true (bool)
            ],
        });

        let c_i32 = Constant {
            name: Symbol::from_ref("c_i32"),
            module: Symbol::from_ref("m"),
        };
        module
            .constants
            .register_byte_data(c_i32, TY_I32, alloc_id, 0);

        let c_u32 = Constant {
            name: Symbol::from_ref("c_u32"),
            module: Symbol::from_ref("m"),
        };
        module
            .constants
            .register_byte_data(c_u32, TY_U32, alloc_id, 4);

        let c_f32 = Constant {
            name: Symbol::from_ref("c_f32"),
            module: Symbol::from_ref("m"),
        };
        module
            .constants
            .register_byte_data(c_f32, TY_F32, alloc_id, 8);

        let c_bool = Constant {
            name: Symbol::from_ref("c_bool"),
            module: Symbol::from_ref("m"),
        };
        module
            .constants
            .register_byte_data(c_bool, TY_BOOL, alloc_id, 12);

        module.fn_sigs.register(
            function,
            FnSig {
                name: Symbol::from_ref("f"),
                ty: TY_DUMMY,
                args: vec![],
                ret_ty: Some(TY_I32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let n_i32 = rvsdg.register_constant(&module, c_i32);
        let n_u32 = rvsdg.register_constant(&module, c_u32);
        let n_f32 = rvsdg.register_constant(&module, c_f32);
        let n_bool = rvsdg.register_constant(&module, c_bool);

        let (_, region) = rvsdg.register_function(&module, function, [n_i32, n_u32, n_f32, n_bool]);

        let load_i32 = rvsdg.add_op_load(
            region,
            ValueInput::argument(rvsdg.ty().register(TypeKind::Ptr(TY_I32)), 0),
            StateOrigin::Argument,
        );
        let load_u32 = rvsdg.add_op_load(
            region,
            ValueInput::argument(rvsdg.ty().register(TypeKind::Ptr(TY_U32)), 1),
            StateOrigin::Argument,
        );
        let load_f32 = rvsdg.add_op_load(
            region,
            ValueInput::argument(rvsdg.ty().register(TypeKind::Ptr(TY_F32)), 2),
            StateOrigin::Argument,
        );
        let load_bool = rvsdg.add_op_load(
            region,
            ValueInput::argument(rvsdg.ty().register(TypeKind::Ptr(TY_BOOL)), 3),
            StateOrigin::Argument,
        );

        let user_i32 = rvsdg.add_value_proxy(region, ValueInput::output(TY_I32, load_i32, 0));
        let user_u32 = rvsdg.add_value_proxy(region, ValueInput::output(TY_U32, load_u32, 0));
        let user_f32 = rvsdg.add_value_proxy(region, ValueInput::output(TY_F32, load_f32, 0));
        let user_bool = rvsdg.add_value_proxy(region, ValueInput::output(TY_BOOL, load_bool, 0));

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: user_i32,
                output: 0,
            },
        );

        let mut inliner = ConstDependencyInliner::new();

        inliner.inline_in_fn(&module, &mut rvsdg, function);

        // Verify i32
        let ValueOrigin::Output {
            producer: p_i32, ..
        } = rvsdg[user_i32].value_inputs()[0].origin
        else {
            panic!("expected output origin for i32");
        };
        assert_eq!(rvsdg[p_i32].expect_const_i32().value(), 42);
        assert!(!rvsdg.is_live_node(load_i32));

        // Verify u32
        let ValueOrigin::Output {
            producer: p_u32, ..
        } = rvsdg[user_u32].value_inputs()[0].origin
        else {
            panic!("expected output origin for u32");
        };
        assert_eq!(rvsdg[p_u32].expect_const_u32().value(), u32::MAX);
        assert!(!rvsdg.is_live_node(load_u32));

        // Verify f32
        let ValueOrigin::Output {
            producer: p_f32, ..
        } = rvsdg[user_f32].value_inputs()[0].origin
        else {
            panic!("expected output origin for f32");
        };
        assert_eq!(rvsdg[p_f32].expect_const_f32().value(), 1.0);
        assert!(!rvsdg.is_live_node(load_f32));

        // Verify bool
        let ValueOrigin::Output {
            producer: p_bool, ..
        } = rvsdg[user_bool].value_inputs()[0].origin
        else {
            panic!("expected output origin for bool");
        };
        assert_eq!(rvsdg[p_bool].expect_const_bool().value(), true);
        assert!(!rvsdg.is_live_node(load_bool));
    }

    #[test]
    fn load_in_switch_branch() {
        let mut module = Module::new(Symbol::from_ref("m"));
        let function = Function {
            name: Symbol::from_ref("f"),
            module: Symbol::from_ref("m"),
        };

        let alloc_id = module.allocations.register(Allocation {
            bytes: vec![42, 0, 0, 0],
        });
        let constant = Constant {
            name: Symbol::from_ref("c"),
            module: Symbol::from_ref("m"),
        };
        module
            .constants
            .register_byte_data(constant, TY_I32, alloc_id, 0);

        module.fn_sigs.register(
            function,
            FnSig {
                name: Symbol::from_ref("f"),
                ty: TY_DUMMY,
                args: vec![],
                ret_ty: Some(TY_I32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let const_node = rvsdg.register_constant(&module, constant);
        let (_, region) = rvsdg.register_function(&module, function, [const_node]);

        let ptr_ty = rvsdg.ty().register(TypeKind::Ptr(TY_I32));
        let pred_node = rvsdg.add_const_predicate(region, 0);
        let switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::output(TY_PREDICATE, pred_node, 0),
                ValueInput::argument(ptr_ty, 0),
            ],
            vec![ValueOutput::new(TY_I32)],
            None,
        );

        let branch = rvsdg.add_switch_branch(switch_node);
        let load_node = rvsdg.add_op_load(
            branch,
            ValueInput::argument(ptr_ty, 0),
            StateOrigin::Argument,
        );
        rvsdg.reconnect_region_result(
            branch,
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

        let mut inliner = ConstDependencyInliner::new();

        inliner.inline_in_fn(&module, &mut rvsdg, function);

        let ValueOrigin::Output { producer, .. } = rvsdg[branch].value_results()[0].origin else {
            panic!("expected output origin in branch");
        };

        assert_eq!(rvsdg[producer].expect_const_i32().value(), 42);

        assert!(!rvsdg.is_live_node(load_node));
    }

    #[test]
    fn load_in_loop_region_loop_invariant() {
        let mut module = Module::new(Symbol::from_ref("m"));
        let function = Function {
            name: Symbol::from_ref("f"),
            module: Symbol::from_ref("m"),
        };

        let alloc_id = module.allocations.register(Allocation {
            bytes: vec![42, 0, 0, 0],
        });
        let constant = Constant {
            name: Symbol::from_ref("c"),
            module: Symbol::from_ref("m"),
        };
        module
            .constants
            .register_byte_data(constant, TY_I32, alloc_id, 0);

        module.fn_sigs.register(
            function,
            FnSig {
                name: Symbol::from_ref("f"),
                ty: TY_DUMMY,
                args: vec![],
                ret_ty: Some(TY_I32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let const_node = rvsdg.register_constant(&module, constant);
        let (_, region) = rvsdg.register_function(&module, function, [const_node]);

        let ptr_ty = rvsdg.ty().register(TypeKind::Ptr(TY_I32));
        let dummy_node = rvsdg.add_const_i32(region, 0);
        let (loop_node, _) = rvsdg.add_loop(
            region,
            vec![
                ValueInput::argument(ptr_ty, 0),
                ValueInput::output(TY_I32, dummy_node, 0),
            ],
            Some(StateOrigin::Argument),
        );

        let loop_region = rvsdg[loop_node].expect_loop().loop_region();
        let reentry_condition = rvsdg.add_const_bool(loop_region, false);
        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: reentry_condition,
                output: 0,
            },
        );

        // Loop invariant for the pointer
        rvsdg.reconnect_region_result(loop_region, 1, ValueOrigin::Argument(0));

        // Load inside loop region
        let load_node = rvsdg.add_op_load(
            loop_region,
            ValueInput::argument(ptr_ty, 0),
            StateOrigin::Argument,
        );
        rvsdg.reconnect_region_result(
            loop_region,
            2,
            ValueOrigin::Output {
                producer: load_node,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: loop_node,
                output: 1, // Output index 1 is the i32 result
            },
        );

        let mut inliner = ConstDependencyInliner::new();

        inliner.inline_in_fn(&module, &mut rvsdg, function);

        let ValueOrigin::Output { producer, .. } = rvsdg[loop_region].value_results()[2].origin
        else {
            panic!("expected output origin in loop region");
        };

        assert_eq!(rvsdg[producer].expect_const_i32().value(), 42);

        assert!(!rvsdg.is_live_node(load_node));
    }
}
