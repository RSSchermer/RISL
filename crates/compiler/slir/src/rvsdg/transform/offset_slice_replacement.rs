use rustc_hash::FxHashMap;

use crate::rvsdg::visit::region_nodes::RegionNodesVisitor;
use crate::rvsdg::{
    Connectivity, Node, NodeKind, Region, Rvsdg, SimpleNode, ValueInput, ValueOrigin, visit,
};
use crate::ty::TY_U32;
use crate::{BinaryOperator, Function, Module};

struct NodeCollector<'a, 'b> {
    get_offset_queue: &'a mut Vec<Node>,
    add_offset_queue: &'b mut Vec<Node>,
}

impl RegionNodesVisitor for NodeCollector<'_, '_> {
    fn visit_node(&mut self, rvsdg: &Rvsdg, node: Node) {
        use NodeKind::*;
        use SimpleNode::*;

        match rvsdg[node].kind() {
            Simple(OpGetSliceOffset(_)) => self.get_offset_queue.push(node),
            Simple(OpOffsetSlice(_)) => self.add_offset_queue.push(node),
            _ => (),
        }

        visit::region_nodes::visit_node(self, rvsdg, node);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PtrOffset {
    Zero,
    Value(ValueOrigin),
}

pub struct PtrOffsetReplacer {
    get_slice_offset_queue: Vec<Node>,
    offset_slice_queue: Vec<Node>,
    slice_offset_cache: FxHashMap<(Region, ValueOrigin), PtrOffset>,
}

impl PtrOffsetReplacer {
    pub fn new() -> Self {
        Self {
            get_slice_offset_queue: vec![],
            offset_slice_queue: vec![],
            slice_offset_cache: FxHashMap::default(),
        }
    }

    pub fn replace_in_fn(&mut self, rvsdg: &mut Rvsdg, function: Function) {
        self.slice_offset_cache.clear();

        let fn_node = rvsdg
            .get_function_node(function)
            .expect("function not registered");
        let body_region = rvsdg[fn_node].expect_function().body_region();

        let mut collector = NodeCollector {
            get_offset_queue: &mut self.get_slice_offset_queue,
            add_offset_queue: &mut self.offset_slice_queue,
        };

        collector.visit_region(rvsdg, body_region);

        // First replace all [OpGetPtrOffset] nodes
        while let Some(node) = self.get_slice_offset_queue.pop() {
            self.replace_op_get_slice_offset(rvsdg, node);
        }

        // Now that there should no longer be any [OpGetPtrOffset] nodes, "dissolve" all
        // [OpAddPtrOffset] nodes.
        while let Some(node) = self.offset_slice_queue.pop() {
            self.dissolve_op_offset_slice(rvsdg, node);
        }
    }

    fn replace_op_get_slice_offset(&mut self, rvsdg: &mut Rvsdg, node: Node) {
        let region = rvsdg[node].region();
        let data = rvsdg[node].expect_op_get_slice_offset();
        let ptr_origin = data.ptr_input().origin;
        let user_count = data.value_output().users.len();
        let ptr_offset = self.resolve_slice_offset(rvsdg, region, ptr_origin);
        let offset_origin = match ptr_offset {
            PtrOffset::Zero => {
                let zero = rvsdg.add_const_u32(region, 0);

                ValueOrigin::Output {
                    producer: zero,
                    output: 0,
                }
            }
            PtrOffset::Value(origin) => origin,
        };

        for i in (0..user_count).rev() {
            let user = rvsdg[node]
                .expect_op_get_slice_offset()
                .value_output()
                .users[i];

            rvsdg.reconnect_value_user(region, user, offset_origin);
        }

        rvsdg.remove_node(node);
    }

    fn dissolve_op_offset_slice(&mut self, rvsdg: &mut Rvsdg, node: Node) {
        let region = rvsdg[node].region();
        let data = rvsdg[node].expect_op_offset_slice();
        let ptr_origin = data.ptr_input().origin;
        let user_count = data.value_output().users.len();

        for i in (0..user_count).rev() {
            let user = rvsdg[node].expect_op_offset_slice().value_output().users[i];

            rvsdg.reconnect_value_user(region, user, ptr_origin);
        }

        rvsdg.remove_node(node);
    }

    fn resolve_slice_offset(
        &mut self,
        rvsdg: &mut Rvsdg,
        region: Region,
        ptr_origin: ValueOrigin,
    ) -> PtrOffset {
        if let Some(offset) = self.slice_offset_cache.get(&(region, ptr_origin)) {
            *offset
        } else {
            let offset = self.offset_for_origin(rvsdg, region, ptr_origin);

            self.slice_offset_cache.insert((region, ptr_origin), offset);

            offset
        }
    }

    fn offset_for_origin(
        &mut self,
        rvsdg: &mut Rvsdg,
        region: Region,
        ptr_origin: ValueOrigin,
    ) -> PtrOffset {
        match ptr_origin {
            ValueOrigin::Argument(argument) => self.offset_for_argument(rvsdg, region, argument),
            ValueOrigin::Output { producer, output } => {
                self.offset_for_output(rvsdg, producer, output)
            }
        }
    }

    fn offset_for_argument(
        &mut self,
        rvsdg: &mut Rvsdg,
        region: Region,
        ptr_argument: u32,
    ) -> PtrOffset {
        use NodeKind::*;
        

        let owner = rvsdg[region].owner();

        match rvsdg[owner].kind() {
            Switch(_) => self.offset_for_switch_branch_argument(rvsdg, region, ptr_argument),
            Loop(_) => self.offset_for_loop_region_argument(rvsdg, owner, ptr_argument),
            Function(_) => panic!("cannot track offset through function calls; inline first"),
            _ => unreachable!("node kind cannot own a region"),
        }
    }

    fn offset_for_output(&mut self, rvsdg: &mut Rvsdg, node: Node, ptr_output: u32) -> PtrOffset {
        use NodeKind::*;
        use SimpleNode::*;

        match rvsdg[node].kind() {
            Switch(_) => self.offset_for_switch_output(rvsdg, node, ptr_output),
            Loop(_) => self.offset_for_loop_output(rvsdg, node, ptr_output),
            Simple(OpOffsetSlice(_)) => self.offset_for_op_offset_slice(rvsdg, node),
            Simple(
                OpAlloca(_) | OpFieldPtr(_) | OpElementPtr(_) | OpVariantPtr(_) | OpExtractField(_)
                | OpExtractElement(_) | ConstPtr(_) | ConstFallback(_),
            ) => PtrOffset::Zero,
            _ => panic!("unsupported node kind"),
        }
    }

    fn offset_for_op_offset_slice(&mut self, rvsdg: &mut Rvsdg, node: Node) -> PtrOffset {
        let region = rvsdg[node].region();
        let data = rvsdg[node].expect_op_offset_slice();
        let ptr_origin = data.ptr_input().origin;
        let added_offset_origin = data.offset_input().origin;
        let prior_offset = self.resolve_slice_offset(rvsdg, region, ptr_origin);

        match prior_offset {
            PtrOffset::Zero => PtrOffset::Value(added_offset_origin),
            PtrOffset::Value(prior_offset_origin) => {
                let combined = rvsdg.add_op_binary(
                    region,
                    BinaryOperator::Add,
                    ValueInput {
                        ty: TY_U32,
                        origin: prior_offset_origin,
                    },
                    ValueInput {
                        ty: TY_U32,
                        origin: added_offset_origin,
                    },
                );

                PtrOffset::Value(ValueOrigin::Output {
                    producer: combined,
                    output: 0,
                })
            }
        }
    }

    fn offset_for_switch_output(
        &mut self,
        rvsdg: &mut Rvsdg,
        switch_node: Node,
        ptr_output: u32,
    ) -> PtrOffset {
        let data = rvsdg[switch_node].expect_switch();
        let branch_count = data.branches().len();

        let offset_output = rvsdg.add_switch_output(switch_node, TY_U32);

        for i in 0..branch_count {
            let branch = rvsdg[switch_node].expect_switch().branches()[i];
            let ptr_origin = rvsdg[branch].value_results()[ptr_output as usize].origin;
            let ptr_offset = self.resolve_slice_offset(rvsdg, branch, ptr_origin);

            let offset_origin = match ptr_offset {
                PtrOffset::Zero => {
                    let zero = rvsdg.add_const_u32(branch, 0);

                    ValueOrigin::Output {
                        producer: zero,
                        output: 0,
                    }
                }
                PtrOffset::Value(offset_origin) => offset_origin,
            };

            rvsdg.reconnect_region_result(branch, offset_output, offset_origin);
        }

        PtrOffset::Value(ValueOrigin::Output {
            producer: switch_node,
            output: offset_output,
        })
    }

    fn offset_for_switch_branch_argument(
        &mut self,
        rvsdg: &mut Rvsdg,
        branch: Region,
        ptr_argument: u32,
    ) -> PtrOffset {
        let ptr_index = ptr_argument + 1;
        let switch_node = rvsdg[branch].owner();
        let outer_region = rvsdg[switch_node].region();
        let ptr_origin = rvsdg[switch_node].value_inputs()[ptr_index as usize].origin;
        let outer_offset = self.resolve_slice_offset(rvsdg, outer_region, ptr_origin);

        let inner_offset = match outer_offset {
            PtrOffset::Zero => PtrOffset::Zero,
            PtrOffset::Value(origin) => {
                let input = rvsdg.add_switch_input(switch_node, ValueInput { ty: TY_U32, origin });

                PtrOffset::Value(ValueOrigin::Argument(input - 1))
            }
        };

        // Make sure we only do this for the first branch we encounter by adding a cache entry for
        // every branch; other branches should short-circuit to using this cached value.
        for branch in rvsdg[switch_node].expect_switch().branches() {
            self.slice_offset_cache
                .insert((*branch, ValueOrigin::Argument(ptr_argument)), inner_offset);
        }

        inner_offset
    }

    fn offset_for_loop_output(
        &mut self,
        rvsdg: &mut Rvsdg,
        loop_node: Node,
        ptr_output: u32,
    ) -> PtrOffset {
        let offset_input = self.create_loop_value(rvsdg, loop_node, ptr_output);

        PtrOffset::Value(ValueOrigin::Output {
            producer: loop_node,
            output: offset_input,
        })
    }

    fn offset_for_loop_region_argument(
        &mut self,
        rvsdg: &mut Rvsdg,
        loop_node: Node,
        ptr_argument: u32,
    ) -> PtrOffset {
        let offset_input = self.create_loop_value(rvsdg, loop_node, ptr_argument);

        PtrOffset::Value(ValueOrigin::Argument(offset_input))
    }

    fn create_loop_value(&mut self, rvsdg: &mut Rvsdg, loop_node: Node, ptr_input: u32) -> u32 {
        let outer_region = rvsdg[loop_node].region();
        let data = rvsdg[loop_node].expect_loop();
        let loop_region = data.loop_region();
        let input_ptr_origin = data.value_inputs()[ptr_input as usize].origin;
        let input_offset = self.resolve_slice_offset(rvsdg, outer_region, input_ptr_origin);

        let input_offset_origin = match input_offset {
            PtrOffset::Zero => {
                let zero = rvsdg.add_const_u32(outer_region, 0);

                ValueOrigin::Output {
                    producer: zero,
                    output: 0,
                }
            }
            PtrOffset::Value(origin) => origin,
        };

        let offset_input = rvsdg.add_loop_input(
            loop_node,
            ValueInput {
                ty: TY_U32,
                origin: input_offset_origin,
            },
        );
        let offset_result = offset_input + 1;

        // Add a cache entry for the new argument that we've just created, before resolving a
        // pointer offset for the new result we've created: the new result may resolve to that
        // argument, and we don't want to create another duplicate input.
        self.slice_offset_cache.insert(
            (loop_region, ValueOrigin::Argument(ptr_input)),
            PtrOffset::Value(ValueOrigin::Argument(offset_input)),
        );

        let ptr_result = ptr_input + 1;
        let inner_ptr_origin = rvsdg[loop_region].value_results()[ptr_result as usize].origin;
        let inner_offset = self.resolve_slice_offset(rvsdg, loop_region, inner_ptr_origin);

        match inner_offset {
            PtrOffset::Zero => {
                let zero = rvsdg.add_const_u32(loop_region, 0);

                rvsdg.reconnect_region_result(
                    loop_region,
                    offset_result,
                    ValueOrigin::Output {
                        producer: zero,
                        output: 0,
                    },
                );
            }
            PtrOffset::Value(origin) => {
                rvsdg.reconnect_region_result(loop_region, offset_result, origin);
            }
        }

        offset_input
    }
}

pub fn transform_entry_points(module: &Module, rvsdg: &mut Rvsdg) {
    let mut replacer = PtrOffsetReplacer::new();

    for (entry_point, _) in module.entry_points.iter() {
        replacer.replace_in_fn(rvsdg, entry_point);
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::rvsdg::{StateOrigin, ValueOutput, ValueUser};
    use crate::ty::{TY_DUMMY, TY_PREDICATE, TY_PTR_U32, TypeKind};
    use crate::{FnArg, FnSig, Symbol, thin_set};

    #[test]
    fn test_single_op_add_ptr_offset() {
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
        let array_ty = module.ty.register(TypeKind::Array {
            element_ty: TY_U32,
            count: 4,
            stride: 4,
        });
        let array_ptr_ty = module.ty.register(TypeKind::Ptr(array_ty));

        let array_node = rvsdg.add_op_alloca(region, array_ty);
        let offset_node = rvsdg.add_const_u32(region, 2);
        let add_offset_node = rvsdg.add_op_offset_slice(
            region,
            ValueInput::output(array_ptr_ty, array_node, 0),
            ValueInput::output(TY_U32, offset_node, 0),
        );
        let get_offset_node = rvsdg
            .add_op_get_ptr_offset(region, ValueInput::output(array_ptr_ty, add_offset_node, 0));
        let index_node = rvsdg.add_const_u32(region, 1);
        let add_node = rvsdg.add_op_binary(
            region,
            BinaryOperator::Add,
            ValueInput::output(TY_U32, get_offset_node, 0),
            ValueInput::output(TY_U32, index_node, 0),
        );
        let element_ptr_node = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(array_ptr_ty, add_offset_node, 0),
            ValueInput::output(TY_U32, add_node, 0),
        );
        let load_node = rvsdg.add_op_load(
            region,
            ValueInput::output(TY_PTR_U32, element_ptr_node, 0),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_node,
                output: 0,
            },
        );

        let mut replacer = PtrOffsetReplacer::new();

        replacer.replace_in_fn(&mut rvsdg, function);

        assert_eq!(
            rvsdg[add_node].expect_op_binary().lhs_input().origin,
            ValueOrigin::Output {
                producer: offset_node,
                output: 0,
            }
        );
        assert_eq!(
            rvsdg[element_ptr_node]
                .expect_op_element_ptr()
                .ptr_input()
                .origin,
            ValueOrigin::Output {
                producer: array_node,
                output: 0,
            }
        );

        assert!(!rvsdg.is_live_node(add_offset_node));
        assert!(!rvsdg.is_live_node(get_offset_node));
    }

    #[test]
    fn test_double_op_add_ptr_offset() {
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
        let array_ty = module.ty.register(TypeKind::Array {
            element_ty: TY_U32,
            count: 4,
            stride: 4,
        });
        let array_ptr_ty = module.ty.register(TypeKind::Ptr(array_ty));

        let array_node = rvsdg.add_op_alloca(region, array_ty);
        let offset_0_node = rvsdg.add_const_u32(region, 1);
        let add_offset_0_node = rvsdg.add_op_offset_slice(
            region,
            ValueInput::output(array_ptr_ty, array_node, 0),
            ValueInput::output(TY_U32, offset_0_node, 0),
        );
        let offset_1_node = rvsdg.add_const_u32(region, 1);
        let add_offset_1_node = rvsdg.add_op_offset_slice(
            region,
            ValueInput::output(array_ptr_ty, add_offset_0_node, 0),
            ValueInput::output(TY_U32, offset_1_node, 0),
        );
        let get_offset_node = rvsdg.add_op_get_ptr_offset(
            region,
            ValueInput::output(array_ptr_ty, add_offset_1_node, 0),
        );
        let index_node = rvsdg.add_const_u32(region, 1);
        let index_add_node = rvsdg.add_op_binary(
            region,
            BinaryOperator::Add,
            ValueInput::output(TY_U32, get_offset_node, 0),
            ValueInput::output(TY_U32, index_node, 0),
        );
        let element_ptr_node = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(array_ptr_ty, add_offset_1_node, 0),
            ValueInput::output(TY_U32, index_add_node, 0),
        );
        let load_node = rvsdg.add_op_load(
            region,
            ValueInput::output(TY_PTR_U32, element_ptr_node, 0),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_node,
                output: 0,
            },
        );

        let mut replacer = PtrOffsetReplacer::new();

        replacer.replace_in_fn(&mut rvsdg, function);

        let ValueOrigin::Output {
            producer: offset_add_node,
            output: 0,
        } = rvsdg[index_add_node].expect_op_binary().lhs_input().origin
        else {
            panic!(
                "the index-add-node's LHS should be connected to the first output of the offset-add-node"
            )
        };
        let offset_add_data = rvsdg[offset_add_node].expect_op_binary();

        assert_eq!(offset_add_data.operator(), BinaryOperator::Add);
        assert_eq!(
            offset_add_data.lhs_input().origin,
            ValueOrigin::Output {
                producer: offset_0_node,
                output: 0,
            }
        );
        assert_eq!(
            offset_add_data.rhs_input().origin,
            ValueOrigin::Output {
                producer: offset_1_node,
                output: 0,
            }
        );

        assert_eq!(
            rvsdg[element_ptr_node]
                .expect_op_element_ptr()
                .ptr_input()
                .origin,
            ValueOrigin::Output {
                producer: array_node,
                output: 0,
            }
        );

        assert!(!rvsdg.is_live_node(add_offset_0_node));
        assert!(!rvsdg.is_live_node(add_offset_1_node));
        assert!(!rvsdg.is_live_node(get_offset_node));
    }

    #[test]
    fn test_add_offset_then_add_offset_in_switch() {
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
                    ty: TY_PREDICATE,
                    shader_io_binding: None,
                }],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());
        let array_ty = module.ty.register(TypeKind::Array {
            element_ty: TY_U32,
            count: 4,
            stride: 4,
        });
        let array_ptr_ty = module.ty.register(TypeKind::Ptr(array_ty));

        let array_node = rvsdg.add_op_alloca(region, array_ty);
        let offset_0_node = rvsdg.add_const_u32(region, 1);
        let add_offset_0_node = rvsdg.add_op_offset_slice(
            region,
            ValueInput::output(array_ptr_ty, array_node, 0),
            ValueInput::output(TY_U32, offset_0_node, 0),
        );

        let switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(array_ptr_ty, add_offset_0_node, 0),
            ],
            vec![ValueOutput::new(array_ptr_ty)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);

        let offset_1_node = rvsdg.add_const_u32(branch_0, 1);
        let add_offset_1_node = rvsdg.add_op_offset_slice(
            branch_0,
            ValueInput::argument(array_ptr_ty, 0),
            ValueInput::output(TY_U32, offset_1_node, 0),
        );

        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: add_offset_1_node,
                output: 0,
            },
        );

        let branch_1 = rvsdg.add_switch_branch(switch_node);

        rvsdg.reconnect_region_result(branch_1, 0, ValueOrigin::Argument(0));

        let get_offset_node =
            rvsdg.add_op_get_ptr_offset(region, ValueInput::output(array_ptr_ty, switch_node, 0));
        let index_node = rvsdg.add_const_u32(region, 1);
        let index_add_node = rvsdg.add_op_binary(
            region,
            BinaryOperator::Add,
            ValueInput::output(TY_U32, get_offset_node, 0),
            ValueInput::output(TY_U32, index_node, 0),
        );
        let element_ptr_node = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(array_ptr_ty, switch_node, 0),
            ValueInput::output(TY_U32, index_add_node, 0),
        );
        let load_node = rvsdg.add_op_load(
            region,
            ValueInput::output(TY_PTR_U32, element_ptr_node, 0),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_node,
                output: 0,
            },
        );

        let mut replacer = PtrOffsetReplacer::new();

        replacer.replace_in_fn(&mut rvsdg, function);

        assert_eq!(
            rvsdg[switch_node].expect_switch().value_inputs().len(),
            3,
            "should have added an additional input to the switch node"
        );
        assert_eq!(
            rvsdg[switch_node].expect_switch().value_inputs()[2].origin,
            ValueOrigin::Output {
                producer: offset_0_node,
                output: 0,
            }
        );

        assert_eq!(
            &rvsdg[branch_0].value_arguments()[0].users,
            &thin_set![ValueUser::Result(0)],
            "the first branch's pointer argument should connect directly to the branch's first \
            result"
        );
        assert_eq!(
            rvsdg[branch_0].value_arguments()[1].users.len(),
            1,
            "the first branch's offset argument should have one user"
        );

        let ValueUser::Input {
            consumer: offset_add_node,
            input: 0,
        } = rvsdg[branch_0].value_arguments()[1].users[0]
        else {
            panic!("the first branch's the offset argument should have a user")
        };

        let offset_add_data = rvsdg[offset_add_node].expect_op_binary();

        assert_eq!(offset_add_data.operator(), BinaryOperator::Add);
        assert_eq!(offset_add_data.lhs_input().origin, ValueOrigin::Argument(1));
        assert_eq!(
            offset_add_data.rhs_input().origin,
            ValueOrigin::Output {
                producer: offset_1_node,
                output: 0,
            }
        );
        assert_eq!(
            &offset_add_data.value_output().users,
            &thin_set![ValueUser::Result(1)],
            "the offset-add-node's output should connect directly to the branch's second result"
        );

        assert_eq!(
            &rvsdg[branch_1].value_arguments()[0].users,
            &thin_set![ValueUser::Result(0)],
            "the second branch's pointer argument should connect directly to the branch's first \
            result"
        );
        assert_eq!(
            &rvsdg[branch_1].value_arguments()[1].users,
            &thin_set![ValueUser::Result(1)],
            "the second branch's new offset argument should connect directly to the branch's second \
            result"
        );

        assert_eq!(
            rvsdg[index_add_node].expect_op_binary().lhs_input().origin,
            ValueOrigin::Output {
                producer: switch_node,
                output: 1,
            },
            "the index-add-node's LHS should be connected to the offset output of the switch node"
        );
        assert_eq!(
            rvsdg[element_ptr_node]
                .expect_op_element_ptr()
                .ptr_input()
                .origin,
            ValueOrigin::Output {
                producer: switch_node,
                output: 0,
            },
            "the ptr-element-ptr-node's pointer input should be connected to the pointer output of \
            the switch node"
        );

        assert!(!rvsdg.is_live_node(add_offset_0_node));
        assert!(!rvsdg.is_live_node(add_offset_1_node));
        assert!(!rvsdg.is_live_node(get_offset_node));
    }

    #[test]
    fn test_add_offset_then_add_offset_in_loop() {
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
                    ty: TY_PREDICATE,
                    shader_io_binding: None,
                }],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());
        let array_ty = module.ty.register(TypeKind::Array {
            element_ty: TY_U32,
            count: 4,
            stride: 4,
        });
        let array_ptr_ty = module.ty.register(TypeKind::Ptr(array_ty));

        let array_node = rvsdg.add_op_alloca(region, array_ty);
        let offset_0_node = rvsdg.add_const_u32(region, 1);
        let add_offset_0_node = rvsdg.add_op_offset_slice(
            region,
            ValueInput::output(array_ptr_ty, array_node, 0),
            ValueInput::output(TY_U32, offset_0_node, 0),
        );

        let (loop_node, loop_region) = rvsdg.add_loop(
            region,
            vec![ValueInput::output(array_ptr_ty, add_offset_0_node, 0)],
            None,
        );

        let reentry_predicate_node = rvsdg.add_const_bool(loop_region, false);
        let offset_1_node = rvsdg.add_const_u32(loop_region, 1);
        let add_offset_1_node = rvsdg.add_op_offset_slice(
            loop_region,
            ValueInput::argument(array_ptr_ty, 0),
            ValueInput::output(TY_U32, offset_1_node, 0),
        );

        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: reentry_predicate_node,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            loop_region,
            1,
            ValueOrigin::Output {
                producer: add_offset_1_node,
                output: 0,
            },
        );

        let get_offset_node =
            rvsdg.add_op_get_ptr_offset(region, ValueInput::output(array_ptr_ty, loop_node, 0));
        let index_node = rvsdg.add_const_u32(region, 1);
        let index_add_node = rvsdg.add_op_binary(
            region,
            BinaryOperator::Add,
            ValueInput::output(TY_U32, get_offset_node, 0),
            ValueInput::output(TY_U32, index_node, 0),
        );
        let element_ptr_node = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(array_ptr_ty, loop_node, 0),
            ValueInput::output(TY_U32, index_add_node, 0),
        );
        let load_node = rvsdg.add_op_load(
            region,
            ValueInput::output(TY_PTR_U32, element_ptr_node, 0),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_node,
                output: 0,
            },
        );

        let mut replacer = PtrOffsetReplacer::new();

        replacer.replace_in_fn(&mut rvsdg, function);

        assert_eq!(
            rvsdg[loop_node].expect_loop().value_inputs().len(),
            2,
            "should have added an additional input to the loop node"
        );
        assert_eq!(
            rvsdg[loop_node].expect_loop().value_inputs()[1].origin,
            ValueOrigin::Output {
                producer: offset_0_node,
                output: 0,
            }
        );

        assert_eq!(
            &rvsdg[loop_region].value_arguments()[0].users,
            &thin_set![ValueUser::Result(1)],
            "the loop region's pointer argument should connect directly to the region's second \
            result"
        );
        assert_eq!(
            rvsdg[loop_region].value_arguments()[1].users.len(),
            1,
            "the lopp region's offset argument should have one user"
        );

        let ValueUser::Input {
            consumer: offset_add_node,
            input: 0,
        } = rvsdg[loop_region].value_arguments()[1].users[0]
        else {
            panic!("the first branch's the offset argument should have a user")
        };

        let offset_add_data = rvsdg[offset_add_node].expect_op_binary();

        assert_eq!(offset_add_data.operator(), BinaryOperator::Add);
        assert_eq!(offset_add_data.lhs_input().origin, ValueOrigin::Argument(1));
        assert_eq!(
            offset_add_data.rhs_input().origin,
            ValueOrigin::Output {
                producer: offset_1_node,
                output: 0,
            }
        );
        assert_eq!(
            &offset_add_data.value_output().users,
            &thin_set![ValueUser::Result(2)],
            "the offset-add-node's output should connect directly to the loop region's third result"
        );

        assert_eq!(
            rvsdg[index_add_node].expect_op_binary().lhs_input().origin,
            ValueOrigin::Output {
                producer: loop_node,
                output: 1,
            },
            "the index-add-node's LHS should be connected to the offset output of the loop node"
        );
        assert_eq!(
            rvsdg[element_ptr_node]
                .expect_op_element_ptr()
                .ptr_input()
                .origin,
            ValueOrigin::Output {
                producer: loop_node,
                output: 0,
            },
            "the ptr-element-ptr-node's pointer input should be connected to the pointer output of \
            the loop node"
        );

        assert!(!rvsdg.is_live_node(add_offset_0_node));
        assert!(!rvsdg.is_live_node(add_offset_1_node));
        assert!(!rvsdg.is_live_node(get_offset_node));
    }
}
