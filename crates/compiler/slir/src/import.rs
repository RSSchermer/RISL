use rustc_hash::FxHashMap;

use crate::cfg::{
    BasicBlock, BlockPosition, Cfg, ConstPtr, InlineConst, LocalBinding, RootIdentifier, Statement,
    StatementData, Terminator, Value,
};
use crate::ty::TypeKind;
use crate::{ConstantKind, Function, Module};

pub struct FunctionImporter {
    bb_mapping: FxHashMap<BasicBlock, BasicBlock>,
    local_value_mapping: FxHashMap<LocalBinding, LocalBinding>,
}

impl FunctionImporter {
    pub fn new() -> Self {
        Self {
            bb_mapping: FxHashMap::default(),
            local_value_mapping: FxHashMap::default(),
        }
    }

    pub fn import_function(
        &mut self,
        from: (&Module, &Cfg),
        to: (&mut Module, &mut Cfg),
        function: Function,
    ) {
        let (src_module, src_cfg) = from;
        let (dst_module, dst_cfg) = to;

        self.import_function_sig(src_module, dst_module, function);
        self.import_function_body((src_module, src_cfg), (dst_module, dst_cfg), function);
    }

    fn import_function_sig(
        &mut self,
        src_module: &Module,
        dst_module: &mut Module,
        function: Function,
    ) {
        let mut sig = src_module.fn_sigs[function].clone();

        sig.ty = dst_module.ty.register(TypeKind::Function(function));

        if let Some(ret_ty) = &mut sig.ret_ty {
            *ret_ty = dst_module.ty.import(&src_module.ty, *ret_ty);
        }

        for arg in &mut sig.args {
            arg.ty = dst_module.ty.import(&src_module.ty, arg.ty);
        }

        dst_module.fn_sigs.register(function, sig);
    }

    fn import_function_body(
        &mut self,
        (src_mod, src_cfg): (&Module, &Cfg),
        (dst_mod, dst_cfg): (&mut Module, &mut Cfg),
        function: Function,
    ) {
        self.bb_mapping.clear();
        self.local_value_mapping.clear();

        let src_body = src_cfg
            .get_function_body(function)
            .expect("function not registered in source CFG");
        let dst_body = dst_cfg.register_function(dst_mod, function);
        let dst_entry_block = dst_body.entry_block();

        for (src_arg, dst_arg) in src_body
            .argument_values()
            .iter()
            .zip(dst_body.argument_values())
        {
            self.local_value_mapping.insert(*src_arg, *dst_arg);
        }

        // First make sure all basic blocks are present in the destination CFG so we can connect the
        // branching terminators in any order.
        for src_bb in src_body.basic_blocks() {
            let dst_bb = if *src_bb == src_body.entry_block() {
                dst_entry_block
            } else {
                dst_cfg.add_basic_block(function)
            };

            self.bb_mapping.insert(*src_bb, dst_bb);
        }

        for src_bb in src_body.basic_blocks() {
            self.import_basic_block((src_mod, src_cfg), (dst_mod, dst_cfg), *src_bb);
        }
    }

    fn import_basic_block(
        &mut self,
        (src_mod, src_cfg): (&Module, &Cfg),
        (dst_mod, dst_cfg): (&mut Module, &mut Cfg),
        src_bb: BasicBlock,
    ) {
        let dst_bb = *self.bb_mapping.get(&src_bb).unwrap();

        for statement in src_cfg[src_bb].statements() {
            self.import_statement((src_mod, src_cfg), (dst_mod, dst_cfg), dst_bb, *statement);
        }

        let terminator = match src_cfg[src_bb].terminator() {
            Terminator::Branch(t) => {
                if let Some(selector) = t.selector() {
                    Terminator::branch_multiple(
                        self.dst_local_value(selector),
                        t.targets().iter().map(|&bb| self.dst_bb(bb)),
                    )
                } else {
                    Terminator::branch_single(self.dst_bb(t.targets()[0]))
                }
            }
            Terminator::Return(Some(value)) => {
                Terminator::return_value(self.dst_value(src_mod, (dst_mod, dst_cfg), *value))
            }
            Terminator::Return(None) => Terminator::return_void(),
        };

        dst_cfg.set_terminator(dst_bb, terminator);
    }

    fn import_statement(
        &mut self,
        (src_mod, src_cfg): (&Module, &Cfg),
        (dst_mod, dst_cfg): (&mut Module, &mut Cfg),
        dst_bb: BasicBlock,
        src_stmt: Statement,
    ) {
        match &src_cfg[src_stmt] {
            StatementData::Bind(_) => {
                self.import_stmt_bind((src_mod, src_cfg), (dst_mod, dst_cfg), dst_bb, src_stmt)
            }
            StatementData::Assign(_) => {
                self.import_stmt_assign((src_mod, src_cfg), (dst_mod, dst_cfg), dst_bb, src_stmt)
            }
            StatementData::Uninitialized(_) => {
                self.import_stmt_uninitialized(src_cfg, dst_cfg, dst_bb, src_stmt)
            }
            StatementData::OpAlloca(_) => {
                self.import_stmt_op_alloca(src_cfg, dst_cfg, dst_bb, src_stmt)
            }
            StatementData::OpLoad(_) => {
                self.import_stmt_op_load((src_mod, src_cfg), (dst_mod, dst_cfg), dst_bb, src_stmt)
            }
            StatementData::OpStore(_) => {
                self.import_stmt_op_store((src_mod, src_cfg), (dst_mod, dst_cfg), dst_bb, src_stmt)
            }
            StatementData::OpExtractValue(_) => self.import_stmt_op_extract_value(
                (src_mod, src_cfg),
                (dst_mod, dst_cfg),
                dst_bb,
                src_stmt,
            ),
            StatementData::OpPtrElementPtr(_) => self.import_stmt_op_ptr_element_ptr(
                (src_mod, src_cfg),
                (dst_mod, dst_cfg),
                dst_bb,
                src_stmt,
            ),
            StatementData::OpPtrVariantPtr(_) => self.import_stmt_op_ptr_variant_ptr(
                (src_mod, src_cfg),
                (dst_mod, dst_cfg),
                dst_bb,
                src_stmt,
            ),
            StatementData::OpGetDiscriminant(_) => self.import_stmt_op_get_discriminant(
                (src_mod, src_cfg),
                (dst_mod, dst_cfg),
                dst_bb,
                src_stmt,
            ),
            StatementData::OpSetDiscriminant(_) => self.import_stmt_op_set_discriminant(
                (src_mod, src_cfg),
                (dst_mod, dst_cfg),
                dst_bb,
                src_stmt,
            ),
            StatementData::OpOffsetSlicePtr(_) => self.import_stmt_op_offset_slice_ptr(
                (src_mod, src_cfg),
                (dst_mod, dst_cfg),
                dst_bb,
                src_stmt,
            ),
            StatementData::OpUnary(_) => {
                self.import_stmt_op_unary((src_mod, src_cfg), (dst_mod, dst_cfg), dst_bb, src_stmt)
            }
            StatementData::OpBinary(_) => {
                self.import_stmt_op_binary((src_mod, src_cfg), (dst_mod, dst_cfg), dst_bb, src_stmt)
            }
            StatementData::OpCall(_) => {
                self.import_stmt_op_call((src_mod, src_cfg), (dst_mod, dst_cfg), dst_bb, src_stmt)
            }
            StatementData::OpCallBuiltin(_) => self.import_stmt_op_call_builtin(
                (src_mod, src_cfg),
                (dst_mod, dst_cfg),
                dst_bb,
                src_stmt,
            ),
            StatementData::OpCaseToBranchPredicate(_) => self
                .import_stmt_op_case_to_branch_predicate(
                    (src_mod, src_cfg),
                    (dst_mod, dst_cfg),
                    dst_bb,
                    src_stmt,
                ),
            StatementData::OpBoolToBranchPredicate(_) => self
                .import_stmt_op_bool_to_branch_predicate(
                    (src_mod, src_cfg),
                    (dst_mod, dst_cfg),
                    dst_bb,
                    src_stmt,
                ),
            StatementData::OpConvertToU32(_) => self.import_stmt_op_convert_to_u32(
                (src_mod, src_cfg),
                (dst_mod, dst_cfg),
                dst_bb,
                src_stmt,
            ),
            StatementData::OpConvertToI32(_) => self.import_stmt_op_convert_to_i32(
                (src_mod, src_cfg),
                (dst_mod, dst_cfg),
                dst_bb,
                src_stmt,
            ),
            StatementData::OpConvertToF32(_) => self.import_stmt_op_convert_to_f32(
                (src_mod, src_cfg),
                (dst_mod, dst_cfg),
                dst_bb,
                src_stmt,
            ),
            StatementData::OpConvertToBool(_) => self.import_stmt_op_convert_to_f32(
                (src_mod, src_cfg),
                (dst_mod, dst_cfg),
                dst_bb,
                src_stmt,
            ),
        }
    }

    fn import_stmt_assign(
        &mut self,
        (src_mod, src_cfg): (&Module, &Cfg),
        (dst_mod, dst_cfg): (&mut Module, &mut Cfg),
        dst_bb: BasicBlock,
        src_stmt: Statement,
    ) {
        let src_data = src_cfg[src_stmt].expect_assign();
        let dst_local = self.dst_local_value(src_data.local_binding());
        let dst_value = self.dst_value(src_mod, (dst_mod, dst_cfg), src_data.value());

        dst_cfg.add_stmt_assign(dst_bb, BlockPosition::Append, dst_local, dst_value);
    }

    fn import_stmt_bind(
        &mut self,
        (src_mod, src_cfg): (&Module, &Cfg),
        (dst_mod, dst_cfg): (&mut Module, &mut Cfg),
        dst_bb: BasicBlock,
        src_stmt: Statement,
    ) {
        let src_data = src_cfg[src_stmt].expect_bind();
        let dst_value = self.dst_value(src_mod, (dst_mod, dst_cfg), src_data.value());

        let (_, dst_local) = dst_cfg.add_stmt_bind(dst_bb, BlockPosition::Append, dst_value);

        self.local_value_mapping
            .insert(src_data.local_binding(), dst_local);
    }

    fn import_stmt_uninitialized(
        &mut self,
        src_cfg: &Cfg,
        dst_cfg: &mut Cfg,
        dst_bb: BasicBlock,
        src_stmt: Statement,
    ) {
        let src_data = src_cfg[src_stmt].expect_uninitialized();
        let src_local = src_data.local_binding();
        let src_ty = src_cfg[src_local].ty();
        let dst_ty = dst_cfg.ty().import(src_cfg.ty(), src_ty);

        let (_, dst_local) = dst_cfg.add_stmt_uninitialized(dst_bb, BlockPosition::Append, dst_ty);

        self.local_value_mapping.insert(src_local, dst_local);
    }

    fn import_stmt_op_alloca(
        &mut self,
        src_cfg: &Cfg,
        dst_cfg: &mut Cfg,
        dst_bb: BasicBlock,
        src_stmt: Statement,
    ) {
        let src_data = src_cfg[src_stmt].expect_op_alloca();
        let ty = dst_cfg.ty().import(src_cfg.ty(), src_data.ty());
        let (_, dst_result) = dst_cfg.add_stmt_op_alloca(dst_bb, BlockPosition::Append, ty);

        self.local_value_mapping
            .insert(src_data.result(), dst_result);
    }

    fn import_stmt_op_load(
        &mut self,
        (src_mod, src_cfg): (&Module, &Cfg),
        (dst_mod, dst_cfg): (&mut Module, &mut Cfg),
        dst_bb: BasicBlock,
        src_stmt: Statement,
    ) {
        let src_data = src_cfg[src_stmt].expect_op_load();
        let dst_pointer = self.dst_value(src_mod, (dst_mod, dst_cfg), src_data.pointer());

        let (_, dst_result) = dst_cfg.add_stmt_op_load(dst_bb, BlockPosition::Append, dst_pointer);

        self.local_value_mapping
            .insert(src_data.result(), dst_result);
    }

    fn import_stmt_op_store(
        &mut self,
        (src_mod, src_cfg): (&Module, &Cfg),
        (dst_mod, dst_cfg): (&mut Module, &mut Cfg),
        dst_bb: BasicBlock,
        src_stmt: Statement,
    ) {
        let src_data = src_cfg[src_stmt].expect_op_store();
        let dst_pointer = self.dst_value(src_mod, (dst_mod, dst_cfg), src_data.pointer());
        let dst_value = self.dst_value(src_mod, (dst_mod, dst_cfg), src_data.value());

        dst_cfg.add_stmt_op_store(dst_bb, BlockPosition::Append, dst_pointer, dst_value);
    }

    fn import_stmt_op_extract_value(
        &mut self,
        (src_mod, src_cfg): (&Module, &Cfg),
        (dst_mod, dst_cfg): (&mut Module, &mut Cfg),
        dst_bb: BasicBlock,
        src_stmt: Statement,
    ) {
        let src_data = src_cfg[src_stmt].expect_op_extract_value();
        let dst_aggregate = self.dst_value(src_mod, (dst_mod, dst_cfg), src_data.aggregate());
        let dst_indices = src_data
            .indices()
            .iter()
            .map(|v| self.dst_value(src_mod, (dst_mod, dst_cfg), *v))
            .collect::<Vec<_>>();

        let (_, dst_result) = dst_cfg.add_stmt_op_extract_value(
            dst_bb,
            BlockPosition::Append,
            dst_aggregate,
            dst_indices,
        );

        self.local_value_mapping
            .insert(src_data.result(), dst_result);
    }

    fn import_stmt_op_ptr_element_ptr(
        &mut self,
        (src_mod, src_cfg): (&Module, &Cfg),
        (dst_mod, dst_cfg): (&mut Module, &mut Cfg),
        dst_bb: BasicBlock,
        src_stmt: Statement,
    ) {
        let src_data = src_cfg[src_stmt].expect_op_ptr_element_ptr();
        let dst_pointer = self.dst_value(src_mod, (dst_mod, dst_cfg), src_data.pointer());
        let dst_indices = src_data
            .indices()
            .iter()
            .map(|v| self.dst_value(src_mod, (dst_mod, dst_cfg), *v))
            .collect::<Vec<_>>();

        let (_, dst_result) = dst_cfg.add_stmt_op_ptr_element_ptr(
            dst_bb,
            BlockPosition::Append,
            dst_pointer,
            dst_indices,
        );

        self.local_value_mapping
            .insert(src_data.result(), dst_result);
    }

    fn import_stmt_op_ptr_variant_ptr(
        &mut self,
        (src_mod, src_cfg): (&Module, &Cfg),
        (dst_mod, dst_cfg): (&mut Module, &mut Cfg),
        dst_bb: BasicBlock,
        src_stmt: Statement,
    ) {
        let src_data = src_cfg[src_stmt].expect_op_ptr_variant_ptr();
        let dst_pointer = self.dst_value(src_mod, (dst_mod, dst_cfg), src_data.pointer());
        let dst_variant_index = src_data.variant_index();

        let (_, dst_result) = dst_cfg.add_stmt_op_ptr_variant_ptr(
            dst_bb,
            BlockPosition::Append,
            dst_pointer,
            dst_variant_index,
        );

        self.local_value_mapping
            .insert(src_data.result(), dst_result);
    }

    fn import_stmt_op_get_discriminant(
        &mut self,
        (src_mod, src_cfg): (&Module, &Cfg),
        (dst_mod, dst_cfg): (&mut Module, &mut Cfg),
        dst_bb: BasicBlock,
        src_stmt: Statement,
    ) {
        let src_data = src_cfg[src_stmt].expect_op_get_discriminant();
        let dst_pointer = self.dst_value(src_mod, (dst_mod, dst_cfg), src_data.pointer());

        let (_, dst_result) =
            dst_cfg.add_stmt_op_get_discriminant(dst_bb, BlockPosition::Append, dst_pointer);

        self.local_value_mapping
            .insert(src_data.result(), dst_result);
    }

    fn import_stmt_op_set_discriminant(
        &mut self,
        (src_mod, src_cfg): (&Module, &Cfg),
        (dst_mod, dst_cfg): (&mut Module, &mut Cfg),
        dst_bb: BasicBlock,
        src_stmt: Statement,
    ) {
        let src_data = src_cfg[src_stmt].expect_op_set_discriminant();
        let dst_pointer = self.dst_value(src_mod, (dst_mod, dst_cfg), src_data.pointer());
        let variant_index = src_data.variant_index();

        dst_cfg.add_stmt_op_set_discriminant(
            dst_bb,
            BlockPosition::Append,
            dst_pointer,
            variant_index,
        );
    }

    fn import_stmt_op_offset_slice_ptr(
        &mut self,
        (src_mod, src_cfg): (&Module, &Cfg),
        (dst_mod, dst_cfg): (&mut Module, &mut Cfg),
        dst_bb: BasicBlock,
        src_stmt: Statement,
    ) {
        let src_data = src_cfg[src_stmt].expect_op_offset_slice_ptr();
        let dst_pointer = self.dst_value(src_mod, (dst_mod, dst_cfg), src_data.pointer());
        let dst_offset = self.dst_value(src_mod, (dst_mod, dst_cfg), src_data.offset());

        let (_, result) = dst_cfg.add_stmt_op_offset_slice(
            dst_bb,
            BlockPosition::Append,
            dst_pointer,
            dst_offset,
        );

        self.local_value_mapping.insert(src_data.result(), result);
    }

    fn import_stmt_op_unary(
        &mut self,
        (src_mod, src_cfg): (&Module, &Cfg),
        (dst_mod, dst_cfg): (&mut Module, &mut Cfg),
        dst_bb: BasicBlock,
        src_stmt: Statement,
    ) {
        let src_data = src_cfg[src_stmt].expect_op_unary();
        let operator = src_data.operator();
        let dst_operand = self.dst_value(src_mod, (dst_mod, dst_cfg), src_data.operand());

        let (_, result) =
            dst_cfg.add_stmt_op_unary(dst_bb, BlockPosition::Append, operator, dst_operand);

        self.local_value_mapping.insert(src_data.result(), result);
    }

    fn import_stmt_op_binary(
        &mut self,
        (src_mod, src_cfg): (&Module, &Cfg),
        (dst_mod, dst_cfg): (&mut Module, &mut Cfg),
        dst_bb: BasicBlock,
        src_stmt: Statement,
    ) {
        let src_data = src_cfg[src_stmt].expect_op_binary();
        let operator = src_data.operator();
        let dst_lhs = self.dst_value(src_mod, (dst_mod, dst_cfg), src_data.lhs());
        let dst_rhs = self.dst_value(src_mod, (dst_mod, dst_cfg), src_data.rhs());

        let (_, result) =
            dst_cfg.add_stmt_op_binary(dst_bb, BlockPosition::Append, operator, dst_lhs, dst_rhs);

        self.local_value_mapping.insert(src_data.result(), result);
    }

    fn import_stmt_op_call(
        &mut self,
        (src_mod, src_cfg): (&Module, &Cfg),
        (dst_mod, dst_cfg): (&mut Module, &mut Cfg),
        dst_bb: BasicBlock,
        src_stmt: Statement,
    ) {
        let src_data = src_cfg[src_stmt].expect_op_call();
        let callee = src_data.callee();
        let ret_ty = src_data.maybe_result().map(|v| {
            let src_ty = src_cfg[v].ty();

            dst_cfg.ty().import(src_cfg.ty(), src_ty)
        });
        let dst_arguments = src_data
            .arguments()
            .iter()
            .map(|v| self.dst_value(src_mod, (dst_mod, dst_cfg), *v))
            .collect::<Vec<_>>();

        let (_, result) =
            dst_cfg.add_stmt_op_call(dst_bb, BlockPosition::Append, callee, ret_ty, dst_arguments);

        if let Some(result) = result {
            self.local_value_mapping
                .insert(src_data.maybe_result().unwrap(), result);
        }
    }

    fn import_stmt_op_call_builtin(
        &mut self,
        (src_mod, src_cfg): (&Module, &Cfg),
        (dst_mod, dst_cfg): (&mut Module, &mut Cfg),
        dst_bb: BasicBlock,
        src_stmt: Statement,
    ) {
        let src_data = src_cfg[src_stmt].expect_op_call_builtin();
        let callee = src_data.callee().clone();
        let dst_arguments = src_data
            .arguments()
            .iter()
            .map(|v| self.dst_value(src_mod, (dst_mod, dst_cfg), *v))
            .collect::<Vec<_>>();

        let (_, result) =
            dst_cfg.add_stmt_op_call_builtin(dst_bb, BlockPosition::Append, callee, dst_arguments);

        if let Some(result) = result {
            self.local_value_mapping
                .insert(src_data.result().unwrap(), result);
        }
    }

    fn import_stmt_op_case_to_branch_predicate(
        &mut self,
        (src_mod, src_cfg): (&Module, &Cfg),
        (dst_mod, dst_cfg): (&mut Module, &mut Cfg),
        dst_bb: BasicBlock,
        src_stmt: Statement,
    ) {
        let src_data = src_cfg[src_stmt].expect_op_case_to_branch_predicate();
        let dst_value = self.dst_value(src_mod, (dst_mod, dst_cfg), src_data.value());
        let cases = src_data.cases().iter().copied();

        let (_, result) = dst_cfg.add_stmt_op_case_to_branch_selector(
            dst_bb,
            BlockPosition::Append,
            dst_value,
            cases,
        );

        self.local_value_mapping.insert(src_data.result(), result);
    }

    fn import_stmt_op_bool_to_branch_predicate(
        &mut self,
        (src_mod, src_cfg): (&Module, &Cfg),
        (dst_mod, dst_cfg): (&mut Module, &mut Cfg),
        dst_bb: BasicBlock,
        src_stmt: Statement,
    ) {
        let src_data = src_cfg[src_stmt].expect_op_bool_to_branch_predicate();
        let dst_value = self.dst_value(src_mod, (dst_mod, dst_cfg), src_data.value());

        let (_, result) =
            dst_cfg.add_stmt_op_bool_to_branch_selector(dst_bb, BlockPosition::Append, dst_value);

        self.local_value_mapping.insert(src_data.result(), result);
    }

    fn import_stmt_op_convert_to_u32(
        &mut self,
        (src_mod, src_cfg): (&Module, &Cfg),
        (dst_mod, dst_cfg): (&mut Module, &mut Cfg),
        dst_bb: BasicBlock,
        src_stmt: Statement,
    ) {
        let src_data = src_cfg[src_stmt].expect_op_convert_to_u32();
        let dst_operand = self.dst_value(src_mod, (dst_mod, dst_cfg), src_data.value());

        let (_, result) =
            dst_cfg.add_stmt_op_convert_to_u32(dst_bb, BlockPosition::Append, dst_operand);

        self.local_value_mapping.insert(src_data.result(), result);
    }

    fn import_stmt_op_convert_to_i32(
        &mut self,
        (src_mod, src_cfg): (&Module, &Cfg),
        (dst_mod, dst_cfg): (&mut Module, &mut Cfg),
        dst_bb: BasicBlock,
        src_stmt: Statement,
    ) {
        let src_data = src_cfg[src_stmt].expect_op_convert_to_i32();
        let dst_operand = self.dst_value(src_mod, (dst_mod, dst_cfg), src_data.value());

        let (_, result) =
            dst_cfg.add_stmt_op_convert_to_i32(dst_bb, BlockPosition::Append, dst_operand);

        self.local_value_mapping.insert(src_data.result(), result);
    }

    fn import_stmt_op_convert_to_f32(
        &mut self,
        (src_mod, src_cfg): (&Module, &Cfg),
        (dst_mod, dst_cfg): (&mut Module, &mut Cfg),
        dst_bb: BasicBlock,
        src_stmt: Statement,
    ) {
        let src_data = src_cfg[src_stmt].expect_op_convert_to_f32();
        let dst_operand = self.dst_value(src_mod, (dst_mod, dst_cfg), src_data.value());

        let (_, result) =
            dst_cfg.add_stmt_op_convert_to_f32(dst_bb, BlockPosition::Append, dst_operand);

        self.local_value_mapping.insert(src_data.result(), result);
    }

    fn import_stmt_op_convert_to_bool(
        &mut self,
        (src_mod, src_cfg): (&Module, &Cfg),
        (dst_mod, dst_cfg): (&mut Module, &mut Cfg),
        dst_bb: BasicBlock,
        src_stmt: Statement,
    ) {
        let src_data = src_cfg[src_stmt].expect_op_convert_to_bool();
        let dst_operand = self.dst_value(src_mod, (dst_mod, dst_cfg), src_data.value());

        let (_, result) =
            dst_cfg.add_stmt_op_convert_to_bool(dst_bb, BlockPosition::Append, dst_operand);

        self.local_value_mapping.insert(src_data.result(), result);
    }

    fn dst_bb(&self, src_bb: BasicBlock) -> BasicBlock {
        *self
            .bb_mapping
            .get(&src_bb)
            .expect("basic block mapping should be registered earlier")
    }

    fn dst_local_value(&self, src_local: LocalBinding) -> LocalBinding {
        *self
            .local_value_mapping
            .get(&src_local)
            .expect("local mapping should be registered earlier")
    }

    fn dst_value(
        &self,
        src_mod: &Module,
        (dst_mod, dst_cfg): (&mut Module, &mut Cfg),
        src_value: Value,
    ) -> Value {
        match src_value {
            Value::Local(local) => Value::Local(self.dst_local_value(local)),
            value @ Value::InlineConst(InlineConst::Ptr(ptr)) => match ptr.root_identifier() {
                RootIdentifier::Local(local) => {
                    let local = self.dst_local_value(local);
                    let inline_const = InlineConst::Ptr(ConstPtr::local_binding(dst_cfg, local));

                    inline_const.into()
                }
                RootIdentifier::Constant(constant) => {
                    // Import the constant if it hasn't already been registered
                    if !dst_mod.constants.contains(constant) {
                        let data = &src_mod.constants[constant];

                        match data.kind() {
                            ConstantKind::ByteData(bytes) => dst_mod.constants.register_byte_data(
                                constant,
                                data.ty(),
                                bytes.clone(),
                            ),
                            ConstantKind::Expression => todo!(),
                            ConstantKind::Overridable(_) => panic!(
                                "overridable constants cannot be imported from other modules"
                            ),
                        }
                    };

                    // The `Constant` token is module-independent, so now that we've ensured that
                    // the constant has been imported, we can simply return the source value
                    // unaltered and it will resolve to the imported constant.
                    value
                }
                _ => panic!("imported functions may not reference module globals"),
            },
            value @ _ => value,
        }
    }
}
