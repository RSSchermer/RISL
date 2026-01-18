use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::cfg::{
    BasicBlock, BlockPosition, Cfg, ConstPtr, InlineConst, IntrinsicOp, LocalBinding,
    RootIdentifier, Statement, StatementData, Terminator, Value,
};
use crate::intrinsic::Intrinsic;
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
                // Note: we have a separate import routine for OpAlloca statements, even though it's
                // currently implemented as an `IntrinsicOp` and all other IntrinsicOps are imported
                // with `import_stmt_intrinsic_op`. This is because the `OpAlloca` statement is the
                // one intrinsic operation that introduces a new type that also needs
                // mapping/importing.
                //
                // TODO: should OpAlloca not be implemented via IntrinsicOp just to clearly enforce
                // its "specialness" in this regard? In all other cases in the code-base it's
                // currently fine to treat an OpAlloca like any other intrinsic operation.
                self.import_stmt_op_alloca(src_cfg, dst_cfg, dst_bb, src_stmt)
            }
            StatementData::OpLoad(op) => {
                self.import_stmt_intrinsic_op(src_mod, (dst_mod, dst_cfg), dst_bb, op)
            }
            StatementData::OpStore(op) => {
                self.import_stmt_intrinsic_op(src_mod, (dst_mod, dst_cfg), dst_bb, op)
            }
            StatementData::OpExtractField(op) => {
                self.import_stmt_intrinsic_op(src_mod, (dst_mod, dst_cfg), dst_bb, op)
            }
            StatementData::OpExtractElement(op) => {
                self.import_stmt_intrinsic_op(src_mod, (dst_mod, dst_cfg), dst_bb, op)
            }
            StatementData::OpFieldPtr(op) => {
                self.import_stmt_intrinsic_op(src_mod, (dst_mod, dst_cfg), dst_bb, op)
            }
            StatementData::OpElementPtr(op) => {
                self.import_stmt_intrinsic_op(src_mod, (dst_mod, dst_cfg), dst_bb, op)
            }
            StatementData::OpVariantPtr(op) => {
                self.import_stmt_intrinsic_op(src_mod, (dst_mod, dst_cfg), dst_bb, op)
            }
            StatementData::OpGetDiscriminant(op) => {
                self.import_stmt_intrinsic_op(src_mod, (dst_mod, dst_cfg), dst_bb, op)
            }
            StatementData::OpSetDiscriminant(op) => {
                self.import_stmt_intrinsic_op(src_mod, (dst_mod, dst_cfg), dst_bb, op)
            }
            StatementData::OpOffsetSlice(op) => {
                self.import_stmt_intrinsic_op(src_mod, (dst_mod, dst_cfg), dst_bb, op)
            }
            StatementData::OpUnary(op) => {
                self.import_stmt_intrinsic_op(src_mod, (dst_mod, dst_cfg), dst_bb, op)
            }
            StatementData::OpBinary(op) => {
                self.import_stmt_intrinsic_op(src_mod, (dst_mod, dst_cfg), dst_bb, op)
            }
            StatementData::OpCaseToBranchSelector(op) => {
                self.import_stmt_intrinsic_op(src_mod, (dst_mod, dst_cfg), dst_bb, op)
            }
            StatementData::OpBoolToBranchSelector(op) => {
                self.import_stmt_intrinsic_op(src_mod, (dst_mod, dst_cfg), dst_bb, op)
            }
            StatementData::OpConvertToU32(op) => {
                self.import_stmt_intrinsic_op(src_mod, (dst_mod, dst_cfg), dst_bb, op)
            }
            StatementData::OpConvertToI32(op) => {
                self.import_stmt_intrinsic_op(src_mod, (dst_mod, dst_cfg), dst_bb, op)
            }
            StatementData::OpConvertToF32(op) => {
                self.import_stmt_intrinsic_op(src_mod, (dst_mod, dst_cfg), dst_bb, op)
            }
            StatementData::OpConvertToBool(op) => {
                self.import_stmt_intrinsic_op(src_mod, (dst_mod, dst_cfg), dst_bb, op)
            }
            StatementData::OpArrayLength(op) => {
                self.import_stmt_intrinsic_op(src_mod, (dst_mod, dst_cfg), dst_bb, op)
            }
            StatementData::OpCall(_) => {
                self.import_stmt_op_call((src_mod, src_cfg), (dst_mod, dst_cfg), dst_bb, src_stmt)
            }
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

    fn import_stmt_intrinsic_op<T>(
        &mut self,
        src_mod: &Module,
        (dst_mod, dst_cfg): (&mut Module, &mut Cfg),
        dst_bb: BasicBlock,
        src_stmt: &IntrinsicOp<T>,
    ) where
        T: Intrinsic + Clone,
        StatementData: From<IntrinsicOp<T>>,
    {
        let arguments: SmallVec<[Value; 6]> = src_stmt
            .arguments()
            .iter()
            .map(|v| self.dst_value(src_mod, (dst_mod, dst_cfg), *v))
            .collect();

        let (_, dst_result) = dst_cfg.add_stmt_intrinsic_op(
            dst_bb,
            BlockPosition::Append,
            src_stmt.intrinsic().clone(),
            arguments,
        );

        match (src_stmt.maybe_result(), dst_result) {
            (Some(src_result), Some(dst_result)) => {
                self.local_value_mapping.insert(src_result, dst_result);
            }
            (None, None) => (),
            _ => unreachable!(),
        }
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
