use std::mem;
use std::ops::Deref;

use rustc_middle::bug;
use rustc_public::abi;
use rustc_public::abi::{ArgAbi, FnAbi, PassMode, ValueAbi};
use rustc_public::mir::mono::{Instance, StaticDef};
use rustc_public::target::MachineSize;
use rustc_public::ty::{Align, Span, VariantIdx};
use rustc_public_bridge::IndexedVal;
use slir::cfg::BlockPosition;
use smallvec::{SmallVec, smallvec};

use crate::slir_build::context::CodegenContext as Cx;
use crate::slir_build::ty::Type;
use crate::slir_build::value::Value;
use crate::stable_cg::traits::{
    AbiBuilderMethods, ArgAbiBuilderMethods, BackendTypes, BuilderMethods, ConstCodegenMethods,
    IntrinsicCallBuilderMethods, LayoutTypeCodegenMethods, StaticBuilderMethods,
};
use crate::stable_cg::{
    AtomicOrdering, AtomicRmwBinOp, IntPredicate, OperandRef, OperandValue, PlaceRef, PlaceValue,
    RealPredicate, Scalar, SynchronizationScope, TyAndLayout, TypeKind,
};

pub struct Builder<'a, 'tcx> {
    cx: &'a Cx<'a, 'tcx>,
    basic_block: slir::cfg::BasicBlock,
}

impl<'a, 'tcx> Deref for Builder<'a, 'tcx> {
    type Target = Cx<'a, 'tcx>;

    fn deref(&self) -> &Self::Target {
        &self.cx
    }
}

impl<'a, 'tcx> BackendTypes for Builder<'a, 'tcx> {
    type Value = Value;
    type Local = slir::cfg::LocalBinding;
    type Function = slir::Function;
    type BasicBlock = slir::cfg::BasicBlock;
    type Type = Type;
}

impl<'a, 'tcx> ArgAbiBuilderMethods for Builder<'a, 'tcx> {
    fn store_fn_arg(&mut self, arg_abi: &ArgAbi, idx: &mut usize, dst: &PlaceRef<Self::Value>) {
        fn next(bx: &mut Builder<'_, '_>, idx: &mut usize) -> Value {
            let val = bx.get_param(*idx);

            *idx += 1;

            val
        }

        match arg_abi.mode {
            PassMode::Ignore => {}
            PassMode::Direct(_) => {
                let next_arg = next(self, idx);

                self.store_arg(arg_abi, next_arg, dst);
            }
            PassMode::Pair(..) => {
                OperandValue::Pair(next(self, idx), next(self, idx)).store(self, dst);
            }
            // TODO: from some cursory analysis of the codegen code, it does not seem that
            // store_fn_arg (or store_arg) ever gets called with in indirect pass-mode (which makes
            // sense, we only alloca and call store_fn_arg when we need to interact with the value
            // via a pointer, and in the case of an indirect argument we clearly already have a
            // pointer). Trying to store an unsized value like a slice (as we do here), will also
            // always cause an ICE (can't directly store an unsized value). Should we just assert
            // that the pass-mode cannot be "indirect" here?
            PassMode::Indirect { .. } if arg_abi.ty.kind().is_slice() => {
                let place_val = PlaceValue {
                    llval: next(self, idx),
                    llextra: Some(next(self, idx)),
                    align: arg_abi.layout.shape().abi_align,
                };

                OperandValue::Ref(place_val).store(self, dst);
            }
            PassMode::Indirect { .. } => {
                let next_arg = next(self, idx);

                self.store_arg(arg_abi, next_arg, dst);
            }
            PassMode::Cast { .. } => bug!("not supported by RISL"),
        }
    }

    fn store_arg(&mut self, arg_abi: &ArgAbi, val: Self::Value, dst: &PlaceRef<Self::Value>) {
        match &arg_abi.mode {
            PassMode::Ignore => {}
            PassMode::Indirect { .. } => {
                let align = arg_abi.layout.shape().abi_align;

                OperandValue::Ref(PlaceValue::new_sized(val, align)).store(self, dst);
            }
            PassMode::Direct(_) | PassMode::Pair(..) => {
                OperandRef::from_immediate_or_packed_pair(
                    self,
                    val,
                    TyAndLayout {
                        ty: arg_abi.ty,
                        layout: arg_abi.layout.shape(),
                    },
                )
                .val
                .store(self, dst);
            }
            PassMode::Cast { .. } => bug!("not supported by RISL"),
        }
    }

    fn arg_memory_ty(&self, arg_abi: &ArgAbi) -> Self::Type {
        todo!()
    }
}

impl<'a, 'tcx> AbiBuilderMethods for Builder<'a, 'tcx> {
    fn get_param(&mut self, index: usize) -> Self::Value {
        let cfg = self.cx.cfg.borrow();
        let function = cfg[self.basic_block].owner();

        cfg.get_function_body(function)
            .expect("function not registered")
            .argument_values()[index]
            .into()
    }
}

impl<'a, 'tcx> IntrinsicCallBuilderMethods for Builder<'a, 'tcx> {
    fn codegen_intrinsic_call(
        &mut self,
        instance: Instance,
        fn_abi: &FnAbi,
        args: &[OperandRef<Self::Value>],
        llresult: Self::Value,
        span: Span,
    ) -> Result<(), Instance> {
        todo!()
    }
}

impl<'a, 'tcx> StaticBuilderMethods for Builder<'a, 'tcx> {
    fn get_static(&mut self, def: StaticDef) -> Self::Value {
        todo!()
    }
}

macro_rules! unary_builder_methods {
    ($($method:ident => $op:ident,)*) => {
        $(
            fn $method(&mut self, value: Self::Value) -> Self::Value {
                let value = value.expect_value();

                let (_, result) = self.cfg.borrow_mut().add_stmt_op_unary(
                    self.basic_block,
                    BlockPosition::Append,
                    slir::UnaryOperator::$op,
                    value,
                );

                result.into()
            }
        )*
    };
}

macro_rules! binary_builder_methods {
    ($($method:ident => $op:ident,)*) => {
        $(
            fn $method(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
                let lhs = lhs.expect_value();
                let rhs = rhs.expect_value();

                let (_, result) = self.cfg.borrow_mut().add_stmt_op_binary(
                    self.basic_block,
                    BlockPosition::Append,
                    slir::BinaryOperator::$op,
                    lhs,
                    rhs,
                );

                result.into()
            }
        )*
    };
}

impl<'a, 'tcx> BuilderMethods<'a> for Builder<'a, 'tcx> {
    type CodegenCx = Cx<'a, 'tcx>;

    fn build(cx: &'a Self::CodegenCx, llbb: Self::BasicBlock) -> Self {
        Builder {
            cx,
            basic_block: llbb,
        }
    }

    fn cx(&self) -> &Self::CodegenCx {
        self.cx
    }

    fn llbb(&self) -> Self::BasicBlock {
        self.basic_block
    }

    fn set_span(&mut self, span: Span) {}

    fn start_block(cx: &'a Self::CodegenCx, llfn: Self::Function) -> Self::BasicBlock {
        cx.cfg.borrow()[llfn].entry_block()
    }

    fn append_block(
        cx: &'a Self::CodegenCx,
        function: Self::Function,
        _name: &str,
    ) -> Self::BasicBlock {
        cx.cfg.borrow_mut().add_basic_block(function)
    }

    fn as_local(&mut self, val: Self::Value) -> Self::Local {
        let val = val.expect_value();

        // If the value already represents a local, return that local.
        if let slir::cfg::Value::Local(local) = val {
            return local;
        }

        // Otherwise create a new local binding.
        let (_, local) =
            self.cfg
                .borrow_mut()
                .add_stmt_bind(self.basic_block, BlockPosition::Append, val);

        local
    }

    fn local_value(&mut self, local: Self::Local) -> Self::Value {
        Value::Value(slir::cfg::Value::Local(local))
    }

    fn append_sibling_block(&mut self, _name: &str) -> Self::BasicBlock {
        let mut cfg = self.cfg.borrow_mut();
        let function = cfg[self.basic_block].owner();

        cfg.add_basic_block(function)
    }

    fn switch_to_block(&mut self, llbb: Self::BasicBlock) {
        self.basic_block = llbb;
    }

    fn ret_void(&mut self) {
        self.cfg
            .borrow_mut()
            .set_terminator(self.basic_block, slir::cfg::Terminator::return_void());
    }

    fn ret(&mut self, v: Self::Value) {
        {
            let v = v.expect_value().expect_local();
            let cfg = self.cx.cfg.borrow();
            let data = &cfg[v];
        }

        self.cfg.borrow_mut().set_terminator(
            self.basic_block,
            slir::cfg::Terminator::return_value(v.expect_value()),
        );
    }

    fn br(&mut self, dest: Self::BasicBlock) {
        self.cfg
            .borrow_mut()
            .set_terminator(self.basic_block, slir::cfg::Terminator::branch_single(dest));
    }

    fn cond_br(
        &mut self,
        cond: Self::Value,
        then_llbb: Self::BasicBlock,
        else_llbb: Self::BasicBlock,
    ) {
        let mut cfg = self.cfg.borrow_mut();

        let (_, predicate) = cfg.add_stmt_op_bool_to_branch_predicate(
            self.basic_block,
            BlockPosition::Append,
            cond.expect_value(),
        );

        cfg.set_terminator(
            self.basic_block,
            slir::cfg::Terminator::branch_multiple(predicate, [then_llbb, else_llbb]),
        );
    }

    fn switch(
        &mut self,
        v: Self::Value,
        else_llbb: Self::BasicBlock,
        cases: impl IntoIterator<Item = (u128, Self::BasicBlock)>,
    ) {
        let mut predicate_cases = vec![];
        let mut branches: SmallVec<[_; 2]> = smallvec![];

        // Note: this loop has to run before we borrow the `cfg` below, as the `cases` iterator will
        // actually call [Builder::append_block], which will also want to borrow the `cfg`, leading
        // to an "already borrowed" error.
        for (case, branch) in cases {
            let Ok(case) = u32::try_from(case) else {
                bug!("validation should not have allowed a case that does not fit a `u32`");
            };

            predicate_cases.push(case);
            branches.push(branch);
        }

        // TODO: in our current examples the else block is always an "unreachable" block, which the
        // RVSDG construction algorithm doesn't like. Figure our if we can just always omit the else
        // block or if we need to handle unreachable blocks.
        // branches.push(else_llbb);

        let mut cfg = self.cfg.borrow_mut();

        let (_, predicate) = cfg.add_stmt_op_case_to_branch_predicate(
            self.basic_block,
            BlockPosition::Append,
            v.expect_value(),
            predicate_cases,
        );

        cfg.set_terminator(
            self.basic_block,
            slir::cfg::Terminator::branch_multiple(predicate, branches),
        );
    }

    fn unreachable(&mut self) {}

    fn get_discriminant(&mut self, ptr: Self::Value) -> Self::Value {
        let ptr = ptr.expect_value();

        let (_, result) = self.cfg.borrow_mut().add_stmt_op_get_discriminant(
            self.basic_block,
            BlockPosition::Append,
            ptr,
        );

        result.into()
    }

    fn set_discriminant(&mut self, ptr: Self::Value, variant_index: VariantIdx) {
        let ptr = ptr.expect_value();
        let variant_index = variant_index.to_index() as u32;

        self.cfg.borrow_mut().add_stmt_op_set_discriminant(
            self.basic_block,
            BlockPosition::Append,
            ptr,
            variant_index,
        );
    }

    unary_builder_methods! {
        neg => Neg,
        fneg => Neg,
        not => Not,
    }

    binary_builder_methods! {
        add => Add,
        fadd => Add,
        sub => Sub,
        fsub => Sub,
        mul => Mul,
        fmul => Mul,
        udiv => Div,
        sdiv => Div,
        fdiv => Div,
        shl => Shl,
        and => And,
        or => Or,
        bit_and => BitAnd,
        bit_or => BitOr,
        bit_xor => BitXor,
    }

    fn fadd_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn fadd_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn fsub_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn fsub_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn fmul_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn fmul_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn exactudiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn exactsdiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn fdiv_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn fdiv_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn urem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn srem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn frem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn frem_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn frem_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn lshr(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn ashr(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn unchecked_sadd(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn unchecked_uadd(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn unchecked_ssub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn unchecked_usub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn unchecked_smul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn unchecked_umul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!()
    }

    fn from_immediate(&mut self, val: Self::Value) -> Self::Value {
        val
    }

    fn to_immediate_scalar(&mut self, val: Self::Value, _scalar: abi::Scalar) -> Self::Value {
        val
    }

    fn alloca(&mut self, layout: &TyAndLayout) -> Self::Value {
        let ty = self.cx.ty_and_layout_resolve(layout);
        let (_, result) =
            self.cfg
                .borrow_mut()
                .add_stmt_op_alloca(self.basic_block, BlockPosition::Append, ty);

        result.into()
    }

    fn assign(&mut self, local: Self::Local, value: Self::Value) {
        self.cfg.borrow_mut().add_stmt_assign(
            self.basic_block,
            BlockPosition::Append,
            local,
            value.expect_value(),
        );
    }

    fn load(&mut self, _ty: Self::Type, ptr: Self::Value, _align: Align) -> Self::Value {
        let (_, result) = self.cfg.borrow_mut().add_stmt_op_load(
            self.basic_block,
            BlockPosition::Append,
            ptr.expect_value(),
        );

        result.into()
    }

    fn volatile_load(&mut self, ty: Self::Type, ptr: Self::Value) -> Self::Value {
        todo!()
    }

    fn atomic_load(
        &mut self,
        ty: Self::Type,
        ptr: Self::Value,
        order: AtomicOrdering,
        size: MachineSize,
    ) -> Self::Value {
        todo!()
    }

    fn load_operand(&mut self, place: &PlaceRef<Self::Value>) -> OperandRef<Self::Value> {
        if place.layout.layout.is_1zst() {
            return OperandRef::zero_sized(place.layout.clone());
        }

        let val = if place.val.llextra.is_some() || place.layout.ty.kind().is_enum() {
            OperandValue::Ref(place.val)
        } else if self.is_backend_immediate(&place.layout) {
            let llval = self.load(
                self.cx.ty_and_layout_resolve(&place.layout).into(),
                place.val.llval,
                place.val.align,
            );

            OperandValue::Immediate(self.to_immediate(llval, &place.layout))
        } else if let ValueAbi::ScalarPair(a, b) = place.layout.layout.abi {
            return OperandRef::from_immediate_or_packed_pair(
                self,
                place.val.llval,
                place.layout.clone(),
            );
        } else {
            OperandValue::Ref(place.val)
        };

        OperandRef {
            val,
            layout: place.layout.clone(),
        }
    }

    fn write_operand_repeatedly(
        &mut self,
        elem: &OperandRef<Self::Value>,
        count: u64,
        dest: &PlaceRef<Self::Value>,
    ) {
        let elem_ty = self.backend_type(&elem.layout);
        let elem = match elem.val {
            OperandValue::Ref(v) => self.load(elem_ty, v.llval, elem.layout.layout.abi_align),
            OperandValue::Immediate(v) => v,
            _ => bug!(),
        };

        let elem = elem.expect_value();
        let dest = dest.val.llval.expect_value();

        let mut cfg = self.cfg.borrow_mut();

        for i in 0..count {
            let index = self.const_usize(i).expect_value();
            let (_, elem_ptr) = cfg.add_stmt_op_ptr_element_ptr(
                self.basic_block,
                BlockPosition::Append,
                dest,
                [index],
            );

            cfg.add_stmt_op_store(
                self.basic_block,
                BlockPosition::Append,
                elem_ptr.into(),
                elem,
            );
        }
    }

    fn store(&mut self, val: Self::Value, ptr: Self::Value, align: Align) -> Self::Value {
        let value = val.expect_value();
        let ptr = ptr.expect_value();

        self.cfg.borrow_mut().add_stmt_op_store(
            self.basic_block,
            BlockPosition::Append,
            ptr,
            value,
        );

        Value::Void
    }

    fn atomic_store(
        &mut self,
        val: Self::Value,
        ptr: Self::Value,
        order: AtomicOrdering,
        size: MachineSize,
    ) {
        todo!()
    }

    fn ptr_element_ptr(
        &mut self,
        _ty: Self::Type,
        ptr: Self::Value,
        indices: &[Self::Value],
    ) -> Self::Value {
        let (_, result) = self.cfg.borrow_mut().add_stmt_op_ptr_element_ptr(
            self.basic_block,
            BlockPosition::Append,
            ptr.expect_value(),
            indices.iter().map(|i| i.expect_value()),
        );

        result.into()
    }

    fn ptr_variant_ptr(&mut self, ptr: Self::Value, variant_idx: VariantIdx) -> Self::Value {
        let ptr = ptr.expect_value();
        let variant_index = variant_idx.to_index();

        let (_, result) = self.cfg.borrow_mut().add_stmt_op_ptr_variant_ptr(
            self.basic_block,
            BlockPosition::Append,
            ptr,
            variant_index as u32,
        );

        result.into()
    }

    fn offset_slice_ptr(
        &mut self,
        ptr: Self::Value,
        offset: Self::Value,
        ty: Self::Type,
    ) -> Self::Value {
        let ptr = ptr.expect_value();
        let offset = offset.expect_value();

        let (_, result) = self.cfg.borrow_mut().add_stmt_op_offset_slice_pointer(
            self.basic_block,
            BlockPosition::Append,
            ptr,
            offset,
        );

        result.into()
    }

    fn trunc(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        todo!()
    }

    fn sext(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        todo!()
    }

    fn fptoui(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        todo!()
    }

    fn fptosi(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        todo!()
    }

    fn uitofp(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        todo!()
    }

    fn sitofp(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        todo!()
    }

    fn fptrunc(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        todo!()
    }

    fn fpext(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        todo!()
    }

    fn bitcast(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        todo!()
    }

    fn intcast(&mut self, val: Self::Value, dest_ty: Self::Type, is_signed: bool) -> Self::Value {
        todo!()
    }

    fn icmp(&mut self, op: IntPredicate, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let operator = match op {
            IntPredicate::IntEQ => slir::BinaryOperator::Eq,
            IntPredicate::IntNE => slir::BinaryOperator::NotEq,
            IntPredicate::IntUGT => slir::BinaryOperator::Gt,
            IntPredicate::IntUGE => slir::BinaryOperator::GtEq,
            IntPredicate::IntULT => slir::BinaryOperator::Lt,
            IntPredicate::IntULE => slir::BinaryOperator::LtEq,
            IntPredicate::IntSGT => slir::BinaryOperator::Gt,
            IntPredicate::IntSGE => slir::BinaryOperator::GtEq,
            IntPredicate::IntSLT => slir::BinaryOperator::Lt,
            IntPredicate::IntSLE => slir::BinaryOperator::LtEq,
        };

        let lhs = lhs.expect_value();
        let rhs = rhs.expect_value();

        let (_, result) = self.cfg.borrow_mut().add_stmt_op_binary(
            self.basic_block,
            BlockPosition::Append,
            operator,
            lhs,
            rhs,
        );

        result.into()
    }

    fn fcmp(&mut self, op: RealPredicate, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let operator = match op {
            RealPredicate::RealOEQ => slir::BinaryOperator::Eq,
            RealPredicate::RealOGT => slir::BinaryOperator::Gt,
            RealPredicate::RealOGE => slir::BinaryOperator::GtEq,
            RealPredicate::RealOLT => slir::BinaryOperator::Lt,
            RealPredicate::RealOLE => slir::BinaryOperator::LtEq,
            RealPredicate::RealONE => slir::BinaryOperator::NotEq,
            RealPredicate::RealORD
            | RealPredicate::RealUNO
            | RealPredicate::RealUEQ
            | RealPredicate::RealUGT
            | RealPredicate::RealUGE
            | RealPredicate::RealULT
            | RealPredicate::RealULE
            | RealPredicate::RealUNE
            | RealPredicate::RealPredicateTrue
            | RealPredicate::RealPredicateFalse => bug!("unsupported fcmp predicate `{:?}`", op),
        };

        let lhs = lhs.expect_value();
        let rhs = rhs.expect_value();

        let (_, result) = self.cfg.borrow_mut().add_stmt_op_binary(
            self.basic_block,
            BlockPosition::Append,
            operator,
            lhs,
            rhs,
        );

        result.into()
    }

    fn select(
        &mut self,
        cond: Self::Value,
        then_val: Self::Value,
        else_val: Self::Value,
    ) -> Self::Value {
        todo!()
    }

    fn extract_element(&mut self, vec: Self::Value, idx: Self::Value) -> Self::Value {
        todo!()
    }

    fn vector_splat(&mut self, num_elts: usize, elt: Self::Value) -> Self::Value {
        todo!()
    }

    fn extract_value(&mut self, agg_val: Self::Value, idx: u64) -> Self::Value {
        let mut cfg = self.cfg.borrow_mut();

        let agg_val = agg_val.expect_value();
        let agg_ty = cfg.value_ty(&agg_val);
        let agg_ty_kind = cfg.ty().kind(agg_ty);

        let index = slir::cfg::InlineConst::U32(idx as u32);

        if let slir::ty::TypeKind::Ptr(pointee_ty) = *agg_ty_kind {
            let pointee_ty_kind = cfg.ty().kind(pointee_ty);

            mem::drop(cfg);

            let val_ptr =
                self.ptr_element_ptr(slir::ty::TY_DUMMY.into(), agg_val.into(), &[index.into()]);

            self.load(slir::ty::TY_DUMMY.into(), val_ptr, 0)
        } else {
            let (_, result) = cfg.add_stmt_op_extract_value(
                self.basic_block,
                BlockPosition::Append,
                agg_val,
                [index.into()],
            );

            result.into()
        }
    }

    fn insert_value(&mut self, agg_val: Self::Value, elt: Self::Value, idx: u64) -> Self::Value {
        todo!()
    }

    fn atomic_cmpxchg(
        &mut self,
        dst: Self::Value,
        cmp: Self::Value,
        src: Self::Value,
        order: AtomicOrdering,
        failure_order: AtomicOrdering,
        weak: bool,
    ) -> (Self::Value, Self::Value) {
        todo!()
    }

    fn atomic_rmw(
        &mut self,
        op: AtomicRmwBinOp,
        dst: Self::Value,
        src: Self::Value,
        order: AtomicOrdering,
    ) -> Self::Value {
        todo!()
    }

    fn atomic_fence(&mut self, order: AtomicOrdering, scope: SynchronizationScope) {
        todo!()
    }

    fn set_invariant_load(&mut self, load: Self::Value) {
        todo!()
    }

    fn lifetime_start(&mut self, ptr: Self::Value, size: MachineSize) {}

    fn lifetime_end(&mut self, ptr: Self::Value, size: MachineSize) {}

    fn call(
        &mut self,
        llty: Self::Type,
        _fn_abi: Option<&FnAbi>,
        llfn: Self::Value,
        args: &[Self::Value],
        _instance: Option<&Instance>,
    ) -> Self::Value {
        let function = llfn.expect_fn_addr();
        let args = args.iter().map(|a| a.expect_value());

        let (_, result) = self.cfg.borrow_mut().add_stmt_op_call(
            self.basic_block,
            BlockPosition::Append,
            function,
            llty.fn_decl_ret_ty(),
            args,
        );

        if let Some(result) = result {
            result.into()
        } else {
            Value::Void
        }
    }

    fn zext(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        todo!()
    }
}
