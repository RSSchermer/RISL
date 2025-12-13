use std::assert_matches::assert_matches;
use std::ops::Deref;

use rustc_public::abi::{FnAbi, Scalar, ValueAbi};
use rustc_public::mir::mono::Instance;
use rustc_public::target::MachineSize;
use rustc_public::ty::{Align, Size, Span, Ty, VariantIdx};

use super::abi::AbiBuilderMethods;
use super::consts::ConstCodegenMethods;
use super::intrinsic::IntrinsicCallBuilderMethods;
use super::misc::MiscCodegenMethods;
use super::type_::{ArgAbiBuilderMethods, BaseTypeCodegenMethods, LayoutTypeCodegenMethods};
use super::{CodegenMethods, StaticBuilderMethods};
use crate::stable_cg::TyAndLayout;
use crate::stable_cg::common::{
    AtomicOrdering, AtomicRmwBinOp, IntPredicate, RealPredicate, SynchronizationScope, TypeKind,
};
use crate::stable_cg::mir::operand::{OperandRef, OperandValue};
use crate::stable_cg::mir::place::{PlaceRef, PlaceValue};

#[derive(Copy, Clone, Debug)]
pub enum OverflowOp {
    Add,
    Sub,
    Mul,
}

pub trait BuilderMethods<'a>:
    Sized
    + Deref<Target = Self::CodegenCx>
    + ArgAbiBuilderMethods
    + AbiBuilderMethods
    + IntrinsicCallBuilderMethods
    + StaticBuilderMethods
{
    // `BackendTypes` is a supertrait of both `CodegenMethods` and
    // `BuilderMethods`. This bound ensures all impls agree on the associated
    // types within.
    type CodegenCx: CodegenMethods<
            Value = Self::Value,
            Function = Self::Function,
            BasicBlock = Self::BasicBlock,
            Type = Self::Type,
        >;

    fn build(cx: &'a Self::CodegenCx, llbb: Self::BasicBlock) -> Self;

    fn cx(&self) -> &Self::CodegenCx;
    fn llbb(&self) -> Self::BasicBlock;

    fn set_span(&mut self, span: Span);

    fn start_block(cx: &'a Self::CodegenCx, llfn: Self::Function) -> Self::BasicBlock;

    // FIXME(eddyb) replace uses of this with `append_sibling_block`.
    fn append_block(cx: &'a Self::CodegenCx, llfn: Self::Function, name: &str) -> Self::BasicBlock;

    fn append_sibling_block(&mut self, name: &str) -> Self::BasicBlock;

    fn as_local(&mut self, val: Self::Value) -> Self::Local;
    fn local_value(&mut self, local: Self::Local) -> Self::Value;

    fn switch_to_block(&mut self, llbb: Self::BasicBlock);

    fn ret_void(&mut self);
    fn ret(&mut self, v: Self::Value);
    fn br(&mut self, dest: Self::BasicBlock);
    fn cond_br(
        &mut self,
        cond: Self::Value,
        then_llbb: Self::BasicBlock,
        else_llbb: Self::BasicBlock,
    );

    fn switch(
        &mut self,
        v: Self::Value,
        else_llbb: Self::BasicBlock,
        cases: impl IntoIterator<Item = (u128, Self::BasicBlock)>,
    );

    fn get_discriminant(&mut self, ptr: Self::Value) -> Self::Value;
    fn set_discriminant(&mut self, ptr: Self::Value, variant_index: VariantIdx);

    fn unreachable(&mut self);

    fn add(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fadd(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fadd_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fadd_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn sub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fsub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fsub_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fsub_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn mul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fmul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fmul_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fmul_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn udiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn exactudiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn sdiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn exactsdiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fdiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fdiv_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fdiv_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn urem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn srem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn frem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn frem_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn frem_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    /// Generate a left-shift. Both operands must have the same size. The right operand must be
    /// interpreted as unsigned and can be assumed to be less than the size of the left operand.
    fn shl(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    /// Generate a logical right-shift. Both operands must have the same size. The right operand
    /// must be interpreted as unsigned and can be assumed to be less than the size of the left
    /// operand.
    fn lshr(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    /// Generate an arithmetic right-shift. Both operands must have the same size. The right operand
    /// must be interpreted as unsigned and can be assumed to be less than the size of the left
    /// operand.
    fn ashr(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn unchecked_sadd(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn unchecked_uadd(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn unchecked_ssub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn unchecked_usub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn unchecked_smul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn unchecked_umul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn and(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn or(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn xor(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn neg(&mut self, v: Self::Value) -> Self::Value;
    fn fneg(&mut self, v: Self::Value) -> Self::Value;
    fn not(&mut self, v: Self::Value) -> Self::Value;

    fn from_immediate(&mut self, val: Self::Value) -> Self::Value;
    fn to_immediate(&mut self, val: Self::Value, layout: &TyAndLayout) -> Self::Value {
        if let ValueAbi::Scalar(scalar) = layout.layout.abi {
            self.to_immediate_scalar(val, scalar)
        } else {
            val
        }
    }
    fn to_immediate_scalar(&mut self, val: Self::Value, scalar: Scalar) -> Self::Value;

    fn alloca(&mut self, layout: &TyAndLayout) -> Self::Value;

    fn assign(&mut self, local: Self::Local, value: Self::Value);

    fn load(&mut self, ty: Self::Type, ptr: Self::Value, align: Align) -> Self::Value;
    fn volatile_load(&mut self, ty: Self::Type, ptr: Self::Value) -> Self::Value;
    fn atomic_load(
        &mut self,
        ty: Self::Type,
        ptr: Self::Value,
        order: AtomicOrdering,
        size: MachineSize,
    ) -> Self::Value;
    fn load_from_place(&mut self, ty: Self::Type, place: PlaceValue<Self::Value>) -> Self::Value {
        assert_eq!(place.llextra, None);
        self.load(ty, place.llval, place.align)
    }
    fn load_operand(&mut self, place: &PlaceRef<Self::Value>) -> OperandRef<Self::Value>;

    /// Called for Rvalue::Repeat when the elem is neither a ZST nor optimizable using memset.
    fn write_operand_repeatedly(
        &mut self,
        elem: &OperandRef<Self::Value>,
        count: u64,
        dest: &PlaceRef<Self::Value>,
    );

    fn store(&mut self, val: Self::Value, ptr: Self::Value, align: Align) -> Self::Value;
    fn store_to_place(&mut self, val: Self::Value, place: PlaceValue<Self::Value>) -> Self::Value {
        assert_eq!(place.llextra, None);
        self.store(val, place.llval, place.align)
    }
    fn atomic_store(
        &mut self,
        val: Self::Value,
        ptr: Self::Value,
        order: AtomicOrdering,
        size: MachineSize,
    );

    fn ptr_element_ptr(
        &mut self,
        ty: Self::Type,
        ptr: Self::Value,
        indices: &[Self::Value],
    ) -> Self::Value;
    fn ptr_variant_ptr(&mut self, ptr: Self::Value, variant_idx: VariantIdx) -> Self::Value;

    fn offset_slice_ptr(
        &mut self,
        ptr: Self::Value,
        offset: Self::Value,
        ty: Self::Type,
    ) -> Self::Value;

    fn trunc(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn sext(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn fptoui(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn fptosi(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn uitofp(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn sitofp(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn fptrunc(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn fpext(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn bitcast(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
    fn intcast(&mut self, val: Self::Value, dest_ty: Self::Type, is_signed: bool) -> Self::Value;

    fn cast_float_to_int(
        &mut self,
        signed: bool,
        x: Self::Value,
        dest_ty: Self::Type,
    ) -> Self::Value {
        let in_ty = self.cx().val_ty(x);

        let (float_ty, int_ty) = if self.cx().type_kind(dest_ty) == TypeKind::Vector
            && self.cx().type_kind(in_ty) == TypeKind::Vector
        {
            (
                self.cx().element_type(in_ty),
                self.cx().element_type(dest_ty),
            )
        } else {
            (in_ty, dest_ty)
        };

        assert_matches!(
            self.cx().type_kind(float_ty),
            TypeKind::Half | TypeKind::Float | TypeKind::Double | TypeKind::FP128
        );
        assert_eq!(self.cx().type_kind(int_ty), TypeKind::Integer);

        if signed {
            self.fptosi(x, dest_ty)
        } else {
            self.fptoui(x, dest_ty)
        }
    }

    fn icmp(&mut self, op: IntPredicate, lhs: Self::Value, rhs: Self::Value) -> Self::Value;
    fn fcmp(&mut self, op: RealPredicate, lhs: Self::Value, rhs: Self::Value) -> Self::Value;

    /// *Typed* copy for non-overlapping places.
    fn typed_place_copy(
        &mut self,
        dst: PlaceValue<Self::Value>,
        src: PlaceValue<Self::Value>,
        layout: &TyAndLayout,
    ) {
        assert!(
            layout.layout.is_sized(),
            "cannot typed-copy an unsigned type"
        );
        assert!(
            src.llextra.is_none(),
            "cannot directly copy from unsized values"
        );
        assert!(
            dst.llextra.is_none(),
            "cannot directly copy into unsized values"
        );

        if self.is_backend_immediate(&layout) {
            let temp = self.load_operand(&src.with_type(layout.clone()));

            temp.val.store(self, &dst.with_type(layout.clone()));
        } else {
            let ty = self.backend_type(&layout);
            let val = self.load_from_place(ty, src);

            self.store_to_place(val, dst);
        }
    }

    /// *Typed* swap for non-overlapping places.
    ///
    /// Avoids `alloca`s for Immediates and ScalarPairs.
    ///
    /// FIXME: Maybe do something smarter for Ref types too?
    /// For now, the `typed_swap_nonoverlapping` intrinsic just doesn't call this for those
    /// cases (in non-debug), preferring the fallback body instead.
    fn typed_place_swap(
        &mut self,
        left: PlaceValue<Self::Value>,
        right: PlaceValue<Self::Value>,
        layout: &TyAndLayout,
    ) {
        let mut temp = self.load_operand(&left.with_type(layout.clone()));
        if let OperandValue::Ref(..) = temp.val {
            // The SSA value isn't stand-alone, so we need to copy it elsewhere
            let alloca = PlaceRef::alloca(self, layout.clone());
            self.typed_place_copy(alloca.val, left, layout);
            temp = self.load_operand(&alloca);
        }
        self.typed_place_copy(left, right, layout);
        temp.val.store(self, &right.with_type(layout.clone()));
    }

    fn select(
        &mut self,
        cond: Self::Value,
        then_val: Self::Value,
        else_val: Self::Value,
    ) -> Self::Value;

    fn extract_element(&mut self, vec: Self::Value, idx: Self::Value) -> Self::Value;
    fn vector_splat(&mut self, num_elts: usize, elt: Self::Value) -> Self::Value;
    fn extract_value(&mut self, agg_val: Self::Value, idx: u64) -> Self::Value;
    fn insert_value(&mut self, agg_val: Self::Value, elt: Self::Value, idx: u64) -> Self::Value;

    fn atomic_cmpxchg(
        &mut self,
        dst: Self::Value,
        cmp: Self::Value,
        src: Self::Value,
        order: AtomicOrdering,
        failure_order: AtomicOrdering,
        weak: bool,
    ) -> (Self::Value, Self::Value);
    fn atomic_rmw(
        &mut self,
        op: AtomicRmwBinOp,
        dst: Self::Value,
        src: Self::Value,
        order: AtomicOrdering,
    ) -> Self::Value;
    fn atomic_fence(&mut self, order: AtomicOrdering, scope: SynchronizationScope);
    fn set_invariant_load(&mut self, load: Self::Value);

    /// Called for `StorageLive`
    fn lifetime_start(&mut self, ptr: Self::Value, size: MachineSize);

    /// Called for `StorageDead`
    fn lifetime_end(&mut self, ptr: Self::Value, size: MachineSize);

    fn call(
        &mut self,
        llty: Self::Type,
        fn_abi: Option<&FnAbi>,
        llfn: Self::Value,
        args: &[Self::Value],
        instance: Option<&Instance>,
    ) -> Self::Value;
    fn zext(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value;
}
