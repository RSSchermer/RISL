use rustc_abi::{Float, Integer};
use rustc_public::abi::{ArgAbi, FnAbi};

use super::{BackendTypes, MiscCodegenMethods};
use crate::stable_cg::TyAndLayout;
use crate::stable_cg::common::TypeKind;
use crate::stable_cg::mir::place::PlaceRef;

pub trait BaseTypeCodegenMethods: BackendTypes {
    fn type_i8(&self) -> Self::Type;
    fn type_i16(&self) -> Self::Type;
    fn type_i32(&self) -> Self::Type;
    fn type_i64(&self) -> Self::Type;
    fn type_i128(&self) -> Self::Type;
    fn type_isize(&self) -> Self::Type;

    fn type_f16(&self) -> Self::Type;
    fn type_f32(&self) -> Self::Type;
    fn type_f64(&self) -> Self::Type;
    fn type_f128(&self) -> Self::Type;

    fn type_array(&self, ty: Self::Type, len: u64) -> Self::Type;
    fn type_func(&self, args: &[Self::Type], ret: Self::Type) -> Self::Type;
    fn type_kind(&self, ty: Self::Type) -> TypeKind;
    fn type_ptr(&self) -> Self::Type;
    fn element_type(&self, ty: Self::Type) -> Self::Type;

    /// Returns the number of elements in `self` if it is an LLVM vector type.
    fn vector_length(&self, ty: Self::Type) -> usize;

    fn float_width(&self, ty: Self::Type) -> usize;

    /// Retrieves the bit width of the integer type `self`.
    fn int_width(&self, ty: Self::Type) -> u64;

    fn val_ty(&self, v: Self::Value) -> Self::Type;
}

pub trait DerivedTypeCodegenMethods: BaseTypeCodegenMethods + MiscCodegenMethods {
    fn type_int(&self) -> Self::Type {
        self.type_i32()
    }

    fn type_from_integer(&self, i: Integer) -> Self::Type {
        use Integer::*;

        match i {
            I8 => self.type_i8(),
            I16 => self.type_i16(),
            I32 => self.type_i32(),
            I64 => self.type_i64(),
            I128 => self.type_i128(),
        }
    }

    fn type_from_float(&self, f: Float) -> Self::Type {
        use Float::*;

        match f {
            F16 => self.type_f16(),
            F32 => self.type_f32(),
            F64 => self.type_f64(),
            F128 => self.type_f128(),
        }
    }
}

impl<T> DerivedTypeCodegenMethods for T where Self: BaseTypeCodegenMethods + MiscCodegenMethods {}

pub trait LayoutTypeCodegenMethods: BackendTypes {
    /// The backend type used for a rust type when it's in memory,
    /// such as when it's stack-allocated or when it's being loaded or stored.
    fn backend_type(&self, layout: &TyAndLayout) -> Self::Type;

    fn fn_decl_backend_type(&self, fn_abi: &FnAbi) -> Self::Type;

    /// The backend type used for a rust type when it's in an SSA register.
    ///
    /// For nearly all types this is the same as the [`Self::backend_type`], however
    /// `bool` (and other `0`-or-`1` values) are kept as `i1` in registers but as
    /// [`BaseTypeCodegenMethods::type_i8`] in memory.
    ///
    /// Converting values between the two different backend types is done using
    /// [`from_immediate`](super::BuilderMethods::from_immediate) and
    /// [`to_immediate_scalar`](super::BuilderMethods::to_immediate_scalar).
    fn immediate_backend_type(&self, layout: &TyAndLayout) -> Self::Type;

    fn is_backend_immediate(&self, layout: &TyAndLayout) -> bool;

    fn is_backend_scalar_pair(&self, layout: &TyAndLayout) -> bool;

    fn scalar_pair_element_backend_type(
        &self,
        layout: &TyAndLayout,
        index: usize,
        immediate: bool,
    ) -> Self::Type;

    /// A type that produces an [`OperandValue::Ref`] when loaded.
    ///
    /// AKA one that's not a ZST, not `is_backend_immediate`, and
    /// not `is_backend_scalar_pair`. For such a type, a
    /// [`load_operand`] doesn't actually `load` anything.
    ///
    /// [`OperandValue::Ref`]: crate::mir::operand::OperandValue::Ref
    /// [`load_operand`]: super::BuilderMethods::load_operand
    fn is_backend_ref(&self, layout: &TyAndLayout) -> bool {
        !(layout.layout.is_1zst()
            || self.is_backend_immediate(layout)
            || self.is_backend_scalar_pair(layout))
    }
}

pub trait ArgAbiBuilderMethods: BackendTypes {
    fn store_fn_arg(&mut self, arg_abi: &ArgAbi, idx: &mut usize, dst: &PlaceRef<Self::Value>);
    fn store_arg(&mut self, arg_abi: &ArgAbi, val: Self::Value, dst: &PlaceRef<Self::Value>);
    fn arg_memory_ty(&self, arg_abi: &ArgAbi) -> Self::Type;
}

pub trait TypeCodegenMethods = DerivedTypeCodegenMethods + LayoutTypeCodegenMethods;
