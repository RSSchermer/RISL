use std::assert_matches::assert_matches;
use std::fmt;

use arrayvec::ArrayVec;
use either::Either;
use rustc_middle::bug;
use rustc_public::abi::ValueAbi;
use rustc_public::target::MachineInfo;
use rustc_public::ty::{Align, Allocation, ConstantKind, MirConst, Size, Ty};
use rustc_public::{abi, mir};
use tracing::debug;

use super::place::{PlaceRef, PlaceValue};
use super::{FunctionCx, LocalRef};
use crate::stable_cg::layout::{ScalarExt, TyAndLayout};
use crate::stable_cg::scalar::Scalar;
use crate::stable_cg::traits::*;

/// The representation of a Rust value. The enum variant is in fact
/// uniquely determined by the value's type, but is kept as a
/// safety check.
#[derive(Copy, Clone, Debug)]
pub enum OperandValue<V> {
    /// A reference to the actual operand. The data is guaranteed
    /// to be valid for the operand's lifetime.
    /// The second value, if any, is the extra data (vtable or length)
    /// which indicates that it refers to an unsized rvalue.
    ///
    /// An `OperandValue` *must* be this variant for any type for which
    /// [`LayoutTypeCodegenMethods::is_backend_ref`] returns `true`.
    /// (That basically amounts to "isn't one of the other variants".)
    ///
    /// This holds a [`PlaceValue`] (like a [`PlaceRef`] does) with a pointer
    /// to the location holding the value. The type behind that pointer is the
    /// one returned by [`LayoutTypeCodegenMethods::backend_type`].
    Ref(PlaceValue<V>),
    /// A single LLVM immediate value.
    ///
    /// An `OperandValue` *must* be this variant for any type for which
    /// [`LayoutTypeCodegenMethods::is_backend_immediate`] returns `true`.
    /// The backend value in this variant must be the *immediate* backend type,
    /// as returned by [`LayoutTypeCodegenMethods::immediate_backend_type`].
    Immediate(V),
    /// A pair of immediate LLVM values. Used by wide pointers too.
    ///
    /// An `OperandValue` *must* be this variant for any type for which
    /// [`LayoutTypeCodegenMethods::is_backend_scalar_pair`] returns `true`.
    /// The backend values in this variant must be the *immediate* backend types,
    /// as returned by [`LayoutTypeCodegenMethods::scalar_pair_element_backend_type`]
    /// with `immediate: true`.
    Pair(V, V),
    /// A value taking no bytes, and which therefore needs no LLVM value at all.
    ///
    /// If you ever need a `V` to pass to something, get a fresh poison value
    /// from [`ConstCodegenMethods::const_poison`].
    ///
    /// An `OperandValue` *must* be this variant for any type for which
    /// `is_zst` on its `Layout` returns `true`. Note however that
    /// these values can still require alignment.
    ZeroSized,
}

impl<V: CodegenObject> OperandValue<V> {
    /// If this is ZeroSized/Immediate/Pair, return an array of the 0/1/2 values.
    /// If this is Ref, return the place.
    #[inline]
    pub(crate) fn immediates_or_place(self) -> Either<ArrayVec<V, 2>, PlaceValue<V>> {
        match self {
            OperandValue::ZeroSized => Either::Left(ArrayVec::new()),
            OperandValue::Immediate(a) => Either::Left(ArrayVec::from_iter([a])),
            OperandValue::Pair(a, b) => Either::Left([a, b].into()),
            OperandValue::Ref(p) => Either::Right(p),
        }
    }

    /// Given an array of 0/1/2 immediate values, return ZeroSized/Immediate/Pair.
    #[inline]
    pub(crate) fn from_immediates(immediates: ArrayVec<V, 2>) -> Self {
        let mut it = immediates.into_iter();
        let Some(a) = it.next() else {
            return OperandValue::ZeroSized;
        };
        let Some(b) = it.next() else {
            return OperandValue::Immediate(a);
        };
        OperandValue::Pair(a, b)
    }

    /// Treat this value as a pointer and return the data pointer and
    /// optional metadata as backend values.
    ///
    /// If you're making a place, use [`Self::deref`] instead.
    pub(crate) fn pointer_parts(self) -> (V, Option<V>) {
        match self {
            OperandValue::Immediate(llptr) => (llptr, None),
            OperandValue::Pair(llptr, llextra) => (llptr, Some(llextra)),
            _ => bug!("OperandValue cannot be a pointer: {self:?}"),
        }
    }

    /// Treat this value as a pointer and return the place to which it points.
    ///
    /// The pointer immediate doesn't inherently know its alignment,
    /// so you need to pass it in. If you want to get it from a type's ABI
    /// alignment, then maybe you want [`OperandRef::deref`] instead.
    ///
    /// This is the inverse of [`PlaceValue::address`].
    pub(crate) fn deref(self, align: Align) -> PlaceValue<V> {
        let (llval, llextra) = self.pointer_parts();

        PlaceValue {
            llval,
            llextra,
            align,
        }
    }

    pub(crate) fn is_expected_variant_for_type<Cx: LayoutTypeCodegenMethods>(
        &self,
        cx: &Cx,
        ty: &TyAndLayout,
    ) -> bool {
        match self {
            OperandValue::ZeroSized => ty.layout.is_1zst(),
            OperandValue::Immediate(_) => cx.is_backend_immediate(ty),
            OperandValue::Pair(_, _) => cx.is_backend_scalar_pair(ty),
            OperandValue::Ref(_) => cx.is_backend_ref(ty),
        }
    }
}

/// An `OperandRef` is an "SSA" reference to a Rust value, along with
/// its type.
///
/// NOTE: unless you know a value's type exactly, you should not
/// generate LLVM opcodes acting on it and instead act via methods,
/// to avoid nasty edge cases. In particular, using `Builder::store`
/// directly is sure to cause problems -- use `OperandRef::store`
/// instead.
#[derive(Clone)]
pub struct OperandRef<V> {
    /// The value.
    pub val: OperandValue<V>,

    /// The layout of value, based on its Rust type.
    pub layout: TyAndLayout,
}

impl<V: CodegenObject> fmt::Debug for OperandRef<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OperandRef({:?} @ {:?})", self.val, self.layout)
    }
}

impl<'a, V: CodegenObject> OperandRef<V> {
    pub fn zero_sized(layout: TyAndLayout) -> OperandRef<V> {
        assert!(layout.layout.is_1zst());

        OperandRef {
            val: OperandValue::ZeroSized,
            layout,
        }
    }

    pub(crate) fn from_const<Bx: BuilderMethods<'a, Value = V>>(
        bx: &mut Bx,
        val: &MirConst,
    ) -> Self {
        let alloc = match val.kind() {
            ConstantKind::Allocated(alloc) => alloc,
            ConstantKind::ZeroSized => {
                return Self::zero_sized(TyAndLayout::expect_from_ty(val.ty()));
            }
            _ => bug!("cannot construct const operand without a data allocation"),
        };

        let ty = val.ty();
        let layout = TyAndLayout::expect_from_ty(ty);
        let alloc_align = alloc.align;

        assert!(alloc_align >= layout.layout.abi_align);

        let machine_info = MachineInfo::target();

        match layout.layout.abi.clone() {
            ValueAbi::Scalar(abi) => {
                let size = abi.size(&machine_info);

                assert_eq!(
                    size, layout.layout.size,
                    "abi::Scalar size does not match layout size"
                );

                let val = Scalar::read_from_alloc(alloc, 0, abi, &layout);
                let val = bx.scalar_to_backend(val);

                return OperandRef {
                    val: OperandValue::Immediate(val),
                    layout,
                };
            }
            ValueAbi::ScalarPair(a, b) => {
                let b_offset = a.size(&machine_info).bytes();

                let a_val = Scalar::read_from_alloc(alloc, 0, a, &layout.field(0));
                let b_val = Scalar::read_from_alloc(alloc, b_offset, b, &layout.field(1));

                let a_val = bx.scalar_to_backend(a_val);
                let b_val = bx.scalar_to_backend(b_val);

                return OperandRef {
                    val: OperandValue::Pair(a_val, b_val),
                    layout,
                };
            }
            _ if layout.layout.is_1zst() => return OperandRef::zero_sized(layout),
            _ => {}
        }

        // Neither a scalar nor scalar pair. Load from a place
        // FIXME: should we cache `const_data_from_alloc` to avoid repeating this for the
        // same `ConstAllocation`?
        let init = bx.const_data_from_alloc(alloc, &layout);
        let addr = bx.static_addr_of(init, alloc_align, None);

        bx.load_operand(&PlaceRef::new_sized(addr, layout))
    }

    /// Asserts that this operand refers to a scalar and returns
    /// a reference to its value.
    pub fn immediate(self) -> V {
        match self.val {
            OperandValue::Immediate(s) => s,
            _ => bug!("not immediate: {:?}", self),
        }
    }

    /// Asserts that this operand is a pointer (or reference) and returns
    /// the place to which it points.  (This requires no code to be emitted
    /// as we represent places using the pointer to the place.)
    ///
    /// This uses [`Ty::builtin_deref`] to include the type of the place and
    /// assumes the place is aligned to the pointee's usual ABI alignment.
    ///
    /// If you don't need the type, see [`OperandValue::pointer_parts`]
    /// or [`OperandValue::deref`].
    pub fn deref<Cx: CodegenMethods>(self, cx: &Cx) -> PlaceRef<V> {
        if self.layout.ty.kind().is_box() {
            // Derefer should have removed all Box derefs
            bug!("dereferencing {:?} in codegen", self.layout.ty);
        }

        let projected_ty = self
            .layout
            .ty
            .kind()
            .builtin_deref(true)
            .unwrap_or_else(|| bug!("deref of non-pointer {:?}", self))
            .ty;

        let layout = projected_ty
            .layout()
            .expect("type should have known layout")
            .shape();

        self.val.deref(layout.abi_align).with_type(TyAndLayout {
            ty: projected_ty,
            layout,
        })
    }

    /// If this operand is a `Pair`, we return an aggregate with the two values.
    /// For other cases, see `immediate`.
    pub fn immediate_or_packed_pair<Bx: BuilderMethods<'a, Value = V>>(self, bx: &mut Bx) -> V {
        if let OperandValue::Pair(a, b) = self.val {
            let a_layout = self.layout.field(0);
            let b_layout = self.layout.field(1);

            let a_llty = bx.immediate_backend_type(&a_layout);
            let b_llty = bx.immediate_backend_type(&b_layout);

            // Reconstruct the immediate aggregate.
            let mut alloca = bx.alloca(&self.layout);

            let a_ptr = bx.element_ptr(a_llty, alloca, &[bx.const_usize(0)]);

            bx.store(a, a_ptr, a_layout.layout.abi_align);

            let b_ptr = bx.element_ptr(b_llty, alloca, &[bx.const_usize(1)]);

            bx.store(b, b_ptr, b_layout.layout.abi_align);

            let llty = bx.immediate_backend_type(&self.layout);

            bx.load(llty, alloca, self.layout.layout.abi_align)
        } else {
            self.immediate()
        }
    }

    /// If the type is a pair, we return a `Pair`, otherwise, an `Immediate`.
    pub fn from_immediate_or_packed_pair<Bx: BuilderMethods<'a, Value = V>>(
        bx: &mut Bx,
        llval: V,
        layout: TyAndLayout,
    ) -> Self {
        let val = if let ValueAbi::ScalarPair(..) = layout.layout.abi {
            debug!(
                "Operand::from_immediate_or_packed_pair: unpacking {:?} @ {:?}",
                llval, layout
            );

            // Deconstruct the immediate aggregate.
            let a_llval = bx.extract_value(llval, 0);
            let b_llval = bx.extract_value(llval, 1);

            OperandValue::Pair(a_llval, b_llval)
        } else {
            OperandValue::Immediate(llval)
        };
        OperandRef { val, layout }
    }

    pub(crate) fn extract_field<Bx: BuilderMethods<'a, Value = V>>(
        &self,
        bx: &mut Bx,
        i: usize,
    ) -> Self {
        let field = self.layout.field(i);
        let is_first_field = self.layout.layout.fields.fields_by_offset_order()[0] == i;

        let mut val = match (self.val, &self.layout.layout.abi) {
            // If the field is ZST, it has no data.
            _ if field.layout.is_1zst() => OperandValue::ZeroSized,

            // Newtype of a scalar, scalar pair or vector.
            (OperandValue::Immediate(_) | OperandValue::Pair(..), _)
                if field.layout.size == self.layout.layout.size =>
            {
                self.val
            }

            // Extract a scalar component from a pair.
            (OperandValue::Pair(a_llval, b_llval), ValueAbi::ScalarPair(..)) => {
                if is_first_field {
                    OperandValue::Immediate(a_llval)
                } else {
                    OperandValue::Immediate(b_llval)
                }
            }

            // `#[repr(simd)]` types are also immediate.
            (OperandValue::Immediate(llval), ValueAbi::Vector { .. }) => {
                OperandValue::Immediate(bx.extract_element(llval, bx.const_usize(i as u64)))
            }

            _ => bug!("OperandRef::extract_field({:?}): not applicable", self),
        };

        match (&mut val, field.layout.abi.clone()) {
            (OperandValue::ZeroSized, _) => {}
            (
                OperandValue::Immediate(llval),
                ValueAbi::Scalar(_) | ValueAbi::ScalarPair(..) | ValueAbi::Vector { .. },
            ) => {
                // Bools in union fields needs to be truncated.
                *llval = bx.to_immediate(*llval, &field);
            }
            (OperandValue::Pair(a, b), ValueAbi::ScalarPair(a_abi, b_abi)) => {
                // Bools in union fields needs to be truncated.
                *a = bx.to_immediate_scalar(*a, a_abi);
                *b = bx.to_immediate_scalar(*b, b_abi);
            }
            // Newtype vector of array, e.g. #[repr(simd)] struct S([i32; 4]);
            (OperandValue::Immediate(llval), ValueAbi::Aggregate { sized: true }) => {
                assert_matches!(&self.layout.layout.abi, ValueAbi::Vector { .. });

                let llfield_ty = bx.cx().backend_type(&field);

                // Can't bitcast an aggregate, so round trip through memory.
                let llptr = bx.alloca(&field);

                bx.store(*llval, llptr, field.layout.abi_align);

                *llval = bx.load(llfield_ty, llptr, field.layout.abi_align);
            }
            (OperandValue::Immediate(_), ValueAbi::Aggregate { sized: false })
            | (OperandValue::Pair(..), _)
            | (OperandValue::Ref(..), _) => bug!(),
        }

        OperandRef { val, layout: field }
    }
}

impl<'a, 'tcx, V: CodegenObject> OperandValue<V> {
    /// Returns an `OperandValue` that's generally UB to use in any way.
    ///
    /// Depending on the `layout`, returns `ZeroSized` for ZSTs, an `Immediate` or
    /// `Pair` containing poison value(s), or a `Ref` containing a poison pointer.
    ///
    /// Supports sized types only.
    pub fn poison<Bx: BuilderMethods<'a, Value = V>>(
        bx: &mut Bx,
        layout: TyAndLayout,
    ) -> OperandValue<V> {
        assert!(layout.layout.is_sized());

        if layout.layout.is_1zst() {
            OperandValue::ZeroSized
        } else if bx.cx().is_backend_immediate(&layout) {
            let ibty = bx.cx().immediate_backend_type(&layout);

            OperandValue::Immediate(bx.const_poison(ibty))
        } else if bx.cx().is_backend_scalar_pair(&layout) {
            let ibty0 = bx.cx().scalar_pair_element_backend_type(&layout, 0, true);
            let ibty1 = bx.cx().scalar_pair_element_backend_type(&layout, 1, true);

            OperandValue::Pair(bx.const_poison(ibty0), bx.const_poison(ibty1))
        } else {
            let ptr = bx.cx().type_ptr();

            OperandValue::Ref(PlaceValue::new_sized(
                bx.const_poison(ptr),
                layout.layout.abi_align,
            ))
        }
    }

    pub fn store<Bx: BuilderMethods<'a, Value = V>>(self, bx: &mut Bx, dest: &PlaceRef<V>) {
        debug!("OperandRef::store: operand={:?}, dest={:?}", self, dest);

        let dest_layout = &dest.layout.layout;

        match self {
            OperandValue::ZeroSized => {
                // Avoid generating stores of zero-sized values, because the only way to have a
                // zero-sized value is through `undef`/`poison`, and the store itself is useless.
            }
            OperandValue::Ref(val) => {
                assert!(
                    dest_layout.is_sized(),
                    "cannot directly store unsized values"
                );

                if val.llextra.is_some() {
                    bug!("cannot directly store unsized values");
                }

                bx.typed_place_copy(dest.val, val, &dest.layout);
            }
            OperandValue::Immediate(s) => {
                let val = bx.from_immediate(s);

                bx.store(val, dest.val.llval, dest.val.align);
            }
            OperandValue::Pair(a, b) => {
                let ValueAbi::ScalarPair(a_scalar, b_scalar) = &dest_layout.abi else {
                    bug!(
                        "store_with_flags: invalid ScalarPair layout: {:#?}",
                        dest.layout
                    );
                };

                let a_val = bx.from_immediate(a);
                let a_ptr = dest.project_field(bx, 0);

                bx.store(a_val, a_ptr.val.llval, a_ptr.val.align);

                let b_val = bx.from_immediate(b);
                let b_ptr = dest.project_field(bx, 1);

                bx.store(b_val, b_ptr.val.llval, b_ptr.val.align);
            }
        }
    }
    //
    // pub fn volatile_store<Bx: BuilderMethods<'a, Value = V>>(
    //     self,
    //     bx: &mut Bx,
    //     dest: PlaceRef<V>,
    // ) {
    //     self.store_with_flags(bx, dest, MemFlags::VOLATILE);
    // }
    //
    // pub fn unaligned_volatile_store<Bx: BuilderMethods<'a, Value = V>>(
    //     self,
    //     bx: &mut Bx,
    //     dest: PlaceRef<V>,
    // ) {
    //     self.store_with_flags(bx, dest, MemFlags::VOLATILE | MemFlags::UNALIGNED);
    // }
    //
    // pub fn nontemporal_store<Bx: BuilderMethods<'a, Value = V>>(
    //     self,
    //     bx: &mut Bx,
    //     dest: PlaceRef<V>,
    // ) {
    //     self.store_with_flags(bx, dest, MemFlags::NONTEMPORAL);
    // }
    //
    // pub(crate) fn store_with_flags<Bx: BuilderMethods<'a, Value = V>>(
    //     self,
    //     bx: &mut Bx,
    //     dest: PlaceRef<V>,
    //     flags: MemFlags,
    // ) {
    //     debug!("OperandRef::store: operand={:?}, dest={:?}", self, dest);
    //
    //     let dest_shape = dest.layout.layout.shape();
    //
    //     match self {
    //         OperandValue::ZeroSized => {
    //             // Avoid generating stores of zero-sized values, because the only way to have a
    //             // zero-sized value is through `undef`/`poison`, and the store itself is useless.
    //         }
    //         OperandValue::Ref(val) => {
    //             assert!(
    //                 dest_shape.is_sized(),
    //                 "cannot directly store unsized values"
    //             );
    //
    //             if val.llextra.is_some() {
    //                 bug!("cannot directly store unsized values");
    //             }
    //
    //             bx.typed_place_copy_with_flags(dest.val, val, dest.layout, flags);
    //         }
    //         OperandValue::Immediate(s) => {
    //             let val = bx.from_immediate(s);
    //
    //             bx.store_with_flags(val, dest.val.llval, dest.val.align, flags);
    //         }
    //         OperandValue::Pair(a, b) => {
    //             let ValueAbi::ScalarPair(a_scalar, b_scalar) = &dest_shape.abi else {
    //                 bug!(
    //                     "store_with_flags: invalid ScalarPair layout: {:#?}",
    //                     dest.layout
    //                 );
    //             };
    //             let align = dest.val.align;
    //
    //             let a_val = bx.from_immediate(a);
    //             let a_ptr = dest.project_field(bx,0);
    //
    //             bx.store_with_flags(a_val, a_ptr, align, flags);
    //
    //             let b_val = bx.from_immediate(b);
    //             let b_ptr = dest.project_field(bx,1);
    //
    //             bx.store_with_flags(b_val, b_ptr, align, flags);
    //         }
    //     }
    // }
}

impl<'a, Bx: BuilderMethods<'a>> FunctionCx<'a, Bx> {
    fn maybe_codegen_consume_direct(
        &mut self,
        bx: &mut Bx,
        place_ref: mir::visit::PlaceRef,
    ) -> Option<OperandRef<Bx::Value>> {
        match self.locals[place_ref.local].clone() {
            LocalRef::Operand(mut o) => {
                // Moves out of scalar and scalar pair fields are trivial.
                for elem in place_ref.projection.iter() {
                    match elem {
                        mir::ProjectionElem::Field(f, _) => {
                            o = o.extract_field(bx, *f);
                        }
                        mir::ProjectionElem::Index(_)
                        | mir::ProjectionElem::ConstantIndex { .. } => {
                            // ZSTs don't require any actual memory access.
                            let elem = o.layout.field(0);

                            if elem.layout.is_1zst() {
                                o = OperandRef::zero_sized(elem);
                            } else {
                                return None;
                            }
                        }
                        _ => return None,
                    }
                }

                Some(o)
            }
            LocalRef::PendingOperand => {
                bug!("use of place before def");
            }
            LocalRef::Place(..) | LocalRef::UnsizedPlace(..) => {
                // watch out for locals that do not have an
                // alloca; they are handled somewhat differently
                None
            }
        }
    }

    pub fn codegen_consume(
        &mut self,
        bx: &mut Bx,
        place_ref: mir::visit::PlaceRef,
    ) -> OperandRef<Bx::Value> {
        let ty = place_ref.ty(self.mir.locals()).unwrap();
        let layout = ty.layout().unwrap().shape();

        // ZSTs don't require any actual memory access.
        if layout.is_1zst() {
            return OperandRef::zero_sized(TyAndLayout { ty, layout });
        }

        // TODO: rustc_public::visit::PlaceRef currently does not implement Copy (it should)
        let place_ref_copy = mir::visit::PlaceRef {
            local: place_ref.local,
            projection: place_ref.projection,
        };

        if let Some(o) = self.maybe_codegen_consume_direct(bx, place_ref_copy) {
            return o;
        }

        // for most places, to consume them we just load them
        // out from their home
        let place = self.codegen_place(bx, place_ref);

        bx.load_operand(&place)
    }

    pub fn codegen_operand(
        &mut self,
        bx: &mut Bx,
        operand: &mir::Operand,
    ) -> OperandRef<Bx::Value> {
        debug!("codegen_operand(operand={:?})", operand);

        match operand {
            mir::Operand::Copy(place) | mir::Operand::Move(place) => self.codegen_consume(
                bx,
                mir::visit::PlaceRef {
                    local: place.local,
                    projection: &place.projection,
                },
            ),
            mir::Operand::Constant(constant) => OperandRef::from_const(bx, &constant.const_),
        }
    }
}
