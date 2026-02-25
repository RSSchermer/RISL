use rustc_middle::bug;
use rustc_public::abi::{FieldsShape, VariantsShape};
use rustc_public::mir;
use rustc_public::mir::Mutability;
use rustc_public::ty::{Align, RigidTy, Ty, TyKind, VariantIdx};
use tracing::instrument;

use super::operand::OperandValue;
use super::{FunctionCx, LocalRef};
use crate::stable_cg::layout::TyAndLayout;
use crate::stable_cg::traits::*;

/// The location and extra runtime properties of the place.
///
/// Typically found in a [`PlaceRef`] or an [`OperandValue::Ref`].
///
/// As a location in memory, this has no specific type. If you want to
/// load or store it using a typed operation, use [`Self::with_type`].
#[derive(Copy, Clone, Debug)]
pub struct PlaceValue<V> {
    /// A pointer to the contents of the place.
    pub llval: V,

    /// This place's extra data if it is unsized, or `None` if null.
    pub llextra: Option<V>,

    /// The alignment we know for this place.
    pub align: Align,
}

impl<V: CodegenObject> PlaceValue<V> {
    /// Constructor for the ordinary case of `Sized` types.
    ///
    /// Sets `llextra` to `None`.
    pub fn new_sized(llval: V, align: Align) -> PlaceValue<V> {
        PlaceValue {
            llval,
            llextra: None,
            align,
        }
    }

    /// Allocates a stack slot in the function for a value
    /// of the specified size and alignment.
    ///
    /// The allocation itself is untyped.
    pub fn alloca<'a, Bx: BuilderMethods<'a, Value = V>>(
        bx: &mut Bx,
        layout: &TyAndLayout,
    ) -> PlaceValue<V> {
        let llval = bx.alloca(layout);

        PlaceValue::new_sized(llval, layout.layout.abi_align)
    }

    /// Creates a `PlaceRef` to this location with the given type.
    pub fn with_type(self, layout: TyAndLayout) -> PlaceRef<V> {
        PlaceRef { val: self, layout }
    }

    /// Gets the pointer to this place as an [`OperandValue::Immediate`]
    /// or, for those needing metadata, an [`OperandValue::Pair`].
    ///
    /// This is the inverse of [`OperandValue::deref`].
    pub fn address(self) -> OperandValue<V> {
        if let Some(llextra) = self.llextra {
            OperandValue::Pair(self.llval, llextra)
        } else {
            OperandValue::Immediate(self.llval)
        }
    }
}

#[derive(Clone, Debug)]
pub struct PlaceRef<V> {
    /// The location and extra runtime properties of the place.
    pub val: PlaceValue<V>,

    /// The monomorphized type of this place, including variant information.
    ///
    /// You probably shouldn't use the alignment from this layout;
    /// rather you should use the `.val.align` of the actual place,
    /// which might be different from the type's normal alignment.
    pub layout: TyAndLayout,
}

impl<'a, V: CodegenObject> PlaceRef<V> {
    pub fn new_sized(llval: V, layout: TyAndLayout) -> PlaceRef<V> {
        let align = layout.layout.abi_align;

        PlaceRef::new_sized_aligned(llval, layout, align)
    }

    pub fn new_sized_aligned(llval: V, layout: TyAndLayout, align: Align) -> PlaceRef<V> {
        assert!(layout.layout.is_sized());

        PlaceValue::new_sized(llval, align).with_type(layout)
    }

    // FIXME(eddyb) pass something else for the name so no work is done
    // unless LLVM IR names are turned on (e.g. for `--emit=llvm-ir`).
    pub fn alloca<Bx: BuilderMethods<'a, Value = V>>(bx: &mut Bx, layout: TyAndLayout) -> Self {
        assert!(
            layout.layout.is_sized(),
            "tried to statically allocate unsized place"
        );

        PlaceValue::alloca(bx, &layout).with_type(layout)
    }

    /// Returns a place for an indirect reference to an unsized place.
    // FIXME(eddyb) pass something else for the name so no work is done
    // unless LLVM IR names are turned on (e.g. for `--emit=llvm-ir`).
    pub fn alloca_unsized_indirect<Bx: BuilderMethods<'a, Value = V>>(
        bx: &mut Bx,
        layout: TyAndLayout,
    ) -> Self {
        assert!(
            layout.layout.is_unsized(),
            "tried to allocate indirect place for sized values"
        );

        let ptr_ty = Ty::new_ptr(layout.ty, Mutability::Mut);

        Self::alloca(bx, TyAndLayout::expect_from_ty(ptr_ty))
    }

    pub fn len<Cx: ConstCodegenMethods<Value = V>>(&self, cx: &Cx) -> V {
        if let FieldsShape::Array { count, .. } = &self.layout.layout.fields {
            if self.layout.layout.is_unsized() {
                assert_eq!(*count, 0);

                self.val.llextra.unwrap()
            } else {
                cx.const_usize(*count)
            }
        } else {
            bug!("unexpected layout `{:#?}` in PlaceRef::len", self.layout)
        }
    }
}

impl<'a, V: CodegenObject> PlaceRef<V> {
    /// Access a field, at a point when the value's case is known.
    pub fn project_field<Bx: BuilderMethods<'a, Value = V>>(&self, bx: &mut Bx, ix: usize) -> Self {
        let field = self.layout.field(ix);
        let ptr = bx.element_ptr(
            bx.backend_type(&field),
            self.val.llval,
            &[bx.const_u32(ix as u32)],
        );
        let val = PlaceValue {
            llval: ptr,
            llextra: self.val.llextra,
            align: field.layout.abi_align,
        };

        val.with_type(field)
    }

    pub fn project_index<Bx: BuilderMethods<'a, Value = V>>(
        &self,
        bx: &mut Bx,
        llindex: V,
    ) -> Self {
        let layout = self.layout.field(0);
        let llval = bx.element_ptr(bx.backend_type(&layout), self.val.llval, &[llindex]);

        PlaceValue::new_sized(llval, layout.layout.abi_align).with_type(layout)
    }

    pub fn project_subslice<Bx: BuilderMethods<'a, Value = V>>(
        &self,
        bx: &mut Bx,
        lloffset: V,
        slice_layout: TyAndLayout,
    ) -> Self {
        let llval = bx.offset_slice_ptr(self.val.llval, lloffset, bx.backend_type(&slice_layout));

        PlaceValue::new_sized(llval, slice_layout.layout.abi_align).with_type(slice_layout)
    }

    /// Obtain the actual discriminant of a value.
    #[instrument(level = "trace", skip(bx))]
    pub fn codegen_get_discr<Bx: BuilderMethods<'a, Value = V>>(&self, bx: &mut Bx) -> V {
        bx.get_discriminant(self.val.llval)
    }

    /// Sets the discriminant for a new value of the given case of the given
    /// representation.
    pub fn codegen_set_discr<Bx: BuilderMethods<'a, Value = V>>(
        &self,
        bx: &mut Bx,
        variant_index: VariantIdx,
    ) {
        match self.layout.layout.variants {
            VariantsShape::Empty => bug!("cannot set discriminant on a type without variants"),
            VariantsShape::Single { index } => assert_eq!(index, variant_index),
            VariantsShape::Multiple { .. } => {
                bx.set_discriminant(self.val.llval, variant_index);
            }
        }
    }

    pub fn project_downcast<Bx: BuilderMethods<'a, Value = V>>(
        &self,
        bx: &mut Bx,
        variant_index: VariantIdx,
    ) -> Self {
        match &self.layout.layout.variants {
            VariantsShape::Empty => (*self).clone(),
            VariantsShape::Single { index } => {
                assert_eq!(*index, variant_index);

                (*self).clone()
            }
            VariantsShape::Multiple { .. } => {
                let mut downcast = (*self).clone();

                downcast.val.llval = bx.ptr_variant_ptr(self.val.llval, variant_index);
                downcast.layout = downcast.layout.for_variant(variant_index);

                downcast
            }
        }
    }

    // pub fn project_type<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
    //     &self,
    //     bx: &mut Bx,
    //     ty: Ty<'tcx>,
    // ) -> Self {
    //     let mut downcast = *self;
    //     downcast.layout = bx.cx().layout_of(ty);
    //     downcast
    // }

    pub fn storage_live<Bx: BuilderMethods<'a, Value = V>>(&self, bx: &mut Bx) {
        bx.lifetime_start(self.val.llval, self.layout.layout.size);
    }

    pub fn storage_dead<Bx: BuilderMethods<'a, Value = V>>(&self, bx: &mut Bx) {
        bx.lifetime_end(self.val.llval, self.layout.layout.size);
    }
}

impl<'a, Bx: BuilderMethods<'a>> FunctionCx<'a, Bx> {
    pub fn codegen_place(
        &mut self,
        bx: &mut Bx,
        place_ref: mir::visit::PlaceRef,
    ) -> PlaceRef<Bx::Value> {
        let cx = self.cx;

        let mut base = 0;

        let mut cg_base = match &self.locals[place_ref.local] {
            LocalRef::Place(place) => (*place).clone(),
            LocalRef::UnsizedPlace(place) => bx.load_operand(place).deref(cx),
            LocalRef::Operand(..) => {
                if matches!(
                    place_ref.projection.first(),
                    Some(mir::ProjectionElem::Deref)
                ) {
                    base = 1;

                    let cg_base = self.codegen_consume(
                        bx,
                        mir::visit::PlaceRef {
                            local: place_ref.local,
                            projection: &[],
                        },
                    );

                    cg_base.deref(cx)
                } else {
                    bug!("using operand local as place");
                }
            }
            LocalRef::PendingOperand => {
                bug!("using still-pending operand local as place");
            }
        };

        for elem in place_ref.projection[base..].iter() {
            cg_base = match *elem {
                mir::ProjectionElem::Deref => bx.load_operand(&cg_base).deref(cx),
                mir::ProjectionElem::Field(field, _) => cg_base.project_field(bx, field),
                mir::ProjectionElem::OpaqueCast(ty) => {
                    bug!("encountered OpaqueCast({ty}) in codegen")
                }
                mir::ProjectionElem::Index(index) => {
                    let index = &mir::Operand::Copy(mir::Place::from(index));
                    let index = self.codegen_operand(bx, index);
                    let llindex = index.immediate();

                    cg_base.project_index(bx, llindex)
                }
                mir::ProjectionElem::ConstantIndex {
                    offset,
                    from_end: false,
                    min_length: _,
                } => {
                    let lloffset = bx.const_usize(offset);

                    cg_base.project_index(bx, lloffset)
                }
                mir::ProjectionElem::ConstantIndex {
                    offset,
                    from_end: true,
                    min_length: _,
                } => {
                    let lloffset = bx.const_usize(offset);
                    let lllen = cg_base.len(cx);
                    let llindex = bx.sub(lllen, lloffset);

                    cg_base.project_index(bx, llindex)
                }
                mir::ProjectionElem::Subslice { from, to, from_end } => {
                    let TyKind::RigidTy(ty) = cg_base.layout.ty.kind() else {
                        bug!("type should be rigid")
                    };

                    let projected_ty = match ty {
                        RigidTy::Slice(..) => cg_base.layout.ty,
                        RigidTy::Array(inner, _) if !from_end => {
                            Ty::try_new_array(inner, to - from).unwrap()
                        }
                        RigidTy::Array(inner, size) if from_end => {
                            let size = size
                                .eval_target_usize()
                                .expect("expected subslice projection on fixed-size array");
                            let len = size - from - to;

                            Ty::try_new_array(inner, len).unwrap()
                        }
                        _ => bug!("cannot subslice non-array type: `{:?}`", cg_base.layout.ty),
                    };

                    let layout = TyAndLayout::expect_from_ty(projected_ty);
                    let mut subslice = cg_base.project_subslice(bx, bx.const_usize(from), layout);

                    if subslice.layout.layout.is_unsized() {
                        assert!(from_end, "slice subslices should be `from_end`");

                        subslice.val.llextra =
                            Some(bx.sub(cg_base.val.llextra.unwrap(), bx.const_usize(from + to)));
                    }

                    subslice
                }
                mir::ProjectionElem::Downcast(v) => cg_base.project_downcast(bx, v),
            };
        }

        cg_base
    }
}
