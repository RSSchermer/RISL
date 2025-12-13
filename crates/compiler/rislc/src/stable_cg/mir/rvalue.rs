use std::assert_matches::assert_matches;

use arrayvec::ArrayVec;
use rustc_middle::{bug, span_bug};
use rustc_public::abi::ValueAbi;
use rustc_public::mir::{AggregateKind, CastKind, Mutability, NullOp, PointerCoercion, Rvalue};
use rustc_public::ty::{Region, RegionKind, RigidTy, Span, Ty, TyKind, UintTy, VariantIdx};
use rustc_public::{abi, mir};
use rustc_public_bridge::IndexedVal;
use tracing::{debug, instrument, trace};

use super::operand::{OperandRef, OperandValue};
use super::place::PlaceRef;
use super::{FunctionCx, LocalRef};
use crate::stable_cg::common::{IntPredicate, RealPredicate, TypeKind};
use crate::stable_cg::layout::{ScalarExt, TyAndLayout};
use crate::stable_cg::traits::*;

pub(crate) fn shift_mask_val<'a, Bx: BuilderMethods<'a>>(
    bx: &mut Bx,
    llty: Bx::Type,
    mask_llty: Bx::Type,
    invert: bool,
) -> Bx::Value {
    let kind = bx.type_kind(llty);

    match kind {
        TypeKind::Integer => {
            // i8/u8 can shift by at most 7, i16/u16 by at most 15, etc.
            let val = bx.int_width(llty) - 1;

            if invert {
                bx.const_int(mask_llty, !val as i64)
            } else {
                bx.const_uint(mask_llty, val)
            }
        }
        TypeKind::Vector => {
            let mask = shift_mask_val(
                bx,
                bx.element_type(llty),
                bx.element_type(mask_llty),
                invert,
            );
            bx.vector_splat(bx.vector_length(mask_llty), mask)
        }
        _ => bug!(
            "shift_mask_val: expected Integer or Vector, found {:?}",
            kind
        ),
    }
}

/// Returns `rhs` sufficiently masked, truncated, and/or extended so that it can be used to shift
/// `lhs`: it has the same size as `lhs`, and the value, when interpreted unsigned (no matter its
/// type), will not exceed the size of `lhs`.
///
/// Shifts in MIR are all allowed to have mismatched LHS & RHS types, and signed RHS.
/// The shift methods in `BuilderMethods`, however, are fully homogeneous
/// (both parameters and the return type are all the same size) and assume an unsigned RHS.
///
/// If `is_unchecked` is false, this masks the RHS to ensure it stays in-bounds,
/// as the `BuilderMethods` shifts are UB for out-of-bounds shift amounts.
/// For 32- and 64-bit types, this matches the semantics
/// of Java. (See related discussion on #1877 and #10183.)
///
/// If `is_unchecked` is true, this does no masking, and adds sufficient `assume`
/// calls or operation flags to preserve as much freedom to optimize as possible.
pub(crate) fn build_shift_expr_rhs<'a, Bx: BuilderMethods<'a>>(
    bx: &mut Bx,
    lhs: Bx::Value,
    mut rhs: Bx::Value,
    is_unchecked: bool,
) -> Bx::Value {
    // Shifts may have any size int on the rhs
    let mut rhs_llty = bx.cx().val_ty(rhs);
    let mut lhs_llty = bx.cx().val_ty(lhs);

    let mask = shift_mask_val(bx, lhs_llty, rhs_llty, false);

    if !is_unchecked {
        rhs = bx.and(rhs, mask);
    }

    if bx.type_kind(rhs_llty) == TypeKind::Vector {
        rhs_llty = bx.element_type(rhs_llty)
    }

    if bx.type_kind(lhs_llty) == TypeKind::Vector {
        lhs_llty = bx.element_type(lhs_llty)
    }

    let rhs_sz = bx.int_width(rhs_llty);
    let lhs_sz = bx.int_width(lhs_llty);

    if lhs_sz < rhs_sz {
        bx.trunc(rhs, lhs_llty)
    } else if lhs_sz > rhs_sz {
        // We zero-extend even if the RHS is signed. So e.g. `(x: i32) << -1i8` will zero-extend the
        // RHS to `255i32`. But then we mask the shift amount to be within the size of the LHS
        // anyway so the result is `31` as it should be. All the extra bits introduced by zext
        // are masked off so their value does not matter.
        // FIXME: if we ever support 512bit integers, this will be wrong! For such large integers,
        // the extra bits introduced by zext are *not* all masked away any more.
        assert!(lhs_sz <= 256);

        bx.zext(rhs, lhs_llty)
    } else {
        rhs
    }
}

fn bin_op_to_icmp_predicate(op: mir::BinOp, signed: bool) -> IntPredicate {
    match (op, signed) {
        (mir::BinOp::Eq, _) => IntPredicate::IntEQ,
        (mir::BinOp::Ne, _) => IntPredicate::IntNE,
        (mir::BinOp::Lt, true) => IntPredicate::IntSLT,
        (mir::BinOp::Lt, false) => IntPredicate::IntULT,
        (mir::BinOp::Le, true) => IntPredicate::IntSLE,
        (mir::BinOp::Le, false) => IntPredicate::IntULE,
        (mir::BinOp::Gt, true) => IntPredicate::IntSGT,
        (mir::BinOp::Gt, false) => IntPredicate::IntUGT,
        (mir::BinOp::Ge, true) => IntPredicate::IntSGE,
        (mir::BinOp::Ge, false) => IntPredicate::IntUGE,
        op => bug!(
            "bin_op_to_icmp_predicate: expected comparison operator, found {:?}",
            op
        ),
    }
}

fn bin_op_to_fcmp_predicate(op: mir::BinOp) -> RealPredicate {
    match op {
        mir::BinOp::Eq => RealPredicate::RealOEQ,
        mir::BinOp::Ne => RealPredicate::RealUNE,
        mir::BinOp::Lt => RealPredicate::RealOLT,
        mir::BinOp::Le => RealPredicate::RealOLE,
        mir::BinOp::Gt => RealPredicate::RealOGT,
        mir::BinOp::Ge => RealPredicate::RealOGE,
        op => bug!(
            "bin_op_to_fcmp_predicate: expected comparison operator, found {:?}",
            op
        ),
    }
}

/// Coerces `src`, which is a reference to a value of type `src_ty`,
/// to a value of type `dst_ty`, and stores the result in `dst`.
fn coerce_unsized_into<'a, Bx: BuilderMethods<'a>>(
    bx: &mut Bx,
    src: &PlaceRef<Bx::Value>,
    dst: &PlaceRef<Bx::Value>,
) {
    let src_ty = src.layout.ty;
    let dst_ty = dst.layout.ty;

    let TyKind::RigidTy(src_kind) = src_ty.kind() else {
        bug!("can only unsize rigid types")
    };

    let TyKind::RigidTy(dst_kind) = dst_ty.kind() else {
        bug!("can only unsize rigid types")
    };

    use RigidTy::*;

    match (src_kind, dst_kind) {
        (Ref(..), Ref(..) | RawPtr(..)) | (RawPtr(..), RawPtr(..)) => {
            let (base, info) = match bx.load_operand(src).val {
                OperandValue::Pair(base, info) => unsize_ptr(bx, base, src_ty, dst_ty, Some(info)),
                OperandValue::Immediate(base) => unsize_ptr(bx, base, src_ty, dst_ty, None),
                OperandValue::Ref(..) | OperandValue::ZeroSized => bug!(),
            };

            OperandValue::Pair(base, info).store(bx, dst);
        }

        (Adt(def_a, _), Adt(def_b, _)) => {
            assert_eq!(def_a, def_b); // implies same number of fields

            let field_count = def_a.variant(VariantIdx::to_val(0)).unwrap().fields().len();

            for i in 0..field_count {
                let src_field = src.project_field(bx, i);
                let dst_field = dst.project_field(bx, i);

                if dst_field.layout.layout.is_1zst() {
                    // No data here, nothing to copy/coerce.
                    continue;
                }

                if src_field.layout.ty == dst_field.layout.ty {
                    bx.typed_place_copy(dst_field.val, src_field.val, &src_field.layout);
                } else {
                    coerce_unsized_into(bx, &src_field, &dst_field);
                }
            }
        }
        _ => bug!(
            "coerce_unsized_into: invalid coercion {:?} -> {:?}",
            src_ty,
            dst_ty,
        ),
    }
}

/// Retrieves the information we are losing (making dynamic) in an unsizing
/// adjustment.
///
/// The `old_info` argument is a bit odd. It is intended for use in an upcast,
/// where the new vtable for an object will be derived from the old one.
fn unsized_info<'a, Bx: BuilderMethods<'a>>(
    bx: &mut Bx,
    source: Ty,
    target: Ty,
    old_info: Option<Bx::Value>,
) -> Bx::Value {
    let TyKind::RigidTy(src_kind) = source.kind() else {
        bug!("can only unsize rigid types")
    };

    let TyKind::RigidTy(dst_kind) = target.kind() else {
        bug!("can only unsize rigid types")
    };

    use RigidTy::*;

    match (src_kind, dst_kind) {
        (Array(_, len), Slice(_)) => {
            let len = len
                .eval_target_usize()
                .expect("expected monomorphic const in codegen");

            bx.const_usize(len)
        }
        (_, Dynamic(..)) => {
            bug!("dyn unsizing is not supported by RISL")
        }
        _ => bug!(
            "unsized_info: invalid unsizing {:?} -> {:?}",
            source,
            target
        ),
    }
}

/// Coerces `src` to `dst_ty`. `src_ty` must be a pointer.
fn unsize_ptr<'a, Bx: BuilderMethods<'a>>(
    bx: &mut Bx,
    src: Bx::Value,
    src_ty: Ty,
    dst_ty: Ty,
    old_info: Option<Bx::Value>,
) -> (Bx::Value, Bx::Value) {
    debug!("unsize_ptr: {:?} => {:?}", src_ty, dst_ty);

    let TyKind::RigidTy(src_kind) = src_ty.kind() else {
        bug!("can only unsize rigid types")
    };

    let TyKind::RigidTy(dst_kind) = dst_ty.kind() else {
        bug!("can only unsize rigid types")
    };

    use RigidTy::*;

    match (src_kind, dst_kind) {
        (Ref(_, a, _), Ref(_, b, _) | RawPtr(b, _)) | (RawPtr(a, _), RawPtr(b, _)) => {
            assert_eq!(a.layout().unwrap().shape().is_sized(), old_info.is_none());

            (src, unsized_info(bx, a, b, old_info))
        }
        (Adt(def_a, _), Adt(def_b, _)) => {
            assert_eq!(def_a, def_b); // implies same number of fields

            let src_layout = TyAndLayout {
                ty: src_ty,
                layout: src_ty.layout().unwrap().shape(),
            };
            let dst_layout = TyAndLayout {
                ty: dst_ty,
                layout: dst_ty.layout().unwrap().shape(),
            };

            if src_ty == dst_ty {
                return (src, old_info.unwrap());
            }

            let mut result = None;

            for i in 0..src_layout.layout.fields.count() {
                let src_field = src_layout.field(i);

                if src_field.layout.is_1zst() {
                    // We are looking for the one non-1-ZST field; this is not it.
                    continue;
                }

                let dst_field = dst_layout.field(i);

                assert_ne!(src_field.ty, dst_field.ty);
                assert_eq!(result, None);

                result = Some(unsize_ptr(bx, src, src_field.ty, dst_field.ty, old_info));
            }
            result.unwrap()
        }
        _ => bug!("unsize_ptr: called on bad types"),
    }
}

impl<'a, Bx: BuilderMethods<'a>> FunctionCx<'a, Bx> {
    #[instrument(level = "trace", skip(self, bx))]
    pub(crate) fn codegen_rvalue(
        &mut self,
        bx: &mut Bx,
        dest: &PlaceRef<Bx::Value>,
        rvalue: &mir::Rvalue,
    ) {
        match rvalue {
            mir::Rvalue::Use(operand) => {
                let cg_operand = self.codegen_operand(bx, operand);
                // FIXME: consider not copying constants through stack. (Fixable by codegen'ing
                // constants into `OperandValue::Ref`; why don’t we do that yet if we don’t?)
                cg_operand.val.store(bx, dest);
            }

            mir::Rvalue::Cast(
                mir::CastKind::PointerCoercion(PointerCoercion::Unsize),
                source,
                _,
            ) => {
                // The destination necessarily contains a wide pointer, so if
                // it's a scalar pair, it's a wide pointer or newtype thereof.
                if bx.cx().is_backend_scalar_pair(&dest.layout) {
                    // Into-coerce of a thin pointer to a wide pointer -- just
                    // use the operand path.
                    let temp = self.codegen_rvalue_operand(bx, rvalue);
                    temp.val.store(bx, dest);
                    return;
                }

                // Unsize of a nontrivial struct. I would prefer for
                // this to be eliminated by MIR building, but
                // `CoerceUnsized` can be passed by a where-clause,
                // so the (generic) MIR may not be able to expand it.
                let operand = self.codegen_operand(bx, source);

                match operand.val {
                    OperandValue::Pair(..) | OperandValue::Immediate(_) => {
                        // Unsize from an immediate structure. We don't
                        // really need a temporary alloca here, but
                        // avoiding it would require us to have
                        // `coerce_unsized_into` use `extractvalue` to
                        // index into the struct, and this case isn't
                        // important enough for it.
                        debug!("codegen_rvalue: creating ugly alloca");

                        let scratch = PlaceRef::alloca(bx, operand.layout);

                        scratch.storage_live(bx);
                        operand.val.store(bx, &scratch);
                        coerce_unsized_into(bx, &scratch, dest);
                        scratch.storage_dead(bx);
                    }
                    OperandValue::Ref(val) => {
                        if val.llextra.is_some() {
                            bug!("unsized coercion on an unsized rvalue");
                        }

                        coerce_unsized_into(bx, &val.with_type(operand.layout), dest);
                    }
                    OperandValue::ZeroSized => {
                        bug!("unsized coercion on a ZST rvalue");
                    }
                }
            }

            mir::Rvalue::Cast(mir::CastKind::Transmute, ..) => {
                bug!("r-value {:?} not supported by RISL", rvalue)
            }

            mir::Rvalue::Repeat(elem, count) => {
                let cg_elem = self.codegen_operand(bx, elem);

                // Do not generate the loop for zero-sized elements or empty arrays.
                if dest.layout.layout.is_1zst() {
                    return;
                }

                trace!(?cg_elem.val);

                let count = count
                    .eval_target_usize()
                    .expect("expected monomorphic const in codegen");

                bx.write_operand_repeatedly(&cg_elem, count, dest);
            }

            // This implementation does field projection, so never use it for `RawPtr`,
            // which will always be fine with the `codegen_rvalue_operand` path below.
            mir::Rvalue::Aggregate(kind, operands)
                if !matches!(kind, mir::AggregateKind::RawPtr(..)) =>
            {
                let (variant_index, variant_dest, active_field_index) = match kind {
                    mir::AggregateKind::Adt(_, variant_index, _, _, active_field_index) => {
                        let variant_dest = dest.project_downcast(bx, *variant_index);

                        (*variant_index, variant_dest, *active_field_index)
                    }
                    _ => (VariantIdx::to_val(0), dest.clone(), None),
                };

                if active_field_index.is_some() {
                    assert_eq!(operands.len(), 1);
                }

                for (i, operand) in operands.iter().enumerate() {
                    let op = self.codegen_operand(bx, operand);

                    // Do not generate stores for zero-sized fields.
                    if !op.layout.layout.is_1zst() {
                        let field_index = active_field_index.unwrap_or(i);

                        let field = if matches!(kind, mir::AggregateKind::Array(_)) {
                            let llindex = bx.const_usize(field_index as u64);

                            variant_dest.project_index(bx, llindex)
                        } else {
                            variant_dest.project_field(bx, field_index)
                        };

                        op.val.store(bx, &field);
                    }
                }

                dest.codegen_set_discr(bx, variant_index);
            }

            _ => {
                assert!(self.rvalue_creates_operand(rvalue));

                let temp = self.codegen_rvalue_operand(bx, rvalue);
                temp.val.store(bx, dest);
            }
        }
    }

    /// Cast one of the immediates from an [`OperandValue::Immediate`]
    /// or an [`OperandValue::Pair`] to an immediate of the target type.
    ///
    /// Returns `None` if the cast is not possible.
    fn cast_immediate(
        &self,
        bx: &mut Bx,
        mut imm: Bx::Value,
        from_scalar: abi::Scalar,
        from_backend_ty: Bx::Type,
        to_scalar: abi::Scalar,
        to_backend_ty: Bx::Type,
    ) -> Option<Bx::Value> {
        use abi::Primitive::*;

        imm = match (from_scalar.primitive(), to_scalar.primitive()) {
            (Pointer(..), Pointer(..)) | (Int { .. }, Pointer(..)) => {
                bug!("pointer casting is not supported by RISL")
            }
            (Int { signed, .. }, Int { .. }) => bx.intcast(imm, to_backend_ty, *signed),
            (Float { .. }, Float { .. }) => {
                let src_size = bx.float_width(from_backend_ty);
                let dst_size = bx.float_width(to_backend_ty);

                if dst_size > src_size {
                    bx.fpext(imm, to_backend_ty)
                } else if src_size > dst_size {
                    bx.fptrunc(imm, to_backend_ty)
                } else {
                    imm
                }
            }
            (Int { signed, .. }, Float { .. }) => {
                if *signed {
                    bx.sitofp(imm, to_backend_ty)
                } else {
                    bx.uitofp(imm, to_backend_ty)
                }
            }
            (Float { .. }, Int { signed, .. }) => bx.cast_float_to_int(*signed, imm, to_backend_ty),
            _ => return None,
        };
        Some(imm)
    }

    pub(crate) fn codegen_rvalue_operand(
        &mut self,
        bx: &mut Bx,
        rvalue: &mir::Rvalue,
    ) -> OperandRef<Bx::Value> {
        assert!(
            self.rvalue_creates_operand(rvalue),
            "cannot codegen {rvalue:?} to operand",
        );

        match rvalue {
            mir::Rvalue::Cast(kind, source, mir_cast_ty) => {
                let operand = self.codegen_operand(bx, source);

                debug!("cast operand is {:?}", operand);

                let layout = mir_cast_ty.layout().unwrap().shape();
                let cast = TyAndLayout {
                    ty: *mir_cast_ty,
                    layout,
                };

                let val = match *kind {
                    mir::CastKind::PointerCoercion(
                        PointerCoercion::MutToConstPointer | PointerCoercion::ArrayToPointer,
                    ) => {
                        bug!("{kind:?} is for borrowck, and should never appear in codegen");
                    }
                    mir::CastKind::PtrToPtr
                    | mir::CastKind::FnPtrToPtr
                    | mir::CastKind::PointerExposeAddress
                    | mir::CastKind::PointerWithExposedProvenance
                    | mir::CastKind::Transmute
                    | mir::CastKind::Subtype
                    | mir::CastKind::PointerCoercion(PointerCoercion::ReifyFnPointer(_))
                    | mir::CastKind::PointerCoercion(PointerCoercion::ClosureFnPointer(_))
                    | mir::CastKind::PointerCoercion(PointerCoercion::UnsafeFnPointer) => {
                        bug!("cast not supported by RISL")
                    }
                    mir::CastKind::PointerCoercion(PointerCoercion::Unsize) => {
                        assert!(bx.cx().is_backend_scalar_pair(&cast));

                        let (lldata, llextra) = operand.val.pointer_parts();
                        let (lldata, llextra) =
                            unsize_ptr(bx, lldata, operand.layout.ty, cast.ty, llextra);

                        OperandValue::Pair(lldata, llextra)
                    }
                    mir::CastKind::IntToInt
                    | mir::CastKind::FloatToInt
                    | mir::CastKind::FloatToFloat
                    | mir::CastKind::IntToFloat => {
                        let imm = operand.clone().immediate();
                        let operand_kind = self.value_kind(&operand.layout);

                        let OperandValueKind::Immediate(from_scalar) = operand_kind else {
                            bug!("Found {operand_kind:?} for operand {operand:?}");
                        };

                        let from_backend_ty = bx.cx().immediate_backend_type(&operand.layout);

                        assert!(bx.is_backend_immediate(&cast));

                        let to_backend_ty = bx.immediate_backend_type(&cast);
                        let cast_kind = self.value_kind(&cast);
                        let OperandValueKind::Immediate(to_scalar) = cast_kind else {
                            bug!("Found {cast_kind:?} for operand {cast:?}");
                        };

                        self.cast_immediate(
                            bx,
                            imm,
                            from_scalar,
                            from_backend_ty,
                            to_scalar,
                            to_backend_ty,
                        )
                        .map(OperandValue::Immediate)
                        .unwrap_or_else(|| {
                            bug!("Unsupported cast of {operand:?} to {cast:?}");
                        })
                    }
                };
                OperandRef { val, layout: cast }
            }

            mir::Rvalue::Ref(_, bk, place) => {
                let mk_ref = move |ty: Ty| {
                    Ty::new_ref(
                        Region {
                            kind: RegionKind::ReErased,
                        },
                        ty,
                        bk.to_mutable_lossy(),
                    )
                };

                self.codegen_place_to_pointer(bx, place, mk_ref)
            }

            mir::Rvalue::CopyForDeref(place) => {
                self.codegen_operand(bx, &mir::Operand::Copy(place.clone()))
            }

            mir::Rvalue::Len(place) => {
                let size = self.evaluate_array_len(bx, place);
                let ty = Ty::usize_ty();

                OperandRef {
                    val: OperandValue::Immediate(size),
                    layout: TyAndLayout::expect_from_ty(ty),
                }
            }

            mir::Rvalue::BinaryOp(op, lhs, rhs) | mir::Rvalue::CheckedBinaryOp(op, lhs, rhs) => {
                let lhs = self.codegen_operand(bx, lhs);
                let rhs = self.codegen_operand(bx, rhs);

                let llresult = match (lhs.val, rhs.val) {
                    (
                        OperandValue::Pair(lhs_addr, lhs_extra),
                        OperandValue::Pair(rhs_addr, rhs_extra),
                    ) => self.codegen_wide_ptr_binop(
                        bx,
                        *op,
                        lhs_addr,
                        lhs_extra,
                        rhs_addr,
                        rhs_extra,
                        lhs.layout.ty,
                    ),

                    (OperandValue::Immediate(lhs_val), OperandValue::Immediate(rhs_val)) => {
                        self.codegen_scalar_binop(bx, *op, lhs_val, rhs_val, lhs.layout.ty)
                    }

                    _ => bug!(),
                };

                let ty = op.ty(lhs.layout.ty, rhs.layout.ty);

                OperandRef {
                    val: OperandValue::Immediate(llresult),
                    layout: TyAndLayout::expect_from_ty(ty),
                }
            }

            mir::Rvalue::UnaryOp(op, operand) => {
                let operand = self.codegen_operand(bx, operand);
                let is_float = operand.layout.ty.kind().is_float();

                let (val, layout) = match op {
                    mir::UnOp::Not => {
                        let layout = operand.layout.clone();
                        let llval = bx.not(operand.immediate());

                        (OperandValue::Immediate(llval), layout)
                    }
                    mir::UnOp::Neg => {
                        let layout = operand.layout.clone();

                        let llval = if is_float {
                            bx.fneg(operand.immediate())
                        } else {
                            bx.neg(operand.immediate())
                        };

                        (OperandValue::Immediate(llval), layout)
                    }
                    mir::UnOp::PtrMetadata => {
                        let (_, meta) = operand.val.pointer_parts();

                        if let Some(meta) = meta {
                            (OperandValue::Immediate(meta), operand.layout.field(1))
                        } else {
                            (
                                OperandValue::ZeroSized,
                                TyAndLayout::expect_from_ty(Ty::new_tuple(&[])),
                            )
                        }
                    }
                };
                assert!(
                    val.is_expected_variant_for_type(self.cx, &layout),
                    "Made wrong variant {val:?} for type {layout:?}",
                );
                OperandRef { val, layout }
            }

            mir::Rvalue::Discriminant(place) => {
                let discr = self
                    .codegen_place(
                        bx,
                        mir::visit::PlaceRef {
                            local: place.local,
                            projection: &place.projection,
                        },
                    )
                    .codegen_get_discr(bx);
                let ty = Ty::unsigned_ty(UintTy::U32);

                OperandRef {
                    val: OperandValue::Immediate(discr),
                    layout: TyAndLayout::expect_from_ty(ty),
                }
            }

            mir::Rvalue::Use(operand) => self.codegen_operand(bx, operand),
            mir::Rvalue::Repeat(..) => bug!("{rvalue:?} in codegen_rvalue_operand"),
            mir::Rvalue::Aggregate(_, fields) => {
                let ty = rvalue.ty(self.mir.locals()).unwrap();
                let layout = TyAndLayout::expect_from_ty(ty);

                // `rvalue_creates_operand` has arranged that we only get here if
                // we can build the aggregate immediate from the field immediates.
                let mut inputs = ArrayVec::<Bx::Value, 2>::new();
                let mut input_scalars = ArrayVec::<abi::Scalar, 2>::new();

                for field_idx in layout.layout.fields.fields_by_offset_order() {
                    let op = self.codegen_operand(bx, &fields[field_idx]);
                    let values = op.val.immediates_or_place().left_or_else(|p| {
                        bug!("Field {field_idx:?} is {p:?} making {layout:?}");
                    });
                    let scalars = self.value_kind(&op.layout).scalars().unwrap();

                    assert_eq!(values.len(), scalars.len());

                    inputs.extend(values);
                    input_scalars.extend(scalars);
                }

                let output_scalars = self.value_kind(&layout).scalars().unwrap();

                itertools::izip!(&mut inputs, input_scalars, output_scalars).for_each(
                    |(v, in_s, out_s)| {
                        if in_s != out_s {
                            // We have to be really careful about bool here, because
                            // `(bool,)` stays i1 but `Cell<bool>` becomes i8.
                            *v = bx.from_immediate(*v);
                            *v = bx.to_immediate_scalar(*v, out_s);
                        }
                    },
                );

                let val = OperandValue::from_immediates(inputs);

                assert!(
                    val.is_expected_variant_for_type(self.cx, &layout),
                    "Made wrong variant {val:?} for type {layout:?}",
                );

                OperandRef { val, layout }
            }
            mir::Rvalue::ThreadLocalRef(..)
            | mir::Rvalue::ShallowInitBox(..)
            | mir::Rvalue::AddressOf(..)
            | mir::Rvalue::NullaryOp(..) => {
                bug!("not supported by RISL")
            }
        }
    }

    fn evaluate_array_len(&mut self, bx: &mut Bx, place: &mir::Place) -> Bx::Value {
        // ZST are passed as operands and require special handling
        // because codegen_place() panics if Local is operand.
        if place.projection.is_empty()
            && let LocalRef::Operand(op) = self.locals[place.local].clone()
            && let TyKind::RigidTy(RigidTy::Array(_, n)) = op.layout.ty.kind()
        {
            let n = n
                .eval_target_usize()
                .expect("expected monomorphic const in codegen");

            return bx.cx().const_usize(n);
        }

        // use common size calculation for non zero-sized types
        let cg_value = self.codegen_place(
            bx,
            mir::visit::PlaceRef {
                local: place.local,
                projection: &place.projection,
            },
        );

        cg_value.len(bx.cx())
    }

    /// Codegen an `Rvalue::RawPtr` or `Rvalue::Ref`
    fn codegen_place_to_pointer(
        &mut self,
        bx: &mut Bx,
        place: &mir::Place,
        mk_ptr_ty: impl FnOnce(Ty) -> Ty,
    ) -> OperandRef<Bx::Value> {
        let cg_place = self.codegen_place(
            bx,
            mir::visit::PlaceRef {
                local: place.local,
                projection: &place.projection,
            },
        );

        let val = cg_place.val.address();
        let ty = cg_place.layout.ty;
        let ptr_ty = mk_ptr_ty(ty);

        OperandRef {
            val,
            layout: TyAndLayout::expect_from_ty(ptr_ty),
        }
    }

    fn codegen_scalar_binop(
        &mut self,
        bx: &mut Bx,
        op: mir::BinOp,
        lhs: Bx::Value,
        rhs: Bx::Value,
        input_ty: Ty,
    ) -> Bx::Value {
        let is_float = input_ty.kind().is_float();
        let is_signed = input_ty.kind().is_signed();

        match op {
            mir::BinOp::Add => {
                if is_float {
                    bx.fadd(lhs, rhs)
                } else {
                    bx.add(lhs, rhs)
                }
            }
            mir::BinOp::AddUnchecked => {
                if is_signed {
                    bx.unchecked_sadd(lhs, rhs)
                } else {
                    bx.unchecked_uadd(lhs, rhs)
                }
            }
            mir::BinOp::Sub => {
                if is_float {
                    bx.fsub(lhs, rhs)
                } else {
                    bx.sub(lhs, rhs)
                }
            }
            mir::BinOp::SubUnchecked => {
                if is_signed {
                    bx.unchecked_ssub(lhs, rhs)
                } else {
                    bx.unchecked_usub(lhs, rhs)
                }
            }
            mir::BinOp::Mul => {
                if is_float {
                    bx.fmul(lhs, rhs)
                } else {
                    bx.mul(lhs, rhs)
                }
            }
            mir::BinOp::MulUnchecked => {
                if is_signed {
                    bx.unchecked_smul(lhs, rhs)
                } else {
                    bx.unchecked_umul(lhs, rhs)
                }
            }
            mir::BinOp::Div => {
                if is_float {
                    bx.fdiv(lhs, rhs)
                } else if is_signed {
                    bx.sdiv(lhs, rhs)
                } else {
                    bx.udiv(lhs, rhs)
                }
            }
            mir::BinOp::Rem => {
                if is_float {
                    bx.frem(lhs, rhs)
                } else if is_signed {
                    bx.srem(lhs, rhs)
                } else {
                    bx.urem(lhs, rhs)
                }
            }
            mir::BinOp::BitOr => bx.or(lhs, rhs),
            mir::BinOp::BitAnd => bx.and(lhs, rhs),
            mir::BinOp::BitXor => bx.xor(lhs, rhs),
            mir::BinOp::Offset => {
                let pointee_type = input_ty
                    .kind()
                    .builtin_deref(true)
                    .unwrap_or_else(|| bug!("deref of non-pointer {:?}", input_ty))
                    .ty;
                let pointee_layout = TyAndLayout::expect_from_ty(pointee_type);

                if pointee_layout.layout.is_1zst() {
                    // `Offset` works in terms of the size of pointee,
                    // so offsetting a pointer to ZST is a noop.
                    lhs
                } else {
                    let llty = bx.cx().backend_type(&pointee_layout);

                    bx.ptr_element_ptr(llty, lhs, &[rhs])
                }
            }
            mir::BinOp::Shl | mir::BinOp::ShlUnchecked => {
                let rhs = build_shift_expr_rhs(bx, lhs, rhs, op == mir::BinOp::ShlUnchecked);

                bx.shl(lhs, rhs)
            }
            mir::BinOp::Shr | mir::BinOp::ShrUnchecked => {
                let rhs = build_shift_expr_rhs(bx, lhs, rhs, op == mir::BinOp::ShrUnchecked);

                if is_signed {
                    bx.ashr(lhs, rhs)
                } else {
                    bx.lshr(lhs, rhs)
                }
            }
            mir::BinOp::Ne
            | mir::BinOp::Lt
            | mir::BinOp::Gt
            | mir::BinOp::Eq
            | mir::BinOp::Le
            | mir::BinOp::Ge => {
                if is_float {
                    bx.fcmp(bin_op_to_fcmp_predicate(op), lhs, rhs)
                } else {
                    bx.icmp(bin_op_to_icmp_predicate(op, is_signed), lhs, rhs)
                }
            }
            mir::BinOp::Cmp => {
                use std::cmp::Ordering;

                assert!(!is_float);

                let pred = |op| bin_op_to_icmp_predicate(op, is_signed);

                let is_lt = bx.icmp(pred(mir::BinOp::Lt), lhs, rhs);
                let is_ne = bx.icmp(pred(mir::BinOp::Ne), lhs, rhs);
                let ge = bx.select(
                    is_ne,
                    bx.cx().const_i8(Ordering::Greater as i8),
                    bx.cx().const_i8(Ordering::Equal as i8),
                );

                bx.select(is_lt, bx.cx().const_i8(Ordering::Less as i8), ge)
            }
        }
    }

    fn codegen_wide_ptr_binop(
        &mut self,
        bx: &mut Bx,
        op: mir::BinOp,
        lhs_addr: Bx::Value,
        lhs_extra: Bx::Value,
        rhs_addr: Bx::Value,
        rhs_extra: Bx::Value,
        _input_ty: Ty,
    ) -> Bx::Value {
        match op {
            mir::BinOp::Eq => {
                let lhs = bx.icmp(IntPredicate::IntEQ, lhs_addr, rhs_addr);
                let rhs = bx.icmp(IntPredicate::IntEQ, lhs_extra, rhs_extra);

                bx.and(lhs, rhs)
            }
            mir::BinOp::Ne => {
                let lhs = bx.icmp(IntPredicate::IntNE, lhs_addr, rhs_addr);
                let rhs = bx.icmp(IntPredicate::IntNE, lhs_extra, rhs_extra);

                bx.or(lhs, rhs)
            }
            mir::BinOp::Le | mir::BinOp::Lt | mir::BinOp::Ge | mir::BinOp::Gt => {
                // a OP b ~ a.0 STRICT(OP) b.0 | (a.0 == b.0 && a.1 OP a.1)
                let (op, strict_op) = match op {
                    mir::BinOp::Lt => (IntPredicate::IntULT, IntPredicate::IntULT),
                    mir::BinOp::Le => (IntPredicate::IntULE, IntPredicate::IntULT),
                    mir::BinOp::Gt => (IntPredicate::IntUGT, IntPredicate::IntUGT),
                    mir::BinOp::Ge => (IntPredicate::IntUGE, IntPredicate::IntUGT),
                    _ => bug!(),
                };
                let lhs = bx.icmp(strict_op, lhs_addr, rhs_addr);
                let and_lhs = bx.icmp(IntPredicate::IntEQ, lhs_addr, rhs_addr);
                let and_rhs = bx.icmp(op, lhs_extra, rhs_extra);
                let rhs = bx.and(and_lhs, and_rhs);

                bx.or(lhs, rhs)
            }
            _ => {
                bug!("unexpected wide ptr binop");
            }
        }
    }

    pub(crate) fn rvalue_creates_operand(&self, rvalue: &Rvalue) -> bool {
        match rvalue {
            Rvalue::Cast(CastKind::Transmute, ..)
            | Rvalue::AddressOf(..)
            | Rvalue::ShallowInitBox(..)
            | Rvalue::ThreadLocalRef(_) => {
                bug!("not supported by risl")
            }
            Rvalue::Ref(..)
            | Rvalue::CopyForDeref(..)
            | Rvalue::Len(..)
            | Rvalue::Cast(..)
            | Rvalue::BinaryOp(..)
            | Rvalue::CheckedBinaryOp(..)
            | Rvalue::UnaryOp(..)
            | Rvalue::Discriminant(..)
            | Rvalue::NullaryOp(..)
            | Rvalue::Use(..) => true,
            // Arrays are always aggregates, so it's not worth checking anything here.
            // (If it's really `[(); N]` or `[T; 0]` and we use the place path, fine.)
            Rvalue::Repeat(..) => false,
            Rvalue::Aggregate(kind, _) => {
                let allowed_kind = match kind {
                    AggregateKind::Array(..) => false,
                    AggregateKind::Tuple => true,
                    AggregateKind::Adt(adt_def, ..) => adt_def.kind().is_struct(),
                    mir::AggregateKind::Closure(..) => true,
                    AggregateKind::RawPtr(..)
                    | AggregateKind::Coroutine(..)
                    | AggregateKind::CoroutineClosure(..) => {
                        bug!("not supported by RISL")
                    }
                };

                allowed_kind && {
                    let ty = rvalue
                        .ty(self.mir.locals())
                        .expect("should be able to resolve type during codegen");

                    !self.cx.is_backend_ref(&TyAndLayout::expect_from_ty(ty))
                }
            }
        }
    }

    /// Gets which variant of [`OperandValue`] is expected for a particular type.
    fn value_kind(&self, layout: &TyAndLayout) -> OperandValueKind {
        if layout.layout.is_1zst() {
            OperandValueKind::ZeroSized
        } else if self.cx.is_backend_immediate(layout) {
            assert!(!self.cx.is_backend_scalar_pair(layout));

            OperandValueKind::Immediate(match layout.layout.abi.clone() {
                ValueAbi::Scalar(s) => s,
                ValueAbi::Vector { element, .. } => element,
                x => bug!("Couldn't translate {x:?} as backend immediate"),
            })
        } else if self.cx.is_backend_scalar_pair(layout) {
            let ValueAbi::ScalarPair(s1, s2) = layout.layout.abi else {
                bug!(
                    "Couldn't translate {:?} as backend scalar pair",
                    layout.layout.abi,
                );
            };

            OperandValueKind::Pair(s1, s2)
        } else {
            OperandValueKind::Ref
        }
    }
}

/// The variants of this match [`OperandValue`], giving details about the
/// backend values that will be held in that other type.
#[derive(Debug, Copy, Clone)]
enum OperandValueKind {
    Ref,
    Immediate(abi::Scalar),
    Pair(abi::Scalar, abi::Scalar),
    ZeroSized,
}

impl OperandValueKind {
    fn scalars(self) -> Option<ArrayVec<abi::Scalar, 2>> {
        Some(match self {
            OperandValueKind::ZeroSized => ArrayVec::new(),
            OperandValueKind::Immediate(a) => ArrayVec::from_iter([a]),
            OperandValueKind::Pair(a, b) => [a, b].into(),
            OperandValueKind::Ref => return None,
        })
    }
}
