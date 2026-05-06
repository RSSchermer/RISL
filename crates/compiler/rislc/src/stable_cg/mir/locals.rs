//! Locals are in a private module as updating `LocalRef::Operand` has to
//! be careful wrt to subtyping. To deal with this we only allow updates by using
//! `FunctionCx::overwrite_local` which handles it automatically.

use std::ops::Index;

use bit_set::BitSet;
use rustc_middle::bug;
use rustc_public::abi::{FieldsShape, VariantsShape};
use rustc_public::mir;
use rustc_public::mir::visit::Location;
use rustc_public::mir::{
    BasicBlockIdx, MirVisitor, Rvalue, Statement, StatementKind, Terminator, TerminatorKind,
};
use tracing::debug;

use crate::stable_cg::mir::{FunctionCx, LocalRef};
use crate::stable_cg::traits::{BuilderMethods, CodegenMethods};
use crate::stable_cg::{OperandRef, OperandValue, TyAndLayout};

pub(super) struct Locals<V> {
    values: Vec<LocalRef<V>>,
}

impl<V> Index<mir::Local> for Locals<V> {
    type Output = LocalRef<V>;

    #[inline]
    fn index(&self, index: mir::Local) -> &LocalRef<V> {
        &self.values[index]
    }
}

impl<V> Locals<V> {
    pub(super) fn empty() -> Locals<V> {
        Locals { values: vec![] }
    }

    pub(super) fn indices(&self) -> impl DoubleEndedIterator<Item = mir::Local> + Clone + '_ {
        0..self.values.len()
    }
}

impl<'a, Bx: BuilderMethods<'a>> FunctionCx<'a, Bx> {
    pub(super) fn initialize_locals(&mut self, values: Vec<LocalRef<Bx::Value>>) {
        assert!(self.locals.values.is_empty());

        self.locals.values = values;
    }

    pub(super) fn reassign_local_operand(
        &mut self,
        bx: &mut Bx,
        local: mir::Local,
        mut new: OperandRef<Bx::Value>,
    ) {
        let LocalRef::Operand(old) = &self.locals[local] else {
            panic!("can only reassign an operand local")
        };

        match (old.val, &mut new.val) {
            (OperandValue::Ref(place), new_val) => {
                new_val.store(bx, &place.with_type(new.layout.clone()));
            }
            (OperandValue::Immediate(old), OperandValue::Immediate(new)) => {
                let local = bx.as_local(old);

                bx.assign(local, *new);

                *new = bx.local_value(local);
            }
            (OperandValue::Pair(a_old, b_old), OperandValue::Pair(a_new, b_new)) => {
                let a_local = bx.as_local(a_old);
                let b_local = bx.as_local(b_old);

                bx.assign(a_local, *a_new);
                bx.assign(b_local, *b_new);

                *a_new = bx.local_value(a_local);
                *b_new = bx.local_value(b_local);
            }
            (OperandValue::ZeroSized, OperandValue::ZeroSized) => {}
            _ => bug!(),
        }

        self.overwrite_local(local, LocalRef::Operand(new));
    }

    pub(super) fn overwrite_local(&mut self, local: mir::Local, mut value: LocalRef<Bx::Value>) {
        match value {
            LocalRef::Place(_) | LocalRef::UnsizedPlace(_) | LocalRef::PendingOperand => (),
            LocalRef::Operand(ref mut op) => {
                let local_ty = self.mir.locals()[local].ty;

                if local_ty != op.layout.ty {
                    // FIXME(#112651): This can be changed to an ICE afterwards.
                    debug!("updating type of operand due to subtyping");

                    op.layout.ty = local_ty;
                }
            }
        };

        self.locals.values[local] = value;
    }
}

pub(super) fn needs_alloca<'a, Bx: BuilderMethods<'a>>(
    fx: &FunctionCx<'a, Bx>,
    traversal_order: &[BasicBlockIdx],
) -> BitSet<usize> {
    let mut analyzer = NeedsInitAnalyzer {
        fx,
        needs_alloca: Default::default(),
    };

    for bb in traversal_order.iter().copied() {
        let data = &fx.mir.blocks[bb];

        analyzer.visit_basic_block(data);
    }

    analyzer.needs_alloca
}

fn ty_contains_enum(ty_layout: &TyAndLayout) -> bool {
    if ty_layout.layout.is_1zst() {
        return false;
    }

    match (&ty_layout.layout.fields, &ty_layout.layout.variants) {
        (FieldsShape::Primitive, _) | (FieldsShape::Union(_), _) => false,
        (FieldsShape::Array { count, .. }, _) => {
            *count > 0 && ty_contains_enum(&ty_layout.field(0))
        }
        (FieldsShape::Arbitrary { .. }, VariantsShape::Empty) => false,
        (FieldsShape::Arbitrary { .. }, VariantsShape::Multiple { .. }) => true,
        (FieldsShape::Arbitrary { .. }, VariantsShape::Single { .. }) => {
            (0..ty_layout.layout.fields.count())
                .any(|field_idx| ty_contains_enum(&ty_layout.field(field_idx)))
        }
    }
}

struct NeedsInitAnalyzer<'a, 'b, Bx: BuilderMethods<'b>> {
    fx: &'a FunctionCx<'b, Bx>,
    needs_alloca: BitSet<usize>,
}

impl<'a, 'b, Bx: BuilderMethods<'b>> NeedsInitAnalyzer<'a, 'b, Bx> {
    fn visit_assign_or_call_dest(&mut self, place: &mir::Place) {
        if place.projection.is_empty() {
            let ty = self.fx.mir.locals()[place.local].ty;
            let layout = TyAndLayout::expect_from_ty(ty);
            let is_immediate_or_scalar_pair = self.fx.cx.is_backend_immediate(&layout)
                || self.fx.cx.is_backend_scalar_pair(&layout);

            if !is_immediate_or_scalar_pair || ty_contains_enum(&layout) {
                self.needs_alloca.insert(place.local);
            }
        }
    }
}

impl<'a, 'b, Bx: BuilderMethods<'b>> MirVisitor for NeedsInitAnalyzer<'a, 'b, Bx> {
    fn visit_place(
        &mut self,
        place: &mir::Place,
        ptx: mir::visit::PlaceContext,
        location: Location,
    ) {
        if !place.projection.is_empty() {
            let is_indirect = matches!(place.projection.first(), Some(mir::ProjectionElem::Deref));

            if !is_indirect {
                if ptx.is_mutating() {
                    // If the projection is not empty, and the access is mutating, then the local
                    // should be allocated and interacted with via a pointer.
                    //
                    // This covers the case of single-field structs where the field has a scalar or
                    // scalar-pair ABI. In such cases, the struct itself is also assigned a scalar
                    // or scalar-pair ABI. In general, structs are always allocated and interacted
                    // with via a pointer, but in such "transparent" scalar or scalar-pair structs,
                    // allocations are not always necessary. However, the machinery for handling
                    // place projections is built around pointer projections. To avoid the need for
                    // special case handling of such transparent structs, we force an allocation
                    // anyway. SLIR's memory-to-value-flow transform will eliminate such allocations
                    // later.

                    self.needs_alloca.insert(place.local);
                } else {
                    // Ensure the only projections are those that
                    // `FunctionCx::maybe_codegen_consume_direct` can handle.

                    for elem in &place.projection {
                        if !matches!(elem, mir::ProjectionElem::Field(..)) {
                            self.needs_alloca.insert(place.local);

                            break;
                        }
                    }
                }
            }
        }

        self.super_place(place, ptx, location);
    }

    fn visit_statement(&mut self, stmt: &Statement, location: Location) {
        if let StatementKind::Assign(place, _) = &stmt.kind {
            self.visit_assign_or_call_dest(place);
        }

        self.super_statement(stmt, location)
    }

    fn visit_terminator(&mut self, term: &Terminator, location: Location) {
        if let TerminatorKind::Call { destination, .. } = &term.kind {
            self.visit_assign_or_call_dest(destination);
        }

        self.super_terminator(term, location)
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue, location: Location) {
        match rvalue {
            Rvalue::Ref(_, _, place)
            | Rvalue::Discriminant(place)
            | Rvalue::AddressOf(_, place) => {
                let is_indirect =
                    matches!(place.projection.first(), Some(mir::ProjectionElem::Deref));

                if !is_indirect {
                    self.needs_alloca.insert(place.local);
                }
            }
            _ => {}
        }

        self.super_rvalue(rvalue, location)
    }
}
