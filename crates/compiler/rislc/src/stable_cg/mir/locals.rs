//! Locals are in a private module as updating `LocalRef::Operand` has to
//! be careful wrt to subtyping. To deal with this we only allow updates by using
//! `FunctionCx::overwrite_local` which handles it automatically.

use std::ops::Index;

use bit_set::BitSet;
use rustc_public::mir;
use rustc_public::mir::visit::Location;
use rustc_public::mir::{
    BasicBlockIdx, MirVisitor, Rvalue, Statement, StatementKind, Terminator, TerminatorKind,
};
use tracing::debug;

use crate::stable_cg::TyAndLayout;
use crate::stable_cg::mir::{FunctionCx, LocalRef};
use crate::stable_cg::traits::{BuilderMethods, CodegenMethods};

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

struct NeedsInitAnalyzer<'a, 'b, Bx: BuilderMethods<'b>> {
    fx: &'a FunctionCx<'b, Bx>,
    needs_alloca: BitSet<usize>,
}

impl<'a, 'b, Bx: BuilderMethods<'b>> NeedsInitAnalyzer<'a, 'b, Bx> {
    fn visit_assign_or_call_dest(&mut self, place: &mir::Place) {
        if place.projection.is_empty() {
            let ty = self.fx.mir.locals()[place.local].ty;
            let layout = TyAndLayout::expect_from_ty(ty);

            if !self.fx.cx.is_backend_immediate(&layout)
                && !self.fx.cx.is_backend_scalar_pair(&layout)
            {
                println!("needs alloca: {:?}", layout);

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
