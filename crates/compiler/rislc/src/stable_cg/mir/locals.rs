//! Locals are in a private module as updating `LocalRef::Operand` has to
//! be careful wrt to subtyping. To deal with this we only allow updates by using
//! `FunctionCx::overwrite_local` which handles it automatically.

use std::ops::Index;

use bit_set::BitSet;
use bit_vec::BitVec;
use rustc_public::mir;
use rustc_public::mir::visit::Location;
use rustc_public::mir::{BasicBlockIdx, MirVisitor, Rvalue, Statement, StatementKind};
use tracing::debug;

use crate::stable_cg::mir::{FunctionCx, LocalRef};
use crate::stable_cg::traits::BuilderMethods;

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
    let local_count = fx.mir.locals().len();
    let visited = BitSet::from_bit_vec(BitVec::from_elem(local_count, false));
    let needs_init = BitSet::from_bit_vec(BitVec::from_elem(local_count, false));

    let mut analyzer = NeedsInitAnalyzer {
        fx,
        visited: Default::default(),
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
    visited: BitSet<usize>,
    needs_alloca: BitSet<usize>,
}

impl<'a, 'b, Bx: BuilderMethods<'b>> MirVisitor for NeedsInitAnalyzer<'a, 'b, Bx> {
    fn visit_statement(&mut self, stmt: &Statement, location: Location) {
        if let StatementKind::Assign(place, rvalue) = &stmt.kind {
            let is_local = place.projection.is_empty();

            if is_local {
                if self.visited.insert(place.local) {
                    if !self.fx.rvalue_creates_operand(rvalue) {
                        self.needs_alloca.insert(place.local);
                    }
                }
            }

            self.super_statement(stmt, location)
        }
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue, location: Location) {
        match rvalue {
            Rvalue::Ref(_, _, place) | Rvalue::Discriminant(place) => {
                let is_local = place.projection.is_empty();

                if is_local {
                    self.needs_alloca.insert(place.local);
                }
            }
            _ => {}
        }

        self.super_rvalue(rvalue, location)
    }
}
