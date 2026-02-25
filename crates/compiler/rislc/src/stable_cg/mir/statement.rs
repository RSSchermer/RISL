use rustc_middle::bug;
use rustc_public::mir;
use rustc_public::mir::NonDivergingIntrinsic;
use tracing::instrument;

use super::{FunctionCx, LocalRef};
use crate::stable_cg::OperandValue;
use crate::stable_cg::traits::*;

impl<'a, Bx: BuilderMethods<'a>> FunctionCx<'a, Bx> {
    #[instrument(level = "debug", skip(self, bx))]
    pub(crate) fn codegen_statement(&mut self, bx: &mut Bx, statement: &mir::Statement) {
        match &statement.kind {
            mir::StatementKind::Assign(place, rvalue) => {
                if place.projection.is_empty() {
                    match self.locals[place.local].clone() {
                        LocalRef::Place(cg_dest) => self.codegen_rvalue(bx, &cg_dest, rvalue),
                        LocalRef::UnsizedPlace(cg_indirect_dest) => bug!("not supported by RISL"),
                        LocalRef::PendingOperand => {
                            let mut operand = self.codegen_rvalue_operand(bx, rvalue);

                            match &mut operand.val {
                                OperandValue::Immediate(val) => {
                                    let local = bx.as_local(*val);

                                    *val = bx.local_value(local);
                                }
                                OperandValue::Pair(a, b) => {
                                    let a_local = bx.as_local(*a);
                                    let b_local = bx.as_local(*b);

                                    *a = bx.local_value(a_local);
                                    *b = bx.local_value(b_local);
                                }
                                _ => {}
                            }

                            self.overwrite_local(place.local, LocalRef::Operand(operand));
                        }
                        LocalRef::Operand(old) => {
                            let mut new = self.codegen_rvalue_operand(bx, rvalue);

                            match (old.val, &mut new.val) {
                                (OperandValue::Ref(place), new_val) => {
                                    new_val.store(bx, &place.with_type(new.layout.clone()));
                                }
                                (OperandValue::Immediate(old), OperandValue::Immediate(new)) => {
                                    let local = bx.as_local(old);

                                    bx.assign(local, *new);

                                    *new = bx.local_value(local);
                                }
                                (
                                    OperandValue::Pair(a_old, b_old),
                                    OperandValue::Pair(a_new, b_new),
                                ) => {
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

                            self.overwrite_local(place.local, LocalRef::Operand(new));
                        }
                    }
                } else {
                    let cg_dest = self.codegen_place(
                        bx,
                        mir::visit::PlaceRef {
                            local: place.local,
                            projection: &place.projection,
                        },
                    );

                    self.codegen_rvalue(bx, &cg_dest, rvalue);
                }
            }
            mir::StatementKind::SetDiscriminant {
                place,
                variant_index,
            } => {
                self.codegen_place(
                    bx,
                    mir::visit::PlaceRef {
                        local: place.local,
                        projection: &place.projection,
                    },
                )
                .codegen_set_discr(bx, *variant_index);
            }
            mir::StatementKind::StorageLive(local) => {
                if let LocalRef::Place(cg_place) = &self.locals[*local] {
                    cg_place.storage_live(bx);
                } else if let LocalRef::UnsizedPlace(cg_indirect_place) = &self.locals[*local] {
                    cg_indirect_place.storage_live(bx);
                }
            }
            mir::StatementKind::StorageDead(local) => {
                if let LocalRef::Place(cg_place) = &self.locals[*local] {
                    cg_place.storage_dead(bx);
                } else if let LocalRef::UnsizedPlace(cg_indirect_place) = &self.locals[*local] {
                    cg_indirect_place.storage_dead(bx);
                }
            }
            mir::StatementKind::Intrinsic(NonDivergingIntrinsic::Assume(op)) => {
                todo!()
                // let op_val = self.codegen_operand(bx, op);
                // bx.assume(op_val.immediate());
            }
            mir::StatementKind::Intrinsic(NonDivergingIntrinsic::CopyNonOverlapping(
                mir::CopyNonOverlapping { count, src, dst },
            )) => {
                todo!()
                // let dst_val = self.codegen_operand(bx, dst);
                // let src_val = self.codegen_operand(bx, src);
                // let count = self.codegen_operand(bx, count).immediate();
                // let pointee_layout = dst_val
                //     .layout
                //     .pointee_info_at(bx, rustc_abi::Size::ZERO)
                //     .expect("Expected pointer");
                // let bytes = bx.mul(count, bx.const_usize(pointee_layout.size.bytes()));
                //
                // let align = pointee_layout.align;
                // let dst = dst_val.immediate();
                // let src = src_val.immediate();
                // bx.memcpy(dst, align, src, align, bytes, crate::MemFlags::empty());
            }
            mir::StatementKind::Coverage { .. }
            | mir::StatementKind::FakeRead(..)
            | mir::StatementKind::Retag { .. }
            | mir::StatementKind::AscribeUserType { .. }
            | mir::StatementKind::ConstEvalCounter
            | mir::StatementKind::PlaceMention(..)
            | mir::StatementKind::Nop => {}
        }
    }
}
