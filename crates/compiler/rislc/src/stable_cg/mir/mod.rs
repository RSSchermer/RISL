use core::fmt;
use std::{io, iter};

use bit_set::BitSet;
use bit_vec::BitVec;
use rustc_middle::bug;
use rustc_public::abi::{FnAbi, PassMode};
use rustc_public::mir::mono::Instance;
use rustc_public::mir::{BasicBlockIdx, Body, Local, LocalDecl};
use rustc_public::ty::{RigidTy, TyKind};
use smallvec::SmallVec;
use tracing::debug;

use crate::stable_cg::layout::TyAndLayout;
use crate::stable_cg::mir::traversal::reachable_reverse_postorder;
use crate::stable_cg::traits::*;

// mod analyze;
mod block;
// mod intrinsic;
mod locals;
pub mod operand;
pub mod place;
mod rvalue;
mod statement;
mod traversal;

use self::operand::{OperandRef, OperandValue};
use self::place::PlaceRef;

const START_BLOCK: usize = 0;
const RETURN_PLACE: usize = 0;

// Used for tracking the state of generated basic blocks.
enum CachedLlbb<T> {
    /// Nothing created yet.
    None,

    /// Has been created.
    Some(T),

    /// Nothing created yet, and nothing should be.
    Skip,
}

/// Master context for codegenning from MIR.
pub struct FunctionCx<'a, Bx: BuilderMethods<'a>> {
    instance: Instance,

    mir: Body,

    llfn: Bx::Function,

    cx: &'a Bx::CodegenCx,

    fn_abi: FnAbi,

    /// A backend `BasicBlock` for each MIR `BasicBlock`, created lazily
    /// as-needed (e.g. RPO reaching it or another block branching to it).
    // FIXME(eddyb) rename `llbbs` and other `ll`-prefixed things to use a
    // more backend-agnostic prefix such as `cg` (i.e. this would be `cgbbs`).
    cached_llbbs: Vec<CachedLlbb<Bx::BasicBlock>>,

    /// Cached basic-block predecessors
    bb_predecessors: Option<Vec<SmallVec<[BasicBlockIdx; 2]>>>,

    /// Cached unreachable block
    unreachable_block: Option<Bx::BasicBlock>,

    /// The location where each MIR arg/var/tmp/ret is stored. This is
    /// usually an `PlaceRef` representing an alloca, but not always:
    /// sometimes we can skip the alloca and just store the value
    /// directly using an `OperandRef`, which makes for tighter LLVM
    /// IR. The conditions for using an `OperandRef` are as follows:
    ///
    /// - the type of the local must be judged "immediate" by `is_llvm_immediate`
    /// - the operand must never be referenced indirectly
    ///     - we should not take its address using the `&` operator
    ///     - nor should it appear in a place path like `tmp.a`
    /// - the operand must be defined by an rvalue that can generate immediate
    ///   values
    ///
    /// Avoiding allocs can also be important for certain intrinsics,
    /// notably `expect`.
    locals: locals::Locals<Bx::Value>,
}

#[derive(Clone)]
enum LocalRef<V> {
    Place(PlaceRef<V>),
    /// `UnsizedPlace(p)`: `p` itself is a thin pointer (indirect place).
    /// `*p` is the wide pointer that references the actual unsized place.
    /// Every time it is initialized, we have to reallocate the place
    /// and update the wide pointer. That's the reason why it is indirect.
    UnsizedPlace(PlaceRef<V>),
    /// The backend [`OperandValue`] has already been generated.
    Operand(OperandRef<V>),
    /// Will be a `Self::Operand` once we get to its definition.
    PendingOperand,
}

impl<'tcx, V: CodegenObject> LocalRef<V> {
    fn new_operand(layout: TyAndLayout) -> LocalRef<V> {
        if layout.layout.is_1zst() {
            // Zero-size temporaries aren't always initialized, which
            // doesn't matter because they don't contain data, but
            // we need something sufficiently aligned in the operand.
            LocalRef::Operand(OperandRef::zero_sized(layout))
        } else {
            LocalRef::PendingOperand
        }
    }
}

impl<V: CodegenObject> fmt::Debug for LocalRef<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LocalRef::Place(p) => write!(f, "Place({:?})", p),
            LocalRef::UnsizedPlace(p) => write!(f, "UnsizedPlace({:?})", p),
            LocalRef::Operand(o) => write!(f, "Operand({:?})", o),
            LocalRef::PendingOperand => write!(f, "PendingOperand"),
        }
    }
}

///////////////////////////////////////////////////////////////////////////

pub fn codegen_mir<'a, Bx: BuilderMethods<'a>>(cx: &'a Bx::CodegenCx, instance: Instance) {
    let llfn = cx.get_fn(&instance);

    let mir = if let Some(body) = instance.body() {
        body
    } else {
        return;
    };

    // mir.dump(&mut io::stdout(), instance.name().as_str());

    let Ok(fn_abi) = instance.fn_abi() else {
        bug!("fn ABI should be available during codegen")
    };

    debug!("fn_abi: {:?}", fn_abi);

    let start_llbb = Bx::start_block(cx, llfn);
    let mut start_bx = Bx::build(cx, start_llbb);

    let cached_llbbs: Vec<CachedLlbb<Bx::BasicBlock>> = (0..mir.blocks.len())
        .map(|i| {
            if i == START_BLOCK {
                CachedLlbb::Some(start_llbb)
            } else {
                CachedLlbb::None
            }
        })
        .collect();

    let traversal_order = reachable_reverse_postorder(&mir);

    let mut fx = FunctionCx {
        instance,
        mir,
        llfn,
        fn_abi,
        cx,
        cached_llbbs,
        bb_predecessors: None,
        unreachable_block: None,
        locals: locals::Locals::empty(),
    };

    let needs_alloca = locals::needs_alloca(&fx, &traversal_order);

    // It may seem like we should iterate over `required_consts` to ensure they all successfully
    // evaluate; however, the `MirUsedCollector` already did that during the collection phase of
    // monomorphization, and if there is an error during collection then codegen never starts -- so
    // we don't have to do it again.

    // Allocate variable and temp allocas
    let local_values = {
        let args = arg_local_refs(&mut start_bx, &mut fx, &needs_alloca);

        let mut allocate_local = |local_index| {
            let decl: &LocalDecl = &fx.mir.locals()[local_index];

            if local_index == 0
            /* return place */
            {
                if let PassMode::Indirect { .. } = fx.fn_abi.ret.mode {
                    let llretptr = start_bx.get_param(0);

                    return LocalRef::Place(PlaceRef::new_sized(
                        llretptr,
                        TyAndLayout::expect_from_ty(decl.ty),
                    ));
                }
            }

            let layout = TyAndLayout::expect_from_ty(decl.ty);

            if needs_alloca.contains(local_index) {
                if layout.layout.is_unsized() {
                    LocalRef::UnsizedPlace(PlaceRef::alloca_unsized_indirect(&mut start_bx, layout))
                } else {
                    LocalRef::Place(PlaceRef::alloca(&mut start_bx, layout))
                }
            } else {
                LocalRef::new_operand(layout)
            }
        };

        let retptr = allocate_local(RETURN_PLACE);

        iter::once(retptr)
            .chain(args.into_iter())
            .chain(inner_local_indices(&fx.mir).map(allocate_local))
            .collect()
    };

    fx.initialize_locals(local_values);

    // The builders will be created separately for each basic block at `codegen_block`.
    // So drop the builder of `start_llbb` to avoid having two at the same time.
    drop(start_bx);

    let mut unreached_blocks = BitSet::from_bit_vec(BitVec::from_elem(fx.mir.blocks.len(), true));

    // Codegen the body of each reachable block using our reverse postorder list.
    for bb in traversal_order {
        fx.codegen_block(bb);
        unreached_blocks.remove(bb);
    }

    // FIXME: These empty unreachable blocks are *mostly* a waste. They are occasionally
    // targets for a SwitchInt terminator, but the reimplementation of the mono-reachable
    // simplification in SwitchInt lowering sometimes misses cases that
    // mono_reachable_reverse_postorder manages to figure out.
    // The solution is to do something like post-mono GVN. But for now we have this hack.
    for bb in unreached_blocks.iter() {
        fx.codegen_block_as_unreachable(bb);
    }
}

/// Returns an iterator over the indices that identify the "inner locals" (not the return local and
/// not the argument locals) of a MIR body.
fn inner_local_indices(body: &Body) -> impl DoubleEndedIterator<Item = Local> {
    (body.arg_locals().len() + 1)..body.locals().len()
}

/// Produces, for each argument, a `Value` pointing at the argument's value.
fn arg_local_refs<'a, Bx: BuilderMethods<'a>>(
    bx: &mut Bx,
    fx: &mut FunctionCx<'a, Bx>,
    needs_alloca: &BitSet<usize>,
) -> Vec<LocalRef<Bx::Value>> {
    let mir = &fx.mir;

    let mut idx = 0;
    let mut llarg_idx = matches!(fx.fn_abi.ret.mode, PassMode::Indirect { .. }) as usize;
    let mut num_untupled = None;

    let args = mir
        .arg_locals()
        .iter()
        .enumerate()
        .map(|(i, arg_decl)| {
            // Arg locals come after the return local
            let local = i + 1;

            let arg_ty = arg_decl.ty;

            if Some(local) == mir.spread_arg() {
                // This argument (e.g., the last argument in the "rust-call" ABI)
                // is a tuple that was spread at the ABI level and now we have
                // to reconstruct it into a tuple local variable, from multiple
                // individual LLVM function arguments.
                let TyKind::RigidTy(RigidTy::Tuple(tupled_arg_tys)) = arg_ty.kind() else {
                    bug!("spread argument should be a tuple");
                };

                let layout = arg_ty
                    .layout()
                    .expect("must have layout during codegen")
                    .shape();

                // FIXME: support unsized params in "rust-call" ABI
                if layout.is_unsized() {
                    bug!("\"rust-call\" ABI does not support unsized params",);
                }

                let place = PlaceRef::alloca(bx, TyAndLayout { ty: arg_ty, layout });

                for i in 0..tupled_arg_tys.len() {
                    let arg = &fx.fn_abi.args[idx];

                    idx += 1;

                    let pr_field = place.project_field(bx, i);

                    bx.store_fn_arg(arg, &mut llarg_idx, &pr_field);
                }

                assert_eq!(
                    None,
                    num_untupled.replace(tupled_arg_tys.len()),
                    "Replaced existing num_tupled"
                );

                return LocalRef::Place(place);
            }

            if fx.fn_abi.c_variadic {
                bug!("RISL does not support variadic functions")
            }

            let arg = &fx.fn_abi.args[idx];

            idx += 1;

            if !needs_alloca.contains(local) {
                let local = |op| LocalRef::Operand(op);

                match arg.mode {
                    PassMode::Ignore => {
                        return local(OperandRef::zero_sized(TyAndLayout {
                            ty: arg_ty,
                            layout: arg.layout.shape(),
                        }));
                    }
                    PassMode::Direct(_) => {
                        let llarg = bx.get_param(llarg_idx);

                        llarg_idx += 1;

                        return local(OperandRef::from_immediate_or_packed_pair(
                            bx,
                            llarg,
                            TyAndLayout {
                                ty: arg_ty,
                                layout: arg.layout.shape(),
                            },
                        ));
                    }
                    PassMode::Pair(..) => {
                        let (a, b) = (bx.get_param(llarg_idx), bx.get_param(llarg_idx + 1));

                        llarg_idx += 2;

                        return local(OperandRef {
                            val: OperandValue::Pair(a, b),
                            layout: TyAndLayout {
                                ty: arg_ty,
                                layout: arg.layout.shape(),
                            },
                        });
                    }
                    _ => {}
                }
            }

            match arg.mode {
                // Unsized indirect qrguments
                PassMode::Indirect { .. } if arg.layout.shape().is_unsized() => {
                    // As the storage for the indirect argument lives during
                    // the whole function call, we just copy the wide pointer.
                    let llarg = bx.get_param(llarg_idx);

                    llarg_idx += 1;

                    let llextra = bx.get_param(llarg_idx);

                    llarg_idx += 1;

                    let indirect_operand = OperandValue::Pair(llarg, llextra);
                    let tmp = PlaceRef::alloca_unsized_indirect(
                        bx,
                        TyAndLayout {
                            ty: arg.ty,
                            layout: arg.layout.shape(),
                        },
                    );

                    indirect_operand.store(bx, &tmp);

                    LocalRef::UnsizedPlace(tmp)
                }
                // Sized indirect arguments
                PassMode::Indirect { .. } => {
                    let llarg = bx.get_param(llarg_idx);

                    llarg_idx += 1;

                    LocalRef::Place(PlaceRef::new_sized(
                        llarg,
                        TyAndLayout {
                            ty: arg.ty,
                            layout: arg.layout.shape(),
                        },
                    ))
                }

                _ => {
                    let tmp = PlaceRef::alloca(
                        bx,
                        TyAndLayout {
                            ty: arg.ty,
                            layout: arg.layout.shape(),
                        },
                    );

                    bx.store_fn_arg(arg, &mut llarg_idx, &tmp);

                    LocalRef::Place(tmp)
                }
            }
        })
        .collect::<Vec<_>>();

    args
}
