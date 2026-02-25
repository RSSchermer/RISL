use rustc_middle::bug;
use rustc_public::abi::{ArgAbi, FnAbi, PassMode, ValueAbi};
use rustc_public::mir;
use rustc_public::mir::mono::{Instance, InstanceKind};
use rustc_public::mir::{BasicBlockIdx, SwitchTargets, TerminatorKind};
use rustc_public::ty::{Abi, IntrinsicDef, RigidTy, Ty, TyKind};
use smallvec::SmallVec;
use tracing::debug;

use super::operand::OperandRef;
use super::operand::OperandValue::{Immediate, Pair, Ref, ZeroSized};
use super::place::{PlaceRef, PlaceValue};
use super::{CachedLlbb, FunctionCx, LocalRef};
use crate::stable_cg::common::IntPredicate;
use crate::stable_cg::layout::TyAndLayout;
use crate::stable_cg::traits::*;

const RETURN_PLACE_REF: mir::visit::PlaceRef = mir::visit::PlaceRef {
    local: 0,
    projection: &[],
};

// Indicates if we are in the middle of merging a BB's successor into it. This
// can happen when BB jumps directly to its successor and the successor has no
// other predecessors.
#[derive(Debug, PartialEq)]
enum MergingSucc {
    False,
    True,
}

/// Used by `FunctionCx::codegen_terminator` for emitting common patterns
/// e.g., creating a basic block, calling a function, etc.
struct TerminatorCodegenHelper<'a> {
    bb: BasicBlockIdx,
    terminator: &'a mir::Terminator,
}

fn funclet_br<'a, Bx: BuilderMethods<'a>>(
    fx: &mut FunctionCx<'a, Bx>,
    bx: &mut Bx,
    target: mir::BasicBlockIdx,
    mergeable_succ: bool,
) -> MergingSucc {
    if mergeable_succ {
        // We can merge the successor into this bb, so no need for a `br`.
        MergingSucc::True
    } else {
        let lltarget = fx.llbb(target);

        bx.br(lltarget);

        MergingSucc::False
    }
}

/// Call `fn_ptr` of `fn_abi` with the arguments `llargs`, the optional
/// return destination `destination` and the unwind action `unwind`.
fn do_call<'a, Bx: BuilderMethods<'a>>(
    fx: &mut FunctionCx<'a, Bx>,
    bx: &mut Bx,
    fn_abi: &FnAbi,
    fn_ptr: Bx::Value,
    llargs: &[Bx::Value],
    destination: Option<(ReturnDest<Bx::Value>, mir::BasicBlockIdx)>,
    copied_constant_arguments: &[PlaceRef<<Bx as BackendTypes>::Value>],
    instance: Option<&Instance>,
    mergeable_succ: bool,
) -> MergingSucc {
    // If there is a cleanup block and the function we're calling can unwind, then
    // do an invoke, otherwise do a call.
    let fn_ty = bx.fn_decl_backend_type(fn_abi);

    let llret = bx.call(fn_ty, Some(fn_abi), fn_ptr, llargs, instance);

    if let Some((ret_dest, target)) = destination {
        for tmp in copied_constant_arguments {
            bx.lifetime_end(tmp.val.llval, tmp.layout.layout.size);
        }

        fx.store_return(bx, ret_dest, &fn_abi.ret, llret);

        funclet_br(fx, bx, target, mergeable_succ)
    } else {
        bx.unreachable();

        MergingSucc::False
    }
}

/// Codegen implementations for some terminator variants.
impl<'a, Bx: BuilderMethods<'a>> FunctionCx<'a, Bx> {
    fn codegen_switchint_terminator(
        &mut self,
        bx: &mut Bx,
        discr: &mir::Operand,
        targets: &SwitchTargets,
    ) {
        let discr = self.codegen_operand(bx, discr);
        let switch_ty = discr.layout.ty;
        let discr_value = discr.immediate();

        // If our discriminant is a constant we can branch directly
        if let Some(const_discr) = bx.const_to_opt_u128(discr_value, false) {
            let (_, target) = targets
                .branches()
                .find(|(test, _)| *test == const_discr)
                .expect("constant discriminant should have matching branch");

            bx.br(self.llbb(target));

            return;
        };

        if targets.len() == 2 {
            // If there are two targets (one conditional, one fallback), emit `br` instead of
            // `switch`.
            let (test_value, target) = targets.branches().next().unwrap();
            let otherwise = targets.otherwise();

            let lltarget = self.llbb(target);
            let llotherwise = self.llbb(otherwise);

            if switch_ty == Ty::bool_ty() {
                // Don't generate trivial icmps when switching on bool.
                match test_value {
                    0 => {
                        bx.cond_br(discr_value, llotherwise, lltarget);
                    }
                    1 => {
                        bx.cond_br(discr_value, lltarget, llotherwise);
                    }
                    _ => bug!(),
                }
            } else {
                let switch_llty =
                    bx.immediate_backend_type(&TyAndLayout::expect_from_ty(switch_ty));
                let llval = bx.const_uint_big(switch_llty, test_value);
                let cmp = bx.icmp(IntPredicate::IntEQ, discr_value, llval);

                bx.cond_br(cmp, lltarget, llotherwise);
            }
        } else {
            bx.switch(
                discr_value,
                self.llbb(targets.otherwise()),
                targets
                    .branches()
                    .map(|(value, target)| (value, self.llbb(target))),
            );
        }
    }

    fn codegen_return_terminator(&mut self, bx: &mut Bx) {
        let llval = match &self.fn_abi.ret.mode {
            PassMode::Ignore | PassMode::Indirect { .. } => {
                bx.ret_void();

                return;
            }

            PassMode::Direct(_) | PassMode::Pair(..) => {
                let op = self.codegen_consume(bx, RETURN_PLACE_REF);

                if let Ref(place_val) = op.val {
                    bx.load_from_place(bx.backend_type(&op.layout), place_val)
                } else {
                    op.immediate_or_packed_pair(bx)
                }
            }

            PassMode::Cast { .. } => bug!("not supported by RISL"),
        };

        bx.ret(llval);
    }

    fn codegen_drop_terminator(
        &mut self,
        bx: &mut Bx,
        location: &mir::Place,
        target: mir::BasicBlockIdx,
        mergeable_succ: bool,
    ) -> MergingSucc {
        let ty = location.ty(self.mir.locals()).unwrap();

        if matches!(ty.kind(), TyKind::RigidTy(RigidTy::Dynamic(..))) {
            bug!("RISL does not support dropping dyn types")
        }

        let drop_fn = Instance::resolve_drop_in_place(ty);

        if drop_fn.is_empty_shim() {
            // we don't actually need to drop anything.
            return funclet_br(self, bx, target, mergeable_succ);
        }

        let fn_abi = drop_fn.fn_abi().unwrap();
        let place = self.codegen_place(
            bx,
            mir::visit::PlaceRef {
                local: location.local,
                projection: &location.projection,
            },
        );
        let (args1, args2);

        let args = if let Some(llextra) = place.val.llextra {
            args2 = [place.val.llval, llextra];
            &args2[..]
        } else {
            args1 = [place.val.llval];
            &args1[..]
        };

        do_call(
            self,
            bx,
            &fn_abi,
            bx.get_fn_addr(&drop_fn),
            args,
            Some((ReturnDest::Nothing, target)),
            &[],
            Some(&drop_fn),
            mergeable_succ,
        )
    }

    fn codegen_call_terminator(
        &mut self,
        bx: &mut Bx,
        terminator: &mir::Terminator,
        func: &mir::Operand,
        args: &[mir::Operand],
        destination: &mir::Place,
        target: Option<mir::BasicBlockIdx>,
        mergeable_succ: bool,
    ) -> MergingSucc {
        let span = terminator.span;

        // Create the callee. This is a fn ptr or zero-sized and hence a kind of scalar.
        let callee = self.codegen_operand(bx, func);

        let TyKind::RigidTy(ty_kind) = callee.layout.ty.kind() else {
            bug!("callee type should be rigid")
        };

        let instance = match ty_kind {
            RigidTy::FnDef(def_id, args) => {
                Instance::resolve(def_id, &args).expect("instance should resolve during codegen")
            }
            RigidTy::FnPtr(..) => bug!("function pointers are not supported by RISL"),
            _ => bug!("{} is not callable", callee.layout.ty),
        };

        if instance.is_empty_shim() {
            // Empty drop glue; a no-op.
            let target = target.unwrap();

            return funclet_br(self, bx, target, mergeable_succ);
        }

        let fn_abi = instance.fn_abi().unwrap();
        let is_indirect_ret = matches!(fn_abi.ret.mode, PassMode::Indirect { .. });

        // The arguments we'll be passing. Plus one to account for outptr, if used.
        let arg_count = fn_abi.args.len() + is_indirect_ret as usize;

        let instance = if instance.kind == InstanceKind::Intrinsic {
            todo!()
        } else {
            instance
        };

        let mut llargs = Vec::with_capacity(arg_count);
        let destination = target.as_ref().map(|&target| {
            (
                self.make_return_dest(
                    bx,
                    destination,
                    &fn_abi.ret,
                    &mut llargs,
                    None,
                    Some(target),
                ),
                target,
            )
        });

        let abi = callee
            .layout
            .ty
            .kind()
            .fn_sig()
            .expect("callee should be a function type")
            .skip_binder()
            .abi;
        let is_rust_call = abi == Abi::RustCall;

        // Split the rust-call tupled arguments off.
        let (first_args, untuple) = if is_rust_call && !args.is_empty() {
            let (tup, args) = args.split_last().unwrap();

            (args, Some(tup))
        } else {
            (args, None)
        };

        let mut copied_constant_arguments = vec![];

        for (i, arg) in first_args.iter().enumerate() {
            let mut op = self.codegen_operand(bx, arg);

            // The callee needs to own the argument memory if we pass it
            // by-ref, so make a local copy of non-immediate constants.
            match (arg, op.val) {
                (
                    &mir::Operand::Copy(_) | &mir::Operand::Constant(_),
                    Ref(PlaceValue { llextra: None, .. }),
                ) => {
                    let tmp = PlaceRef::alloca(bx, op.layout.clone());

                    bx.lifetime_start(tmp.val.llval, tmp.layout.layout.size);
                    op.val.store(bx, &tmp);
                    op.val = Ref(tmp.val);
                    copied_constant_arguments.push(tmp);
                }
                _ => {}
            }

            self.codegen_argument(bx, op, &mut llargs, &fn_abi.args[i]);
        }

        if let Some(tup) = untuple {
            self.codegen_arguments_untupled(bx, tup, &mut llargs, &fn_abi.args[first_args.len()..]);
        }

        do_call(
            self,
            bx,
            &fn_abi,
            bx.get_fn_addr(&instance),
            &llargs,
            destination,
            &copied_constant_arguments,
            Some(&instance),
            mergeable_succ,
        )
    }

    pub(crate) fn codegen_block(&mut self, mut bb: mir::BasicBlockIdx) {
        let llbb = match self.try_llbb(bb) {
            Some(llbb) => llbb,
            None => return,
        };
        let bx = &mut Bx::build(self.cx, llbb);

        // MIR basic blocks stop at any function call. This may not be the case
        // for the backend's basic blocks, in which case we might be able to
        // combine multiple MIR basic blocks into a single backend basic block.
        loop {
            // TODO: this is not a cheap clone, but eliminating it requires some fairly extensive
            // restructuring of the FunctionCx
            let data = self.mir.blocks[bb].clone();

            debug!("codegen_block({:?})", bb);

            for statement in &data.statements {
                self.codegen_statement(bx, statement);
            }

            let merging_succ = self.codegen_terminator(bx, bb, &data.terminator);

            if merging_succ == MergingSucc::False {
                break;
            }

            // We are merging the successor into the produced backend basic
            // block. Record that the successor should be skipped when it is
            // reached.
            //
            // Note: we must not have already generated code for the successor.
            // This is implicitly ensured by the reverse postorder traversal,
            // and the assertion explicitly guarantees that.
            let succ = data.terminator.successors()[0];
            assert!(matches!(self.cached_llbbs[succ], CachedLlbb::None));
            self.cached_llbbs[succ] = CachedLlbb::Skip;
            bb = succ;
        }
    }

    pub(crate) fn codegen_block_as_unreachable(&mut self, bb: mir::BasicBlockIdx) {
        let llbb = match self.try_llbb(bb) {
            Some(llbb) => llbb,
            None => return,
        };
        let bx = &mut Bx::build(self.cx, llbb);

        debug!("codegen_block_as_unreachable({:?})", bb);

        bx.unreachable();
    }

    fn codegen_terminator(
        &mut self,
        bx: &mut Bx,
        bb: BasicBlockIdx,
        terminator: &mir::Terminator,
    ) -> MergingSucc {
        debug!("codegen_terminator: {:?}", terminator);

        let mut mergeable_succ = || {
            // Note: any call to `switch_to_block` will invalidate a `true` value
            // of `mergeable_succ`.
            let successors = terminator.successors();

            if let Some(succ) = successors.first()
                && successors.len() == 1
                && let &[succ_pred] = self.predecessors()[*succ].as_slice()
            {
                // bb has a single successor, and bb is its only predecessor. This
                // makes it a candidate for merging.
                assert_eq!(succ_pred, bb);

                true
            } else {
                false
            }
        };

        match &terminator.kind {
            mir::TerminatorKind::Assert { .. }
            | mir::TerminatorKind::InlineAsm { .. }
            | TerminatorKind::Resume
            | TerminatorKind::Abort => bug!("not supported by RISL"),

            mir::TerminatorKind::Goto { target } => {
                let mergeable_succ = mergeable_succ();

                funclet_br(self, bx, *target, mergeable_succ)
            }

            mir::TerminatorKind::SwitchInt { discr, targets } => {
                self.codegen_switchint_terminator(bx, discr, targets);
                MergingSucc::False
            }

            mir::TerminatorKind::Return => {
                self.codegen_return_terminator(bx);
                MergingSucc::False
            }

            mir::TerminatorKind::Unreachable => {
                bx.unreachable();
                MergingSucc::False
            }

            mir::TerminatorKind::Drop { place, target, .. } => {
                let mergeable_succ = mergeable_succ();

                self.codegen_drop_terminator(bx, place, *target, mergeable_succ)
            }

            mir::TerminatorKind::Call {
                func,
                args,
                destination,
                target,
                ..
            } => {
                let mergeable_succ = mergeable_succ();

                self.codegen_call_terminator(
                    bx,
                    terminator,
                    func,
                    args,
                    destination,
                    *target,
                    mergeable_succ,
                )
            }
        }
    }

    fn codegen_argument(
        &mut self,
        bx: &mut Bx,
        op: OperandRef<Bx::Value>,
        llargs: &mut Vec<Bx::Value>,
        arg: &ArgAbi,
    ) {
        match &arg.mode {
            PassMode::Ignore => return,
            PassMode::Cast { .. } => bug!("not supported by RISL"),
            PassMode::Pair(..) => match op.val {
                Pair(a, b) => {
                    llargs.push(a);
                    llargs.push(b);

                    return;
                }
                _ => bug!("codegen_argument: {:?} invalid for pair argument", op),
            },
            PassMode::Indirect { .. }
                if let Ref(PlaceValue {
                    llval: a,
                    llextra: Some(b),
                    ..
                }) = op.val =>
            {
                llargs.push(a);
                llargs.push(b);

                return;
            }
            _ => {}
        }

        let shape = arg.layout.shape();

        let (mut llval, align, by_ref) = match op.val {
            Immediate(_) | Pair(..) => match &arg.mode {
                PassMode::Indirect { .. } => {
                    let scratch = PlaceValue::alloca(bx, &op.layout);

                    op.val.store(
                        bx,
                        &scratch.with_type(TyAndLayout {
                            ty: arg.ty,
                            layout: arg.layout.shape(),
                        }),
                    );

                    (scratch.llval, scratch.align, true)
                }
                _ => (op.immediate_or_packed_pair(bx), shape.abi_align, false),
            },
            Ref(PlaceValue {
                llval,
                llextra,
                align,
            }) => match &arg.mode {
                PassMode::Indirect { .. } if let Some(llextra) = llextra => {
                    llargs.push(llval);
                    llargs.push(llextra);

                    return;
                }
                _ => (llval, align, true),
            },
            ZeroSized => bug!("ZST {op:?} wasn't ignored, but was passed with abi {arg:?}"),
        };

        if by_ref && !matches!(arg.mode, PassMode::Indirect { .. }) {
            // Have to load the argument

            // We can't use `PlaceRef::load` here because the argument
            // may have a type we don't treat as immediate, but the ABI
            // used for this call is passing it by-value. In that case,
            // the load would just produce `OperandValue::Ref` instead
            // of the `OperandValue::Immediate` we need for the call.
            llval = bx.load(
                bx.backend_type(&TyAndLayout {
                    ty: arg.ty,
                    layout: arg.layout.shape(),
                }),
                llval,
                align,
            );

            if let ValueAbi::Scalar(scalar) = shape.abi {
                llval = bx.to_immediate_scalar(llval, scalar);
            }
        }

        llargs.push(llval);
    }

    fn codegen_arguments_untupled(
        &mut self,
        bx: &mut Bx,
        operand: &mir::Operand,
        llargs: &mut Vec<Bx::Value>,
        args: &[ArgAbi],
    ) {
        let tuple = self.codegen_operand(bx, operand);
        let count = tuple.layout.layout.fields.count();

        // Handle both by-ref and immediate tuples.
        if let Ref(place_val) = tuple.val {
            if place_val.llextra.is_some() {
                bug!("closure arguments must be sized");
            }

            let tuple_ptr = place_val.with_type(tuple.layout);

            for i in 0..count {
                let field_ptr = tuple_ptr.project_field(bx, i);
                let field = bx.load_operand(&field_ptr);

                self.codegen_argument(bx, field, llargs, &args[i]);
            }
        } else {
            // If the tuple is immediate, the elements are as well.
            for i in 0..count {
                let op = tuple.extract_field(bx, i);

                self.codegen_argument(bx, op, llargs, &args[i]);
            }
        }
    }

    fn predecessors(&mut self) -> &Vec<SmallVec<[BasicBlockIdx; 2]>> {
        self.bb_predecessors.get_or_insert_with(|| {
            let mut preds = vec![SmallVec::new(); self.mir.blocks.len()];

            for (bb, data) in self.mir.blocks.iter().enumerate() {
                for succ in data.terminator.successors() {
                    preds[succ].push(bb);
                }
            }

            preds
        })
    }

    fn unreachable_block(&mut self) -> Bx::BasicBlock {
        self.unreachable_block.unwrap_or_else(|| {
            let llbb = Bx::append_block(self.cx, self.llfn, "unreachable");
            let mut bx = Bx::build(self.cx, llbb);

            bx.unreachable();
            self.unreachable_block = Some(llbb);

            llbb
        })
    }

    /// Get the backend `BasicBlock` for a MIR `BasicBlock`, either already
    /// cached in `self.cached_llbbs`, or created on demand (and cached).
    // FIXME(eddyb) rename `llbb` and other `ll`-prefixed things to use a
    // more backend-agnostic prefix such as `cg` (i.e. this would be `cgbb`).
    pub fn llbb(&mut self, bb: mir::BasicBlockIdx) -> Bx::BasicBlock {
        self.try_llbb(bb).unwrap()
    }

    /// Like `llbb`, but may fail if the basic block should be skipped.
    pub(crate) fn try_llbb(&mut self, bb: mir::BasicBlockIdx) -> Option<Bx::BasicBlock> {
        match self.cached_llbbs[bb] {
            CachedLlbb::None => {
                let llbb = Bx::append_block(self.cx, self.llfn, &format!("{bb:?}"));

                self.cached_llbbs[bb] = CachedLlbb::Some(llbb);

                Some(llbb)
            }
            CachedLlbb::Some(llbb) => Some(llbb),
            CachedLlbb::Skip => None,
        }
    }

    fn make_return_dest(
        &mut self,
        bx: &mut Bx,
        dest: &mir::Place,
        fn_ret: &ArgAbi,
        llargs: &mut Vec<Bx::Value>,
        intrinsic: Option<IntrinsicDef>,
        target: Option<mir::BasicBlockIdx>,
    ) -> ReturnDest<Bx::Value> {
        if target.is_none() {
            return ReturnDest::Nothing;
        }

        // If the return is ignored, we can just return a do-nothing `ReturnDest`.
        if matches!(fn_ret.mode, PassMode::Ignore) {
            return ReturnDest::Nothing;
        }

        let dest = if dest.projection.is_empty() {
            match self.locals[dest.local].clone() {
                LocalRef::Place(dest) => dest,
                LocalRef::UnsizedPlace(_) => bug!("return type must be sized"),
                LocalRef::PendingOperand => {
                    // Handle temporary places, specifically `Operand` ones, as
                    // they don't have `alloca`s.
                    return if matches!(fn_ret.mode, PassMode::Indirect { .. }) {
                        // Odd, but possible, case, we have an operand temporary,
                        // but the calling convention has an indirect return.
                        let tmp = PlaceRef::alloca(
                            bx,
                            TyAndLayout {
                                ty: fn_ret.ty,
                                layout: fn_ret.layout.shape(),
                            },
                        );
                        tmp.storage_live(bx);
                        llargs.push(tmp.val.llval);
                        ReturnDest::IndirectOperand(tmp, dest.local)
                    } else if intrinsic.is_some() {
                        // Currently, intrinsics always need a location to store
                        // the result, so we create a temporary `alloca` for the
                        // result.
                        let tmp = PlaceRef::alloca(
                            bx,
                            TyAndLayout {
                                ty: fn_ret.ty,
                                layout: fn_ret.layout.shape(),
                            },
                        );
                        tmp.storage_live(bx);
                        ReturnDest::IndirectOperand(tmp, dest.local)
                    } else {
                        ReturnDest::DirectOperand(dest.local)
                    };
                }
                LocalRef::Operand(_) => {
                    bug!("place local already assigned to");
                }
            }
        } else {
            self.codegen_place(
                bx,
                mir::visit::PlaceRef {
                    local: dest.local,
                    projection: &dest.projection,
                },
            )
        };

        if matches!(fn_ret.mode, PassMode::Indirect { .. }) {
            if dest.val.align < dest.layout.layout.abi_align {
                // Currently, MIR code generation does not create calls
                // that store directly to fields of packed structs (in
                // fact, the calls it creates write only to temps).
                //
                // If someone changes that, please update this code path
                // to create a temporary.
                bug!("can't directly store to unaligned value");
            }

            llargs.push(dest.val.llval);

            ReturnDest::Nothing
        } else {
            ReturnDest::Store(dest)
        }
    }

    // Stores the return value of a function call into it's final location.
    fn store_return(
        &mut self,
        bx: &mut Bx,
        dest: ReturnDest<Bx::Value>,
        ret_abi: &ArgAbi,
        llval: Bx::Value,
    ) {
        use self::ReturnDest::*;

        match dest {
            Nothing => (),
            Store(dst) => bx.store_arg(ret_abi, llval, &dst),
            IndirectOperand(tmp, index) => {
                let op = bx.load_operand(&tmp);

                tmp.storage_dead(bx);
                self.overwrite_local(index, LocalRef::Operand(op));
            }
            DirectOperand(index) => {
                if let PassMode::Cast { .. } = ret_abi.mode {
                    bug!("not supported by RISL")
                }

                let op = OperandRef::from_immediate_or_packed_pair(
                    bx,
                    llval,
                    TyAndLayout {
                        ty: ret_abi.ty,
                        layout: ret_abi.layout.shape(),
                    },
                );

                self.overwrite_local(index, LocalRef::Operand(op));
            }
        }
    }
}

enum ReturnDest<V> {
    /// Do nothing; the return value is indirect or ignored.
    Nothing,
    /// Store the return value to the pointer.
    Store(PlaceRef<V>),
    /// Store an indirect return value to an operand local place.
    IndirectOperand(PlaceRef<V>, mir::Local),
    /// Store a direct return value to an operand local place.
    DirectOperand(mir::Local),
}
