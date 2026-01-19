use rustc_middle::bug;
use rustc_public::CrateDef;
use rustc_public::abi::{PassMode, ValueAbi};
use rustc_public::crate_def::Attribute;
use rustc_public::mir::mono::{Instance, MonoItem};
use slir::BinaryOperator;
use slir::cfg::{BlockPosition, LocalBinding, Terminator};

use crate::slir_build::context::CodegenContext;
use crate::stable_cg::traits::MiscCodegenMethods;

pub fn maybe_rislc_intrinsic(item: MonoItem, cx: &CodegenContext) -> Option<MonoItem> {
    let MonoItem::Fn(instance) = item else {
        return Some(item);
    };

    if let Some(intrinsic) = RislIntrinsic::try_from(&instance) {
        match intrinsic {
            RislIntrinsic::Add => define_add(instance, cx),
            RislIntrinsic::Sub => define_sub(instance, cx),
            RislIntrinsic::Mul => define_mul(instance, cx),
            RislIntrinsic::Div => define_div(instance, cx),
            RislIntrinsic::MemResourceAsRef => define_mem_resource_as_ref(instance, cx),
        }

        None
    } else {
        Some(MonoItem::Fn(instance))
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RislIntrinsic {
    Add,
    Sub,
    Mul,
    Div,
    MemResourceAsRef,
}

impl RislIntrinsic {
    pub fn try_from(instance: &Instance) -> Option<Self> {
        instance
            .def
            .tool_attrs(&["rislc".into(), "intrinsic".into()])
            .first()
            .map(resolve_intrinsic)
    }
}

fn resolve_intrinsic(attr: &Attribute) -> RislIntrinsic {
    match attr.as_str().trim() {
        "#[rislc::intrinsic(add)]" => RislIntrinsic::Add,
        "#[rislc::intrinsic(sub)]" => RislIntrinsic::Sub,
        "#[rislc::intrinsic(mul)]" => RislIntrinsic::Mul,
        "#[rislc::intrinsic(div)]" => RislIntrinsic::Div,
        "#[rislc::intrinsic(mem_resource_as_ref)]" => RislIntrinsic::MemResourceAsRef,
        _ => bug!("unsupported rislc intrinsic: {}", attr.as_str()),
    }
}

fn define_arith_op(instance: Instance, cx: &CodegenContext, op: BinaryOperator) {
    let function = cx.get_fn(&instance);
    let fn_abi = instance
        .fn_abi()
        .expect("should have a known ABI during codegen");

    let lhs_mode = &fn_abi.args[0].mode;
    let rhs_mode = &fn_abi.args[1].mode;
    let ret_mode = &fn_abi.ret.mode;

    let mut cfg = cx.cfg.borrow_mut();
    let body = cfg
        .get_function_body(function)
        .expect("function should have been predefined");
    let bb = body.entry_block();

    let (lhs_arg, rhs_arg, ret_binding) = match ret_mode {
        PassMode::Direct(_) => (body.argument_values()[0], body.argument_values()[1], None),
        PassMode::Indirect { .. } => (
            body.argument_values()[1],
            body.argument_values()[2],
            Some(body.argument_values()[0]),
        ),
        _ => bug!("unexpected return value pass-mode: {:?}", ret_mode),
    };

    let mut resolve_operand = |arg: LocalBinding, mode: &PassMode| match mode {
        PassMode::Direct(_) => arg,
        PassMode::Indirect { .. } => {
            let (_, operand) = cfg.add_stmt_op_load(bb, BlockPosition::Append, arg.into());

            operand
        }
        _ => bug!("unexpected argument pass-mode: {:?}", mode),
    };

    let lhs = resolve_operand(lhs_arg, lhs_mode);
    let rhs = resolve_operand(rhs_arg, rhs_mode);

    let (_, result) = cfg.add_stmt_op_binary(bb, BlockPosition::Append, op, lhs.into(), rhs.into());

    if let Some(ret_binding) = ret_binding {
        cfg.add_stmt_op_store(bb, BlockPosition::Append, ret_binding.into(), result.into());
    } else {
        cfg.set_terminator(bb, Terminator::return_value(result.into()));
    }
}

fn define_add(instance: Instance, cx: &CodegenContext) {
    define_arith_op(instance, cx, BinaryOperator::Add);
}

fn define_sub(instance: Instance, cx: &CodegenContext) {
    define_arith_op(instance, cx, BinaryOperator::Sub);
}

fn define_mul(instance: Instance, cx: &CodegenContext) {
    define_arith_op(instance, cx, BinaryOperator::Mul);
}

fn define_div(instance: Instance, cx: &CodegenContext) {
    define_arith_op(instance, cx, BinaryOperator::Div);
}

fn define_mem_resource_as_ref(instance: Instance, cx: &CodegenContext) {
    let function = cx.get_fn(&instance);

    let module = cx.module.borrow();
    let fn_sig = &module.fn_sigs[function];
    let ptr_ty = fn_sig.args[0].ty;
    let ret_ty = fn_sig.ret_ty.expect("should return a reference");

    let fn_abi = instance
        .fn_abi()
        .expect("intrinsic must have known ABI at codegen time");
    let ret_abi = fn_abi.ret.layout.shape().abi;

    let mut cfg = cx.cfg.borrow_mut();
    let body = cfg
        .get_function_body(function)
        .expect("function should have been predefined");
    let bb = body.entry_block();
    let ptr_arg = body.argument_values()[0];

    // Note that the pointer argument type will have already been unpacked, see
    // super::context::CodegenContext::try_register_as_rislc_mem_resource_ty.

    if matches!(ret_abi, ValueAbi::ScalarPair(_, _)) {
        // The mem resource contains an unsized type. In this case we'll have to return a pair
        // consisting of the pointer and the length of the array or the unsized tail value.
        let (_, alloca_ptr) = cfg.add_stmt_op_alloca(bb, BlockPosition::Append, ret_ty);

        let (_, ptr_ptr) =
            cfg.add_stmt_op_field_ptr(bb, BlockPosition::Append, alloca_ptr.into(), 0);
        cfg.add_stmt_op_store(bb, BlockPosition::Append, ptr_ptr.into(), ptr_arg.into());

        let (_, len_ptr) =
            cfg.add_stmt_op_field_ptr(bb, BlockPosition::Append, alloca_ptr.into(), 1);
        let (_, len) = cfg.add_stmt_op_array_length(bb, BlockPosition::Append, ptr_arg.into());
        cfg.add_stmt_op_store(bb, BlockPosition::Append, len_ptr.into(), len.into());

        let (_, pair) = cfg.add_stmt_op_load(bb, BlockPosition::Append, alloca_ptr.into());

        cfg.set_terminator(bb, Terminator::return_value(pair.into()));
    } else {
        // The mem resource contains a sized type. In this case we simply return the pointer
        // argument.
        cfg.set_terminator(bb, Terminator::return_value(ptr_arg.into()));
    }
}
