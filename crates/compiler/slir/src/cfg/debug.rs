use std::io::Write;

use crate::cfg::{
    BasicBlock, Branch, BranchSelector, Cfg, ConstPtr, InlineConst, LocalBinding, RootIdentifier,
    Statement, StatementData, Terminator, Value,
};
use crate::ty::Type;
use crate::{Constant, Function, StorageBinding, UniformBinding, WorkgroupBinding};

pub type Result = std::io::Result<()>;

pub fn write_body<W: Write>(w: &mut W, cfg: &Cfg, function: Function) -> Result {
    let body = cfg
        .get_function_body(function)
        .expect("function not registered");
    let arg_count = body.argument_values().len();

    write_function_label(w, function)?;

    if arg_count > 0 {
        writeln!(w, "(")?;

        for i in 0..arg_count {
            write!(w, "  ")?;
            write_local_binding_label(w, body.argument_values()[i])?;
            write!(w, ": ")?;
            write_type(w, cfg, cfg[body.argument_values()[i]].ty())?;
            writeln!(w, ",")?;
        }

        writeln!(w, ") {{")?;
    } else {
        writeln!(w, "() {{")?;
    }

    for local_binding in &body.local_bindings()[arg_count..] {
        write_local_binding_decl(w, cfg, *local_binding)?;
    }

    writeln!(w)?;

    for basic_block in body.basic_blocks() {
        write_basic_block(w, cfg, *basic_block)?;
    }

    writeln!(w, "}}")?;

    Ok(())
}

fn write_local_binding_label<W: Write>(w: &mut W, local_binding: LocalBinding) -> Result {
    write!(w, "{:?}", local_binding)
}

fn write_uniform_binding_label<W: Write>(w: &mut W, storage_binding: UniformBinding) -> Result {
    write!(w, "{:?}", storage_binding)
}

fn write_storage_binding_label<W: Write>(w: &mut W, uniform_binding: StorageBinding) -> Result {
    write!(w, "{:?}", uniform_binding)
}

fn write_workgroup_binding_label<W: Write>(
    w: &mut W,
    workgroup_binding: WorkgroupBinding,
) -> Result {
    write!(w, "{:?}", workgroup_binding)
}

fn write_constant_label<W: Write>(w: &mut W, constant: Constant) -> Result {
    write!(w, "{}", constant.name)
}

fn write_function_label<W: Write>(w: &mut W, function: Function) -> Result {
    write!(w, "{}", function.name)
}

fn write_type<W: Write>(w: &mut W, cfg: &Cfg, ty: Type) -> Result {
    let ty_name = ty.to_string(cfg.ty());

    write!(w, "{}", ty_name)
}

fn write_basic_block_label<W: Write>(w: &mut W, basic_block: BasicBlock) -> Result {
    write!(w, "{:?}", basic_block)
}

fn write_value<W: Write>(w: &mut W, value: Value) -> Result {
    match value {
        Value::Local(local_binding) => {
            write_local_binding_label(w, local_binding)?;
        }
        Value::InlineConst(inline_const) => {
            write_inline_const(w, inline_const)?;
        }
    }

    Ok(())
}

fn write_inline_const<W: Write>(w: &mut W, inline_const: InlineConst) -> Result {
    match inline_const {
        InlineConst::U32(v) => {
            write!(w, "{}u32", v)?;
        }
        InlineConst::I32(v) => {
            write!(w, "{}i32", v)?;
        }
        InlineConst::F32(v) => {
            write!(w, "{}f32", v)?;
        }
        InlineConst::Bool(v) => {
            write!(w, "{}", v)?;
        }
        InlineConst::Ptr(v) => {
            write_const_ptr(w, v)?;
        }
    }

    Ok(())
}

fn write_const_ptr<W: Write>(w: &mut W, const_ptr: ConstPtr) -> Result {
    write!(w, "&")?;

    match const_ptr.root_identifier() {
        RootIdentifier::Local(b) => {
            write_local_binding_label(w, b)?;
        }
        RootIdentifier::Uniform(b) => {
            write_uniform_binding_label(w, b)?;
        }
        RootIdentifier::Storage(b) => {
            write_storage_binding_label(w, b)?;
        }
        RootIdentifier::Workgroup(b) => {
            write_workgroup_binding_label(w, b)?;
        }
        RootIdentifier::Constant(c) => {
            write_constant_label(w, c)?;
        }
    }

    Ok(())
}

fn write_local_binding_decl<W: Write>(w: &mut W, cfg: &Cfg, local_binding: LocalBinding) -> Result {
    write!(w, "  ")?;
    write_local_binding_label(w, local_binding)?;
    write!(w, ": ")?;
    write_type(w, cfg, cfg[local_binding].ty())?;
    writeln!(w, ";")?;

    Ok(())
}

fn write_basic_block<W: Write>(w: &mut W, cfg: &Cfg, basic_block: BasicBlock) -> Result {
    let data = &cfg[basic_block];

    write!(w, "  ")?;
    write_basic_block_label(w, basic_block)?;
    write!(w, " {{")?;

    for statemenet in data.statements() {
        writeln!(w)?;
        write!(w, "    ")?;
        write_statement(w, cfg, *statemenet)?;
    }

    writeln!(w)?;
    write!(w, "    ")?;
    write_terminator(w, cfg, data.terminator())?;
    writeln!(w)?;
    writeln!(w, "  }}")?;

    Ok(())
}

fn write_statement<W: Write>(w: &mut W, cfg: &Cfg, stmt: Statement) -> Result {
    match &cfg[stmt] {
        StatementData::Bind(_) => {
            write_stmt_bind(w, cfg, stmt)?;
        }
        StatementData::Uninitialized(_) => {}
        StatementData::Assign(_) => {
            write_stmt_assign(w, cfg, stmt)?;
        }
        StatementData::OpAlloca(_) => {
            write_stmt_op_alloca(w, cfg, stmt)?;
        }
        StatementData::OpLoad(_) => {
            write_stmt_op_load(w, cfg, stmt)?;
        }
        StatementData::OpStore(_) => {
            write_stmt_op_store(w, cfg, stmt)?;
        }
        StatementData::OpExtractElement(_) => {
            write_stmt_op_extract_element(w, cfg, stmt)?;
        }
        StatementData::OpExtractField(_) => {
            write_stmt_op_extract_field(w, cfg, stmt)?;
        }
        StatementData::OpElementPtr(_) => {
            write_stmt_op_element_ptr(w, cfg, stmt)?;
        }
        StatementData::OpFieldPtr(_) => {
            write_stmt_op_field_ptr(w, cfg, stmt)?;
        }
        StatementData::OpVariantPtr(_) => {
            write_stmt_op_variant_ptr(w, cfg, stmt)?;
        }
        StatementData::OpGetDiscriminant(_) => {
            write_stmt_op_get_discriminant(w, cfg, stmt)?;
        }
        StatementData::OpSetDiscriminant(_) => {
            write_stmt_op_set_discriminant(w, cfg, stmt)?;
        }
        StatementData::OpOffsetSlice(_) => {
            write_stmt_op_offset_slice(w, cfg, stmt)?;
        }
        StatementData::OpUnary(_) => {
            write_stmt_op_unary(w, cfg, stmt)?;
        }
        StatementData::OpBinary(_) => {
            write_stmt_op_binary(w, cfg, stmt)?;
        }
        StatementData::OpCall(_) => {
            write_stmt_op_call(w, cfg, stmt)?;
        }
        StatementData::OpConvertToU32(_) => {
            write_stmt_op_convert_to_u32(w, cfg, stmt)?;
        }
        StatementData::OpConvertToI32(_) => {
            write_stmt_op_convert_to_i32(w, cfg, stmt)?;
        }
        StatementData::OpConvertToF32(_) => {
            write_stmt_op_convert_to_f32(w, cfg, stmt)?;
        }
        StatementData::OpConvertToBool(_) => {
            write_stmt_op_convert_to_bool(w, cfg, stmt)?;
        }
        StatementData::OpArrayLength(_) => {
            write_stmt_op_array_length(w, cfg, stmt)?;
        }
    }

    Ok(())
}

fn write_stmt_bind<W: Write>(w: &mut W, cfg: &Cfg, stmt: Statement) -> Result {
    let data = cfg[stmt].expect_bind();

    write_local_binding_label(w, data.local_binding())?;
    write!(w, " = ")?;
    write_value(w, data.value())?;
    write!(w, ";")?;

    Ok(())
}

fn write_stmt_assign<W: Write>(w: &mut W, cfg: &Cfg, stmt: Statement) -> Result {
    let data = cfg[stmt].expect_assign();

    write_local_binding_label(w, data.local_binding())?;
    write!(w, " = ")?;
    write_value(w, data.value())?;
    write!(w, ";")?;

    Ok(())
}

fn write_stmt_op_alloca<W: Write>(w: &mut W, cfg: &Cfg, stmt: Statement) -> Result {
    let data = cfg[stmt].expect_op_alloca();

    write_local_binding_label(w, data.result())?;
    write!(w, " = alloca ")?;
    write_type(w, cfg, data.intrinsic().ty)?;
    write!(w, ";")?;

    Ok(())
}

fn write_stmt_op_load<W: Write>(w: &mut W, cfg: &Cfg, stmt: Statement) -> Result {
    let data = cfg[stmt].expect_op_load();

    write_local_binding_label(w, data.result())?;
    write!(w, " = *")?;
    write_value(w, data.ptr())?;
    write!(w, ";")?;

    Ok(())
}

fn write_stmt_op_store<W: Write>(w: &mut W, _cfg: &Cfg, stmt: Statement) -> Result {
    let data = _cfg[stmt].expect_op_store();

    write!(w, "*")?;
    write_value(w, data.ptr())?;
    write!(w, " = ")?;
    write_value(w, data.value())?;
    write!(w, ";")?;

    Ok(())
}

fn write_stmt_op_extract_element<W: Write>(w: &mut W, _cfg: &Cfg, stmt: Statement) -> Result {
    let data = _cfg[stmt].expect_op_extract_element();

    write_local_binding_label(w, data.result())?;
    write!(w, " = ")?;
    write_value(w, data.value())?;
    write!(w, "[")?;
    write_value(w, data.element_index())?;
    write!(w, "];")?;

    Ok(())
}

fn write_stmt_op_extract_field<W: Write>(w: &mut W, _cfg: &Cfg, stmt: Statement) -> Result {
    let data = _cfg[stmt].expect_op_extract_field();

    write_local_binding_label(w, data.result())?;
    write!(w, " = ")?;
    write_value(w, data.value())?;
    write!(w, ".{};", data.field_index())?;

    Ok(())
}

fn write_stmt_op_element_ptr<W: Write>(w: &mut W, _cfg: &Cfg, stmt: Statement) -> Result {
    let data = _cfg[stmt].expect_op_element_ptr();

    write_local_binding_label(w, data.result())?;
    write!(w, " = &(")?;
    write_value(w, data.ptr())?;
    write!(w, ")[")?;
    write_value(w, data.element_index())?;
    write!(w, "];")?;

    Ok(())
}

fn write_stmt_op_field_ptr<W: Write>(w: &mut W, _cfg: &Cfg, stmt: Statement) -> Result {
    let data = _cfg[stmt].expect_op_field_ptr();

    write_local_binding_label(w, data.result())?;
    write!(w, " = &(")?;
    write_value(w, data.ptr())?;
    write!(w, ").{};", data.field_index())?;

    Ok(())
}

fn write_stmt_op_variant_ptr<W: Write>(w: &mut W, _cfg: &Cfg, stmt: Statement) -> Result {
    let data = _cfg[stmt].expect_op_variant_ptr();

    write_local_binding_label(w, data.result())?;
    write!(w, " = variant_ptr(")?;
    write_value(w, data.ptr())?;
    write!(w, ":{});", data.variant_index())?;

    Ok(())
}

fn write_stmt_op_get_discriminant<W: Write>(w: &mut W, _cfg: &Cfg, stmt: Statement) -> Result {
    let data = _cfg[stmt].expect_op_get_discriminant();

    write_local_binding_label(w, data.result())?;
    write!(w, " = get_discriminant(")?;
    write_value(w, data.ptr())?;
    write!(w, ");")?;

    Ok(())
}

fn write_stmt_op_set_discriminant<W: Write>(w: &mut W, _cfg: &Cfg, stmt: Statement) -> Result {
    let data = _cfg[stmt].expect_op_set_discriminant();

    write!(w, "set_discriminant(")?;
    write_value(w, data.ptr())?;
    write!(w, ", {});", data.variant_index())?;

    Ok(())
}

fn write_stmt_op_offset_slice<W: Write>(w: &mut W, _cfg: &Cfg, stmt: Statement) -> Result {
    let data = _cfg[stmt].expect_op_offset_slice_ptr();

    write_local_binding_label(w, data.result())?;
    write!(w, " = &(")?;
    write_value(w, data.ptr())?;
    write!(w, ")[")?;
    write_value(w, data.offset())?;
    write!(w, "..];")?;

    Ok(())
}

fn write_stmt_op_unary<W: Write>(w: &mut W, _cfg: &Cfg, stmt: Statement) -> Result {
    let data = _cfg[stmt].expect_op_unary();

    write_local_binding_label(w, data.result())?;
    write!(w, " = {}", data.operator())?;
    write_value(w, data.value())?;
    write!(w, ";")?;

    Ok(())
}

fn write_stmt_op_binary<W: Write>(w: &mut W, _cfg: &Cfg, stmt: Statement) -> Result {
    let data = _cfg[stmt].expect_op_binary();

    write_local_binding_label(w, data.result())?;
    write!(w, " = ")?;
    write_value(w, data.lhs())?;
    write!(w, " {} ", data.operator())?;
    write_value(w, data.rhs())?;
    write!(w, ";")?;

    Ok(())
}

fn write_stmt_op_call<W: Write>(w: &mut W, _cfg: &Cfg, stmt: Statement) -> Result {
    let data = _cfg[stmt].expect_op_call();

    if let Some(result) = data.maybe_result() {
        write_local_binding_label(w, result)?;
        write!(w, " = ")?;
    }

    write_function_label(w, data.callee())?;
    write!(w, "(")?;
    for (i, arg) in data.arguments().iter().enumerate() {
        if i > 0 {
            write!(w, ", ")?;
        }
        write_value(w, *arg)?;
    }
    write!(w, ");")?;

    Ok(())
}

fn write_stmt_op_convert_to_u32<W: Write>(w: &mut W, _cfg: &Cfg, stmt: Statement) -> Result {
    let data = _cfg[stmt].expect_op_convert_to_u32();

    write_local_binding_label(w, data.result())?;
    write!(w, " = u32(")?;
    write_value(w, data.value())?;
    write!(w, ");")?;

    Ok(())
}

fn write_stmt_op_convert_to_i32<W: Write>(w: &mut W, _cfg: &Cfg, stmt: Statement) -> Result {
    let data = _cfg[stmt].expect_op_convert_to_i32();

    write_local_binding_label(w, data.result())?;
    write!(w, " = i32(")?;
    write_value(w, data.value())?;
    write!(w, ");")?;

    Ok(())
}

fn write_stmt_op_convert_to_f32<W: Write>(w: &mut W, _cfg: &Cfg, stmt: Statement) -> Result {
    let data = _cfg[stmt].expect_op_convert_to_f32();

    write_local_binding_label(w, data.result())?;
    write!(w, " = f32(")?;
    write_value(w, data.value())?;
    write!(w, ");")?;

    Ok(())
}

fn write_stmt_op_convert_to_bool<W: Write>(w: &mut W, _cfg: &Cfg, stmt: Statement) -> Result {
    let data = _cfg[stmt].expect_op_convert_to_bool();

    write_local_binding_label(w, data.result())?;
    write!(w, " = bool(")?;
    write_value(w, data.value())?;
    write!(w, ");")?;

    Ok(())
}

fn write_stmt_op_array_length<W: Write>(w: &mut W, _cfg: &Cfg, stmt: Statement) -> Result {
    let data = _cfg[stmt].expect_op_array_length();

    write_local_binding_label(w, data.result())?;
    write!(w, " = array_length(")?;
    write_value(w, data.ptr())?;
    write!(w, ");")?;

    Ok(())
}

fn write_terminator<W: Write>(w: &mut W, _cfg: &Cfg, terminator: &Terminator) -> Result {
    match terminator {
        Terminator::Branch(branch) => {
            write_terminator_branch(w, _cfg, branch)?;
        }
        Terminator::Return(value) => {
            write_terminator_return(w, _cfg, value)?;
        }
        Terminator::Unreachable => {
            write_terminator_unreachable(w)?;
        }
    }

    write!(w, ";")?;

    Ok(())
}

fn write_terminator_branch<W: Write>(w: &mut W, _cfg: &Cfg, branch: &Branch) -> Result {
    write!(w, "branch ")?;

    match branch.selector() {
        BranchSelector::Single => {
            write_branch_selector_single(w, branch)?;
        }
        BranchSelector::Bool(cond) => {
            write_branch_selector_bool(w, *cond, branch)?;
        }
        BranchSelector::U32(val) => {
            write_branch_selector_u32(w, *val, branch)?;
        }
        BranchSelector::Case { value, cases } => {
            write_branch_selector_case(w, *value, cases, branch)?;
        }
    }

    Ok(())
}

fn write_branch_selector_single<W: Write>(w: &mut W, branch: &Branch) -> Result {
    write_basic_block_label(w, branch.targets()[0])?;

    Ok(())
}

fn write_branch_selector_bool<W: Write>(w: &mut W, cond: LocalBinding, branch: &Branch) -> Result {
    write!(w, "bool ")?;
    write_local_binding_label(w, cond)?;
    write!(w, ": ")?;
    write_basic_block_label(w, branch.targets()[0])?;
    write!(w, " else ")?;
    write_basic_block_label(w, branch.targets()[1])?;

    Ok(())
}

fn write_branch_selector_u32<W: Write>(w: &mut W, val: LocalBinding, branch: &Branch) -> Result {
    write!(w, "u32 ")?;
    write_local_binding_label(w, val)?;
    write!(w, ": [")?;

    for (i, target) in branch.targets().iter().enumerate() {
        if i > 0 {
            write!(w, ", ")?;
        }
        write_basic_block_label(w, *target)?;
    }

    write!(w, "]")?;

    Ok(())
}

fn write_branch_selector_case<W: Write>(
    w: &mut W,
    value: LocalBinding,
    cases: &[u32],
    branch: &Branch,
) -> Result {
    write!(w, "case ")?;
    write_local_binding_label(w, value)?;
    write!(w, ": [")?;

    for (i, case) in cases.iter().enumerate() {
        if i > 0 {
            write!(w, ", ")?;
        }
        write!(w, "{}", case)?;
        write!(w, " => ")?;
        write_basic_block_label(w, branch.targets()[i])?;
    }

    write!(w, ", default => ")?;
    write_basic_block_label(w, *branch.targets().last().unwrap())?;
    write!(w, "]")?;

    Ok(())
}

fn write_terminator_return<W: Write>(w: &mut W, _cfg: &Cfg, value: &Option<Value>) -> Result {
    write!(w, "return")?;

    if let Some(value) = value {
        write!(w, " ")?;
        write_value(w, *value)?;
    }

    Ok(())
}

fn write_terminator_unreachable<W: Write>(w: &mut W) -> Result {
    write!(w, "unreachable")?;

    Ok(())
}
