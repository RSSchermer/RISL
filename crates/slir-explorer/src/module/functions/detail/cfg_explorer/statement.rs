use leptos::prelude::*;

use crate::module::functions::detail::cfg_explorer::value::Value;
use crate::module::module::use_module_data;
use crate::module::url::function_url;

#[component]
pub fn Statement(statement: slir::cfg::Statement) -> impl IntoView {
    let module_data = use_module_data();

    let inner = match &module_data.cfg.read_value()[statement] {
        slir::cfg::StatementData::Assign(_) => view! { <Assign statement/> }.into_any(),
        slir::cfg::StatementData::Bind(_) => view! { <Bind statement/> }.into_any(),
        slir::cfg::StatementData::Uninitialized(_) => {
            view! { <Uninitialized statement/> }.into_any()
        }
        slir::cfg::StatementData::OpAlloca(_) => view! { <OpAlloca statement/> }.into_any(),
        slir::cfg::StatementData::OpLoad(_) => view! { <OpLoad statement/> }.into_any(),
        slir::cfg::StatementData::OpStore(_) => view! { <OpStore statement/> }.into_any(),
        slir::cfg::StatementData::OpExtractField(_) => {
            view! { <OpExtractField statement/> }.into_any()
        }
        slir::cfg::StatementData::OpExtractElement(_) => {
            view! { <OpExtractElement statement/> }.into_any()
        }
        slir::cfg::StatementData::OpFieldPtr(_) => view! { <OpFieldPtr statement/> }.into_any(),
        slir::cfg::StatementData::OpElementPtr(_) => view! { <OpElementPtr statement/> }.into_any(),
        slir::cfg::StatementData::OpVariantPtr(_) => view! { <OpVariantPtr statement/> }.into_any(),
        slir::cfg::StatementData::OpGetDiscriminant(_) => {
            view! { <OpGetDiscriminant statement/> }.into_any()
        }
        slir::cfg::StatementData::OpSetDiscriminant(_) => {
            view! { <OpSetDiscriminant statement/> }.into_any()
        }
        slir::cfg::StatementData::OpOffsetSlice(_) => {
            view! { <OpOffsetSlice statement/> }.into_any()
        }
        slir::cfg::StatementData::OpUnary(_) => view! { <OpUnary statement/> }.into_any(),
        slir::cfg::StatementData::OpBinary(_) => view! { <OpBinary statement/> }.into_any(),
        slir::cfg::StatementData::OpCall(_) => view! { <OpCall statement/> }.into_any(),
        slir::cfg::StatementData::OpCaseToBranchSelector(_) => {
            view! { <OpCaseToBranchSelector statement/> }.into_any()
        }
        slir::cfg::StatementData::OpBoolToBranchSelector(_) => {
            view! { <OpBoolToBranchSelector statement/> }.into_any()
        }
        slir::cfg::StatementData::OpConvertToU32(_) => {
            view! { <OpConvertToU32 statement/> }.into_any()
        }
        slir::cfg::StatementData::OpConvertToI32(_) => {
            view! { <OpConvertToI32 statement/> }.into_any()
        }
        slir::cfg::StatementData::OpConvertToF32(_) => {
            view! { <OpConvertToF32 statement/> }.into_any()
        }
        slir::cfg::StatementData::OpConvertToBool(_) => {
            view! { <OpConvertToBool statement/> }.into_any()
        }
        slir::cfg::StatementData::OpArrayLength(_) => {
            view! { <OpArrayLength statement/> }.into_any()
        }
    };

    view! {
        <p>{inner}</p>
    }
}

#[component]
pub fn Assign(statement: slir::cfg::Statement) -> impl IntoView {
    let module_data = use_module_data();
    let cfg = module_data.cfg.read_value();
    let stmt = cfg[statement].expect_assign();
    let binding = stmt.local_binding();
    let value = stmt.value();

    view! {
        <Value value=binding.into()/>" = "<Value value/>
    }
}

#[component]
pub fn Bind(statement: slir::cfg::Statement) -> impl IntoView {
    let module_data = use_module_data();
    let cfg = module_data.cfg.read_value();
    let stmt = cfg[statement].expect_bind();
    let binding = stmt.local_binding();
    let value = stmt.value();

    view! {
        <Value value=binding.into()/>" = "<Value value/>
    }
}

#[component]
pub fn Uninitialized(statement: slir::cfg::Statement) -> impl IntoView {
    let module_data = use_module_data();
    let cfg = module_data.cfg.read_value();
    let stmt = cfg[statement].expect_uninitialized();
    let binding = stmt.local_binding();

    view! {
        <Value value=binding.into()/>" = uninitialized"
    }
}

#[component]
pub fn OpAlloca(statement: slir::cfg::Statement) -> impl IntoView {
    let module_data = use_module_data();
    let cfg = module_data.cfg.read_value();
    let stmt = cfg[statement].expect_op_alloca();
    let binding = stmt.result();

    view! {
        <Value value=binding.into()/> " = alloca"
    }
}

#[component]
pub fn OpLoad(statement: slir::cfg::Statement) -> impl IntoView {
    let module_data = use_module_data();
    let cfg = module_data.cfg.read_value();
    let stmt = cfg[statement].expect_op_load();
    let binding = stmt.result();
    let pointer = stmt.ptr();

    view! {
        <Value value=binding.into()/> " = load "<Value value=pointer/>
    }
}

#[component]
pub fn OpStore(statement: slir::cfg::Statement) -> impl IntoView {
    let module_data = use_module_data();
    let cfg = module_data.cfg.read_value();
    let stmt = cfg[statement].expect_op_store();
    let pointer = stmt.ptr();
    let value = stmt.value();

    view! {
        "store "<Value value/>" into "<Value value=pointer/>
    }
}

#[component]
pub fn OpExtractField(statement: slir::cfg::Statement) -> impl IntoView {
    let module_data = use_module_data();
    let cfg = module_data.cfg.read_value();
    let stmt = cfg[statement].expect_op_extract_field();
    let aggregate = stmt.value();
    let field_index = stmt.field_index();
    let binding = stmt.result();

    view! {
        <Value value=binding.into()/>" = "<Value value=aggregate/>"._"{field_index}
    }
}

#[component]
pub fn OpExtractElement(statement: slir::cfg::Statement) -> impl IntoView {
    let module_data = use_module_data();
    let cfg = module_data.cfg.read_value();
    let stmt = cfg[statement].expect_op_extract_element();
    let aggregate = stmt.value();
    let index = stmt.element_index();
    let binding = stmt.result();

    view! {
        <Value value=binding.into()/>" = "<Value value=aggregate/>"["<Value value=index/>"]"
    }
}

#[component]
pub fn OpFieldPtr(statement: slir::cfg::Statement) -> impl IntoView {
    let module_data = use_module_data();
    let cfg = module_data.cfg.read_value();
    let stmt = cfg[statement].expect_op_field_ptr();
    let ptr = stmt.ptr();
    let field_index = stmt.field_index();
    let binding = stmt.result();

    view! {
        <Value value=binding.into()/>" = "<Value value=ptr/>"._"{field_index}
    }
}

#[component]
pub fn OpElementPtr(statement: slir::cfg::Statement) -> impl IntoView {
    let module_data = use_module_data();
    let cfg = module_data.cfg.read_value();
    let stmt = cfg[statement].expect_op_element_ptr();
    let ptr = stmt.ptr();
    let index = stmt.element_index();
    let binding = stmt.result();

    view! {
        <Value value=binding.into()/>" = &"<Value value=ptr/>"["<Value value=index/>"]"
    }
}

#[component]
pub fn OpVariantPtr(statement: slir::cfg::Statement) -> impl IntoView {
    let module_data = use_module_data();
    let cfg = module_data.cfg.read_value();
    let stmt = cfg[statement].expect_op_variant_ptr();
    let ptr = stmt.ptr();
    let variant_index = stmt.variant_index();
    let binding = stmt.result();

    view! {
        <Value value=binding.into()/>" = variant-ptr "<Value value=ptr/>":"{variant_index}
    }
}

#[component]
pub fn OpGetDiscriminant(statement: slir::cfg::Statement) -> impl IntoView {
    let module_data = use_module_data();
    let cfg = module_data.cfg.read_value();
    let stmt = cfg[statement].expect_op_get_discriminant();
    let ptr = stmt.ptr();
    let binding = stmt.result();

    view! {
        <Value value=binding.into()/>" = get-discriminant "<Value value=ptr/>
    }
}

#[component]
pub fn OpSetDiscriminant(statement: slir::cfg::Statement) -> impl IntoView {
    let module_data = use_module_data();
    let cfg = module_data.cfg.read_value();
    let stmt = cfg[statement].expect_op_set_discriminant();
    let ptr = stmt.ptr();
    let variant_index = stmt.variant_index();

    view! {
        "set-discriminant "{variant_index}" on "<Value value=ptr/>
    }
}

#[component]
pub fn OpOffsetSlice(statement: slir::cfg::Statement) -> impl IntoView {
    let module_data = use_module_data();
    let cfg = module_data.cfg.read_value();
    let stmt = cfg[statement].expect_op_offset_slice_ptr();
    let ptr = stmt.ptr();
    let offset = stmt.offset();
    let binding = stmt.result();

    view! {
        <Value value=binding.into()/>
        " = offset "
        <Value value=ptr/>
        " by "
        <Value value=offset/>
    }
}

#[component]
pub fn OpUnary(statement: slir::cfg::Statement) -> impl IntoView {
    let module_data = use_module_data();
    let cfg = module_data.cfg.read_value();
    let stmt = cfg[statement].expect_op_unary();
    let value = stmt.value();
    let operator = stmt.operator();
    let binding = stmt.result();

    view! {
        <Value value=binding.into()/>{format!(" = {}", operator)}<Value value/>
    }
}

#[component]
pub fn OpBinary(statement: slir::cfg::Statement) -> impl IntoView {
    let module_data = use_module_data();
    let cfg = module_data.cfg.read_value();
    let stmt = cfg[statement].expect_op_binary();
    let operator = stmt.operator();
    let lhs = stmt.lhs();
    let rhs = stmt.rhs();
    let binding = stmt.result();

    view! {
        <Value value=binding.into()/>" = "<Value value=lhs/>{format!(" {} ", operator)}<Value value=rhs/>
    }
}

#[component]
pub fn OpCall(statement: slir::cfg::Statement) -> impl IntoView {
    let module_data = use_module_data();
    let module_name = module_data.module.read_value().name;
    let cfg = module_data.cfg.read_value();
    let stmt = cfg[statement].expect_op_call();
    let callee = stmt.callee();
    let binding = stmt.maybe_result();

    let mut arg_views = Vec::new();
    let mut is_first = true;

    for arg in stmt.arguments().iter().copied() {
        if !is_first {
            arg_views.push(view! {", "}.into_any());
        }

        arg_views.push(view! { <Value value=arg/> }.into_any());

        is_first = false;
    }

    view! {
        {{move || binding.map(|binding | view! {
            <Value value=binding.into()/>" = "
        })}}

        <a href=function_url(module_name, callee)>
            {callee.name.to_string()}
        </a>
        "("{arg_views}")"
    }
}

#[component]
pub fn OpCaseToBranchSelector(statement: slir::cfg::Statement) -> impl IntoView {
    let module_data = use_module_data();
    let cfg = module_data.cfg.read_value();
    let stmt = cfg[statement].expect_op_case_to_branch_selector();
    let value = stmt.value();
    let cases = stmt
        .cases()
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let binding = stmt.result();

    view! {
        <Value value=binding.into()/>" = predicate-from-case "<Value value/>" ["{cases}"]"
    }
}

#[component]
pub fn OpBoolToBranchSelector(statement: slir::cfg::Statement) -> impl IntoView {
    let module_data = use_module_data();
    let cfg = module_data.cfg.read_value();
    let stmt = cfg[statement].expect_op_bool_to_branch_selector();
    let value = stmt.value();
    let binding = stmt.result();

    view! {
        <Value value=binding.into()/>" = predicate-from-bool "<Value value/>
    }
}

#[component]
pub fn OpConvertToU32(statement: slir::cfg::Statement) -> impl IntoView {
    let module_data = use_module_data();
    let cfg = module_data.cfg.read_value();
    let stmt = cfg[statement].expect_op_convert_to_u32();
    let value = stmt.value();
    let binding = stmt.result();

    view! {
        <Value value=binding.into()/>" = u32("<Value value/>")"
    }
}

#[component]
pub fn OpConvertToI32(statement: slir::cfg::Statement) -> impl IntoView {
    let module_data = use_module_data();
    let cfg = module_data.cfg.read_value();
    let stmt = cfg[statement].expect_op_convert_to_i32();
    let value = stmt.value();
    let binding = stmt.result();

    view! {
        <Value value=binding.into()/>" = i32("<Value value/>")"
    }
}

#[component]
pub fn OpConvertToF32(statement: slir::cfg::Statement) -> impl IntoView {
    let module_data = use_module_data();
    let cfg = module_data.cfg.read_value();
    let stmt = cfg[statement].expect_op_convert_to_f32();
    let value = stmt.value();
    let binding = stmt.result();

    view! {
        <Value value=binding.into()/>" = f32("<Value value/>")"
    }
}

#[component]
pub fn OpConvertToBool(statement: slir::cfg::Statement) -> impl IntoView {
    let module_data = use_module_data();
    let cfg = module_data.cfg.read_value();
    let stmt = cfg[statement].expect_op_convert_to_bool();
    let value = stmt.value();
    let binding = stmt.result();

    view! {
        <Value value=binding.into()/>" = bool("<Value value/>")"
    }
}

#[component]
pub fn OpArrayLength(statement: slir::cfg::Statement) -> impl IntoView {
    let module_data = use_module_data();
    let cfg = module_data.cfg.read_value();
    let stmt = cfg[statement].expect_op_array_length();
    let value = stmt.ptr();
    let binding = stmt.result();

    view! {
        <Value value=binding.into()/>" = array-length("<Value value/>")"
    }
}
