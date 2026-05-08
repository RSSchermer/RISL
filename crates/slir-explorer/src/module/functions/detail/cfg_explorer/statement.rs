use leptos::prelude::*;
use slir::cfg::Cfg;

use crate::module::functions::detail::cfg_explorer::value::Value;
use crate::module::module::use_module_data;
use crate::module::url::function_url;

#[component]
pub fn Statement(cfg: StoredValue<Cfg>, statement: slir::cfg::Statement) -> impl IntoView {
    let inner = match &cfg.read_value()[statement] {
        slir::cfg::StatementData::Assign(_) => view! { <Assign cfg statement/> }.into_any(),
        slir::cfg::StatementData::Bind(_) => view! { <Bind cfg statement/> }.into_any(),
        slir::cfg::StatementData::Uninitialized(_) => {
            view! { <Uninitialized cfg statement/> }.into_any()
        }
        slir::cfg::StatementData::OpAlloca(_) => view! { <OpAlloca cfg statement/> }.into_any(),
        slir::cfg::StatementData::OpLoad(_) => view! { <OpLoad cfg statement/> }.into_any(),
        slir::cfg::StatementData::OpStore(_) => view! { <OpStore cfg statement/> }.into_any(),
        slir::cfg::StatementData::OpExtractField(_) => {
            view! { <OpExtractField cfg statement/> }.into_any()
        }
        slir::cfg::StatementData::OpExtractElement(_) => {
            view! { <OpExtractElement cfg statement/> }.into_any()
        }
        slir::cfg::StatementData::OpFieldPtr(_) => view! { <OpFieldPtr cfg statement/> }.into_any(),
        slir::cfg::StatementData::OpElementPtr(_) => {
            view! { <OpElementPtr cfg statement/> }.into_any()
        }
        slir::cfg::StatementData::OpVariantPtr(_) => {
            view! { <OpVariantPtr cfg statement/> }.into_any()
        }
        slir::cfg::StatementData::OpGetDiscriminant(_) => {
            view! { <OpGetDiscriminant cfg statement/> }.into_any()
        }
        slir::cfg::StatementData::OpSetDiscriminant(_) => {
            view! { <OpSetDiscriminant cfg statement/> }.into_any()
        }
        slir::cfg::StatementData::OpOffsetSlice(_) => {
            view! { <OpOffsetSlice cfg statement/> }.into_any()
        }
        slir::cfg::StatementData::OpUnary(_) => view! { <OpUnary cfg statement/> }.into_any(),
        slir::cfg::StatementData::OpBinary(_) => view! { <OpBinary cfg statement/> }.into_any(),
        slir::cfg::StatementData::OpMax(_) => view! { <OpMax cfg statement/> }.into_any(),
        slir::cfg::StatementData::OpMin(_) => view! { <OpMin cfg statement/> }.into_any(),
        slir::cfg::StatementData::OpRound(_) => view! { <OpRound cfg statement/> }.into_any(),
        slir::cfg::StatementData::OpFloor(_) => view! { <OpFloor cfg statement/> }.into_any(),
        slir::cfg::StatementData::OpCeil(_) => view! { <OpCeil cfg statement/> }.into_any(),
        slir::cfg::StatementData::OpCall(_) => view! { <OpCall cfg statement/> }.into_any(),
        slir::cfg::StatementData::OpConvertToU32(_) => {
            view! { <OpConvertToU32 cfg statement/> }.into_any()
        }
        slir::cfg::StatementData::OpConvertToI32(_) => {
            view! { <OpConvertToI32 cfg statement/> }.into_any()
        }
        slir::cfg::StatementData::OpConvertToF32(_) => {
            view! { <OpConvertToF32 cfg statement/> }.into_any()
        }
        slir::cfg::StatementData::OpConvertToBool(_) => {
            view! { <OpConvertToBool cfg statement/> }.into_any()
        }
        slir::cfg::StatementData::OpArrayLength(_) => {
            view! { <OpArrayLength cfg statement/> }.into_any()
        }
    };

    view! {
        <p>{inner}</p>
    }
}

#[component]
pub fn Assign(cfg: StoredValue<Cfg>, statement: slir::cfg::Statement) -> impl IntoView {
    let cfg_value = cfg.read_value();
    let stmt = cfg_value[statement].expect_assign();
    let binding = stmt.local_binding();
    let value = stmt.value();

    view! {
        <Value cfg value=binding.into()/>" = "<Value cfg value/>
    }
}

#[component]
pub fn Bind(cfg: StoredValue<Cfg>, statement: slir::cfg::Statement) -> impl IntoView {
    let cfg_value = cfg.read_value();
    let stmt = cfg_value[statement].expect_bind();
    let binding = stmt.local_binding();
    let value = stmt.value();

    view! {
        <Value cfg value=binding.into()/>" = "<Value cfg value/>
    }
}

#[component]
pub fn Uninitialized(cfg: StoredValue<Cfg>, statement: slir::cfg::Statement) -> impl IntoView {
    let cfg_value = cfg.read_value();
    let stmt = cfg_value[statement].expect_uninitialized();
    let binding = stmt.local_binding();

    view! {
        <Value cfg value=binding.into()/>" = uninitialized"
    }
}

#[component]
pub fn OpAlloca(cfg: StoredValue<Cfg>, statement: slir::cfg::Statement) -> impl IntoView {
    let cfg_value = cfg.read_value();
    let stmt = cfg_value[statement].expect_op_alloca();
    let binding = stmt.result();

    view! {
        <Value cfg value=binding.into()/> " = alloca"
    }
}

#[component]
pub fn OpLoad(cfg: StoredValue<Cfg>, statement: slir::cfg::Statement) -> impl IntoView {
    let cfg_value = cfg.read_value();
    let stmt = cfg_value[statement].expect_op_load();
    let binding = stmt.result();
    let pointer = stmt.ptr();

    view! {
        <Value cfg value=binding.into()/> " = load "<Value cfg value=pointer/>
    }
}

#[component]
pub fn OpStore(cfg: StoredValue<Cfg>, statement: slir::cfg::Statement) -> impl IntoView {
    let cfg_value = cfg.read_value();
    let stmt = cfg_value[statement].expect_op_store();
    let pointer = stmt.ptr();
    let value = stmt.value();

    view! {
        "store "<Value cfg value/>" into "<Value cfg value=pointer/>
    }
}

#[component]
pub fn OpExtractField(cfg: StoredValue<Cfg>, statement: slir::cfg::Statement) -> impl IntoView {
    let cfg_value = cfg.read_value();
    let stmt = cfg_value[statement].expect_op_extract_field();
    let aggregate = stmt.value();
    let field_index = stmt.field_index();
    let binding = stmt.result();

    view! {
        <Value cfg value=binding.into()/>" = "<Value cfg value=aggregate/>"._"{field_index}
    }
}

#[component]
pub fn OpExtractElement(cfg: StoredValue<Cfg>, statement: slir::cfg::Statement) -> impl IntoView {
    let cfg_value = cfg.read_value();
    let stmt = cfg_value[statement].expect_op_extract_element();
    let aggregate = stmt.value();
    let index = stmt.element_index();
    let binding = stmt.result();

    view! {
        <Value cfg value=binding.into()/>" = "<Value cfg value=aggregate/>"["<Value cfg value=index/>"]"
    }
}

#[component]
pub fn OpFieldPtr(cfg: StoredValue<Cfg>, statement: slir::cfg::Statement) -> impl IntoView {
    let cfg_value = cfg.read_value();
    let stmt = cfg_value[statement].expect_op_field_ptr();
    let ptr = stmt.ptr();
    let field_index = stmt.field_index();
    let binding = stmt.result();

    view! {
        <Value cfg value=binding.into()/>" = &"<Value cfg value=ptr/>"._"{field_index}
    }
}

#[component]
pub fn OpElementPtr(cfg: StoredValue<Cfg>, statement: slir::cfg::Statement) -> impl IntoView {
    let cfg_value = cfg.read_value();
    let stmt = cfg_value[statement].expect_op_element_ptr();
    let ptr = stmt.ptr();
    let index = stmt.element_index();
    let binding = stmt.result();

    view! {
        <Value cfg value=binding.into()/>" = &"<Value cfg value=ptr/>"["<Value cfg value=index/>"]"
    }
}

#[component]
pub fn OpVariantPtr(cfg: StoredValue<Cfg>, statement: slir::cfg::Statement) -> impl IntoView {
    let cfg_value = cfg.read_value();
    let stmt = cfg_value[statement].expect_op_variant_ptr();
    let ptr = stmt.ptr();
    let variant_index = stmt.variant_index();
    let binding = stmt.result();

    view! {
        <Value cfg value=binding.into()/>" = variant-ptr "<Value cfg value=ptr/>":"{variant_index}
    }
}

#[component]
pub fn OpGetDiscriminant(cfg: StoredValue<Cfg>, statement: slir::cfg::Statement) -> impl IntoView {
    let cfg_value = cfg.read_value();
    let stmt = cfg_value[statement].expect_op_get_discriminant();
    let ptr = stmt.ptr();
    let binding = stmt.result();

    view! {
        <Value cfg value=binding.into()/>" = get-discriminant "<Value cfg value=ptr/>
    }
}

#[component]
pub fn OpSetDiscriminant(cfg: StoredValue<Cfg>, statement: slir::cfg::Statement) -> impl IntoView {
    let cfg_value = cfg.read_value();
    let stmt = cfg_value[statement].expect_op_set_discriminant();
    let ptr = stmt.ptr();
    let variant_index = stmt.variant_index();

    view! {
        "set-discriminant "{variant_index}" on "<Value cfg value=ptr/>
    }
}

#[component]
pub fn OpOffsetSlice(cfg: StoredValue<Cfg>, statement: slir::cfg::Statement) -> impl IntoView {
    let cfg_value = cfg.read_value();
    let stmt = cfg_value[statement].expect_op_offset_slice_ptr();
    let ptr = stmt.ptr();
    let offset = stmt.offset();
    let binding = stmt.result();

    view! {
        <Value cfg value=binding.into()/>
        " = offset "
        <Value cfg value=ptr/>
        " by "
        <Value cfg value=offset/>
    }
}

#[component]
pub fn OpUnary(cfg: StoredValue<Cfg>, statement: slir::cfg::Statement) -> impl IntoView {
    let cfg_value = cfg.read_value();
    let stmt = cfg_value[statement].expect_op_unary();
    let value = stmt.value();
    let operator = stmt.operator();
    let binding = stmt.result();

    view! {
        <Value cfg value=binding.into()/>{format!(" = {}", operator)}<Value cfg value/>
    }
}

#[component]
pub fn OpBinary(cfg: StoredValue<Cfg>, statement: slir::cfg::Statement) -> impl IntoView {
    let cfg_value = cfg.read_value();
    let stmt = cfg_value[statement].expect_op_binary();
    let operator = stmt.operator();
    let lhs = stmt.lhs();
    let rhs = stmt.rhs();
    let binding = stmt.result();

    view! {
        <Value cfg value=binding.into()/>" = "<Value cfg value=lhs/>{format!(" {} ", operator)}<Value cfg value=rhs/>
    }
}

#[component]
pub fn OpMax(cfg: StoredValue<Cfg>, statement: slir::cfg::Statement) -> impl IntoView {
    let cfg_value = cfg.read_value();
    let stmt = cfg_value[statement].expect_op_max();
    let lhs = stmt.lhs();
    let rhs = stmt.rhs();
    let binding = stmt.result();

    view! {
        <Value cfg value=binding.into()/>" = max("<Value cfg value=lhs/>", "<Value cfg value=rhs/>")"
    }
}

#[component]
pub fn OpMin(cfg: StoredValue<Cfg>, statement: slir::cfg::Statement) -> impl IntoView {
    let cfg_value = cfg.read_value();
    let stmt = cfg_value[statement].expect_op_min();
    let lhs = stmt.lhs();
    let rhs = stmt.rhs();
    let binding = stmt.result();

    view! {
        <Value cfg value=binding.into()/>" = min("<Value cfg value=lhs/>", "<Value cfg value=rhs/>")"
    }
}

#[component]
pub fn OpRound(cfg: StoredValue<Cfg>, statement: slir::cfg::Statement) -> impl IntoView {
    let cfg_value = cfg.read_value();
    let stmt = cfg_value[statement].expect_op_round();
    let value = stmt.value();
    let binding = stmt.result();

    view! {
        <Value cfg value=binding.into()/>" = round("<Value cfg value/>")"
    }
}

#[component]
pub fn OpFloor(cfg: StoredValue<Cfg>, statement: slir::cfg::Statement) -> impl IntoView {
    let cfg_value = cfg.read_value();
    let stmt = cfg_value[statement].expect_op_floor();
    let value = stmt.value();
    let binding = stmt.result();

    view! {
        <Value cfg value=binding.into()/>" = floor("<Value cfg value/>")"
    }
}

#[component]
pub fn OpCeil(cfg: StoredValue<Cfg>, statement: slir::cfg::Statement) -> impl IntoView {
    let cfg_value = cfg.read_value();
    let stmt = cfg_value[statement].expect_op_ceil();
    let value = stmt.value();
    let binding = stmt.result();

    view! {
        <Value cfg value=binding.into()/>" = ceil("<Value cfg value/>")"
    }
}

#[component]
pub fn OpCall(cfg: StoredValue<Cfg>, statement: slir::cfg::Statement) -> impl IntoView {
    let module_data = use_module_data();
    let module_name = module_data.module.read_value().name;
    let cfg_value = cfg.read_value();
    let stmt = cfg_value[statement].expect_op_call();
    let callee = stmt.callee();
    let binding = stmt.maybe_result();

    let mut arg_views = Vec::new();
    let mut is_first = true;

    for arg in stmt.arguments().iter().copied() {
        if !is_first {
            arg_views.push(view! {", "}.into_any());
        }

        arg_views.push(view! { <Value cfg value=arg/> }.into_any());

        is_first = false;
    }

    view! {
        {{move || binding.map(|binding | view! {
            <Value cfg value=binding.into()/>" = "
        })}}

        <a href=function_url(module_name, callee)>
            {callee.name.to_string()}
        </a>
        "("{arg_views}")"
    }
}

#[component]
pub fn OpConvertToU32(cfg: StoredValue<Cfg>, statement: slir::cfg::Statement) -> impl IntoView {
    let cfg_value = cfg.read_value();
    let stmt = cfg_value[statement].expect_op_convert_to_u32();
    let value = stmt.value();
    let binding = stmt.result();

    view! {
        <Value cfg value=binding.into()/>" = u32("<Value cfg value/>")"
    }
}

#[component]
pub fn OpConvertToI32(cfg: StoredValue<Cfg>, statement: slir::cfg::Statement) -> impl IntoView {
    let cfg_value = cfg.read_value();
    let stmt = cfg_value[statement].expect_op_convert_to_i32();
    let value = stmt.value();
    let binding = stmt.result();

    view! {
        <Value cfg value=binding.into()/>" = i32("<Value cfg value/>")"
    }
}

#[component]
pub fn OpConvertToF32(cfg: StoredValue<Cfg>, statement: slir::cfg::Statement) -> impl IntoView {
    let cfg_value = cfg.read_value();
    let stmt = cfg_value[statement].expect_op_convert_to_f32();
    let value = stmt.value();
    let binding = stmt.result();

    view! {
        <Value cfg value=binding.into()/>" = f32("<Value cfg value/>")"
    }
}

#[component]
pub fn OpConvertToBool(cfg: StoredValue<Cfg>, statement: slir::cfg::Statement) -> impl IntoView {
    let cfg_value = cfg.read_value();
    let stmt = cfg_value[statement].expect_op_convert_to_bool();
    let value = stmt.value();
    let binding = stmt.result();

    view! {
        <Value cfg value=binding.into()/>" = bool("<Value cfg value/>")"
    }
}

#[component]
pub fn OpArrayLength(cfg: StoredValue<Cfg>, statement: slir::cfg::Statement) -> impl IntoView {
    let cfg_value = cfg.read_value();
    let stmt = cfg_value[statement].expect_op_array_length();
    let value = stmt.ptr();
    let binding = stmt.result();

    view! {
        <Value cfg value=binding.into()/>" = array-length("<Value cfg value/>")"
    }
}
