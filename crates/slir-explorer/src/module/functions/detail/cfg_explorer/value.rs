use leptos::prelude::*;
use slir::cfg::Value;
use slotmap::Key;
use thaw::{Popover, PopoverPosition, PopoverTrigger};

use crate::module::functions::detail::cfg_explorer::HighlightSignal;
use crate::module::module::use_module_data;
use crate::module::ty::Type;

#[component]
pub fn Value(value: slir::cfg::Value) -> impl IntoView {
    match value {
        Value::Local(value) => view! {
            <LocalValue value=value/>
        }
        .into_any(),
        Value::InlineConst(value) => view! {
            <InlineConst value=value/>
        }
        .into_any(),
    }
}

#[component]
fn LocalValue(value: slir::cfg::LocalBinding) -> impl IntoView {
    let module_data = use_module_data();
    let highlight_signal =
        use_context::<HighlightSignal>().expect("can only be used inside the cfg-explorer");
    let (get_highlight, set_highlight) = highlight_signal;

    let is_highlighted_value = move || get_highlight.get() == Some(value);

    let update_highlighted_value = move |_| {
        set_highlight.update(|v| {
            if *v == Some(value) {
                *v = None;
            } else {
                *v = Some(value);
            }
        })
    };

    view! {
        <Popover position=PopoverPosition::Bottom>
            <PopoverTrigger slot>
                <span class="cfg-value" on:click=update_highlighted_value class:highlighted=is_highlighted_value>
                    {format!("V{}", value.data().as_ffi())}
                </span>
            </PopoverTrigger>

            <Type ty=module_data.cfg.read_value()[value].ty()/>
        </Popover>
    }
}

#[component]
fn InlineConst(value: slir::cfg::InlineConst) -> impl IntoView {
    match value {
        slir::cfg::InlineConst::U32(v) => view! {{format!("{}u32", v)}}.into_any(),
        slir::cfg::InlineConst::I32(v) => view! {{format!("{}i32", v)}}.into_any(),
        slir::cfg::InlineConst::F32(v) => view! {{format!("{}f32", v)}}.into_any(),
        slir::cfg::InlineConst::Bool(v) => view! {{format!("{}", v)}}.into_any(),
        slir::cfg::InlineConst::Ptr(v) => view! {
            "&"<RootIdentifier root_identifier=v.root_identifier()/>
        }
        .into_any(),
    }
}

#[component]
fn RootIdentifier(root_identifier: slir::cfg::RootIdentifier) -> impl IntoView {
    match root_identifier {
        slir::cfg::RootIdentifier::Local(v) => view! { <Value value=v.into()/> }.into_any(),
        slir::cfg::RootIdentifier::Uniform(v) => {
            view! {{format!("U{}", v.data().as_ffi())}}.into_any()
        }
        slir::cfg::RootIdentifier::Storage(v) => {
            view! {{format!("S{}", v.data().as_ffi())}}.into_any()
        }
        slir::cfg::RootIdentifier::Workgroup(v) => {
            view! {{format!("W{}", v.data().as_ffi())}}.into_any()
        }
        slir::cfg::RootIdentifier::Constant(c) => view! {{c.name.to_string()}}.into_any(),
    }
}
