use leptos::prelude::*;
use leptos::{component, view, IntoView};
use slotmap::Key;
use thaw::{Popover, PopoverPosition, PopoverTrigger};

use crate::module::functions::detail::scf_explorer::HighlightSignal;
use crate::module::ty::Type;
use crate::module::use_module_data;

#[component]
pub fn LocalBinding(binding: slir::scf::LocalBinding) -> impl IntoView {
    let module_data = use_module_data();
    let ty = module_data.expect_scf().read_value()[binding].ty();
    let (get_highlight, set_highlight) =
        use_context::<HighlightSignal>().expect("can only be used inside the SCF explorer");

    let is_highlighted = move || get_highlight.get() == Some(binding);

    let update_highlight = move |_| {
        set_highlight.update(|v| {
            if *v == Some(binding) {
                *v = None;
            } else {
                *v = Some(binding);
            }
        })
    };

    view! {
        <Popover position=PopoverPosition::Bottom>
            <PopoverTrigger slot>
                <span class="scf-local-binding" on:click=update_highlight class:highlighted=is_highlighted>
                    {format!("L{}", binding.data().as_ffi())}
                </span>
            </PopoverTrigger>

            <Type ty/>
        </Popover>
    }
}
