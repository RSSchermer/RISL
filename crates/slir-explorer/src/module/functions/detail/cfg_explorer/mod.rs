pub mod basic_block;
pub mod statement;
pub mod terminator;
pub mod value;

use leptos::prelude::*;

use crate::module::functions::detail::cfg_explorer::basic_block::BasicBlock;
use crate::module::functions::detail::cfg_explorer::value::Value;
use crate::module::functions::detail::detail::use_function;
use crate::module::module::use_module_data;

type HighlightSignal = (
    ReadSignal<Option<slir::cfg::LocalBinding>>,
    WriteSignal<Option<slir::cfg::LocalBinding>>,
);

#[component]
pub fn CfgExplorer() -> impl IntoView {
    let module_data = use_module_data();
    let function = use_function();
    let highlight_signal: HighlightSignal = signal(None);

    provide_context(highlight_signal);

    view! {
        <div class="params">
            <div class="param-list-header">
                Function Parameters
            </div>
            <ul class="param-list">
                {move || {
                    module_data.cfg.read_value()[function].argument_values().iter().map(|p| view! {
                        <li><Value value=(*p).into()/></li>
                    }).collect_view()
                }}
            </ul>
        </div>

        {move || {
            module_data.cfg.read_value()[function].basic_blocks().iter().copied().map(|bb| view! {
                <BasicBlock bb/>
            }).collect_view()
        }}
    }
}
