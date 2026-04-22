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

#[derive(Copy, Clone)]
pub enum CfgStage {
    Initial,
    Structurized,
}

#[component]
pub fn CfgExplorer(stage: CfgStage) -> impl IntoView {
    let module_data = use_module_data();
    let function = use_function();
    let highlight_signal: HighlightSignal = signal(None);
    let cfg = match stage {
        CfgStage::Initial => Some(module_data.cfg_initial),
        CfgStage::Structurized => module_data.cfg_structurized,
    };

    provide_context(highlight_signal);

    view! {
        {move || {
            if let Some(cfg) = cfg {
                view! {
                    <div class="params">
                        <div class="param-list-header">
                            Function Parameters
                        </div>
                        <ul class="param-list">
                            {move || {
                                cfg.read_value()[function].argument_values().iter().map(|p| view! {
                                    <li><Value cfg value=(*p).into()/></li>
                                }).collect_view()
                            }}
                        </ul>
                    </div>

                    {move || {
                        cfg.read_value()[function].basic_blocks().iter().copied().map(|bb| view! {
                            <BasicBlock cfg bb/>
                        }).collect_view()
                    }}
                }.into_any()
            } else {
                view! {
                    <div class="info-page-container">
                        <p>"No CFG for the current function."</p>
                    </div>
                }.into_any()
            }
        }}
    }
}
