use leptos::prelude::*;
use slotmap::Key;

use crate::module::functions::detail::cfg_explorer::statement::Statement;
use crate::module::functions::detail::cfg_explorer::terminator::Terminator;
use crate::module::use_module_data;

#[component]
pub fn BasicBlock(bb: slir::cfg::BasicBlock) -> impl IntoView {
    let module_data = use_module_data();

    view! {
        <div class="basic-block">
            <div class="basic-block-header" id=format!("BB{}", bb.data().as_ffi())>
                {format!("BB{}", bb.data().as_ffi())}
            </div>
            <div class="basic-block-body">
            {move || {
                module_data.cfg.read_value()[bb].statements().iter().copied().map(|statement| {
                    view! { <Statement statement/> }
                }).collect_view()
            }}
            </div>
            <div class="basic-block-terminator">
            {move || {
                let terminator = module_data.cfg.read_value()[bb].terminator().clone();

                view! { <Terminator terminator/> }
            }}
            </div>
        </div>
    }
}
