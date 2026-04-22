use leptos::prelude::*;
use slir::cfg::Cfg;
use slotmap::Key;

use crate::module::functions::detail::cfg_explorer::statement::Statement;
use crate::module::functions::detail::cfg_explorer::terminator::Terminator;

#[component]
pub fn BasicBlock(cfg: StoredValue<Cfg>, bb: slir::cfg::BasicBlock) -> impl IntoView {
    view! {
        <div class="basic-block">
            <div class="basic-block-header" id=format!("BB{}", bb.data().as_ffi())>
                {format!("BB{}", bb.data().as_ffi())}
            </div>
            <div class="basic-block-body">
            {move || {
                cfg.read_value()[bb].statements().iter().copied().map(|statement| {
                    view! { <Statement cfg statement/> }
                }).collect_view()
            }}
            </div>
            <div class="basic-block-terminator">
            {move || {
                let terminator = cfg.read_value()[bb].terminator().clone();

                view! { <Terminator cfg terminator/> }
            }}
            </div>
        </div>
    }
}
