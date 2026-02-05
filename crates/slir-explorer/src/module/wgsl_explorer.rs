use leptos::prelude::*;

use crate::module::use_module_data;

#[component]
pub fn WgslExplorer() -> impl IntoView {
    let module_data = use_module_data();

    view! {
        {{move || {
            match module_data.wgsl {
                Some(wgsl) => view! {
                    <div class="wgsl-container">
                        {wgsl.read_value().as_str()}
                    </div>
                }.into_any(),
                None => view! {
                    <div class="info-page-container">
                        <h1>"WGSL Not Found"</h1>

                        <p>"No WGSL file exists in this module's SLIR artifact."</p>
                    </div>
                }.into_any()
            }
        }}}
    }
}
