use leptos::prelude::*;
use leptos::{component, IntoView};
use slotmap::Key;
use thaw::*;

use crate::module::module::use_module_data;
use crate::module::uniform_bindings::uniform_binding_name;

#[component]
pub fn List() -> impl IntoView {
    let module_data = use_module_data();

    view! {
        <div class="info-page-container">
            <h1>"Uniform Bindings"</h1>

            <ul>
                {move || {
                    module_data.module.read_value().uniform_bindings.keys().map(|b| view! {
                        <li>
                            <Link href=format!("uniform_bindings/{}", b.data().as_ffi())>
                                {uniform_binding_name(b)}
                            </Link>
                        </li>
                    }).collect_view()
                }}
            </ul>
        </div>
    }
}
