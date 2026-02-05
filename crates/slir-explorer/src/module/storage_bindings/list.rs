use leptos::prelude::*;
use leptos::{component, IntoView};
use slotmap::Key;
use thaw::*;

use crate::module::module::use_module_data;
use crate::module::storage_bindings::storage_binding_name;

#[component]
pub fn List() -> impl IntoView {
    let module_data = use_module_data();

    view! {
        <div class="info-page-container">
            <h1>"Storage Bindings"</h1>

            <ul>
                {move || {
                    module_data.module.read_value().storage_bindings.keys().map(|b| view! {
                        <li>
                            <Link href=format!("storage_bindings/{}", b.data().as_ffi())>
                                {storage_binding_name(b)}
                            </Link>
                        </li>
                    }).collect_view()
                }}
            </ul>
        </div>
    }
}
