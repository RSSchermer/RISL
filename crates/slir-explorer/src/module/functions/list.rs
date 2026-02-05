use leptos::prelude::*;
use leptos::{component, IntoView};
use thaw::*;
use urlencoding::encode as urlencode;

use crate::module::module::use_module_data;

#[component]
pub fn List() -> impl IntoView {
    let module_data = use_module_data();

    let functions = move || {
        let mut functions = module_data
            .module
            .read_value()
            .fn_sigs
            .keys()
            .collect::<Vec<_>>();

        functions.sort_by(|a, b| a.name.cmp(&b.name));

        functions
    };

    view! {
        <div class="info-page-container">
            <h1>"Functions"</h1>

            <ul>
                {move || {
                    functions().into_iter().map(|f| view! {
                        <li>
                            <Link href=format!("functions/{}", urlencode(f.name.as_str()))>
                                {f.name.to_string()}
                            </Link>
                        </li>
                    }).collect_view()
                }}
            </ul>
        </div>
    }
}
