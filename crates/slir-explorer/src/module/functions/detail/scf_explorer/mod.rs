use leptos::prelude::*;
use leptos::{component, view, IntoView};

use crate::module::functions::detail::detail::use_function;
use crate::module::functions::detail::scf_explorer::local_binding::LocalBinding;
use crate::module::functions::detail::scf_explorer::statement::Statement;
use crate::module::use_module_data;

pub mod block;
pub mod expression;
pub mod local_binding;
pub mod statement;

type HighlightSignal = (
    ReadSignal<Option<slir::scf::LocalBinding>>,
    WriteSignal<Option<slir::scf::LocalBinding>>,
);

#[component]
pub fn ScfExplorer() -> impl IntoView {
    let module_data = use_module_data();
    let function = use_function();

    let highlight: HighlightSignal = signal(None);

    provide_context(highlight);

    let body = module_data
        .scf
        .and_then(|scf| scf.read_value().get_function_body(function).cloned());

    if let Some(body) = body {
        let body_block = body.block();

        view! {
            <div class="scf-argument-bindings">
                <h3>Argument Bindings</h3>
                <ul>
                    {move || {
                        body.argument_bindings().iter().map(|binding| view! {
                            <li><LocalBinding binding=*binding/></li>
                        }).collect_view()
                    }}
                </ul>
            </div>
            <div>
                <h3>Function Body</h3>

                <div class="scf-function-body">
                    {move || {
                        let scf = module_data.expect_scf().read_value();
                        let block = &scf[body_block];

                        block.statements().iter().map(|stmt| {
                            view! { <Statement statement=*stmt/> }
                        }).collect_view()
                    }}
                </div>
            </div>
        }
        .into_any()
    } else {
        view! {
            <div class="info-page-container">
                <p>"No SCF for the current function."</p>
            </div>
        }
        .into_any()
    }
}
