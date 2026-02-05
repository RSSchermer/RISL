use leptos::prelude::*;
use leptos::{component, IntoView};

use crate::module::functions::detail::scf_explorer::local_binding::LocalBinding;
use crate::module::functions::detail::scf_explorer::statement::Statement;
use crate::module::use_module_data;

#[component]
pub fn Block(block: slir::scf::Block) -> impl IntoView {
    let module_data = use_module_data();

    view! {
        <div class="scf-block scf-indent">
            {move || {
                let scf = module_data.expect_scf().read_value();
                let block = &scf[block];

                block.statements().iter().map(|stmt| {
                    view! { <Statement statement=*stmt/> }
                }).collect_view()
            }}
            {move || {
                let scf = module_data.expect_scf().read_value();
                let block = &scf[block];

                block.control_flow_var_iter().map(|(var, value)| {
                    view! {
                        <LocalBinding binding=var/>
                        " = "<LocalBinding binding=value/>";"
                    }
                }).collect_view()
            }}
        </div>
    }
}
