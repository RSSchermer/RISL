use leptos::prelude::*;
use leptos::{component, IntoView, Params};
use leptos_router::hooks::use_params;
use leptos_router::params::Params;
use slir::WorkgroupBinding;
use slotmap::KeyData;

use crate::module::module::use_module_data;
use crate::module::ty::Type;
use crate::module::workgroup_bindings::workgroup_binding_name;

#[derive(Params, PartialEq)]
struct WorkgroupBindingParams {
    workgroup_binding_id: Option<u64>,
}

#[component]
pub fn Detail() -> impl IntoView {
    let module_data = use_module_data();
    let params = use_params::<WorkgroupBindingParams>();
    let id = move || {
        params
            .read()
            .as_ref()
            .ok()
            .and_then(|params| params.workgroup_binding_id)
            .unwrap_or_default()
    };
    let binding = move || {
        let binding = WorkgroupBinding::from(KeyData::from_ffi(id()));

        module_data
            .module
            .read_value()
            .workgroup_bindings
            .contains(binding)
            .then_some(binding)
    };

    view! {
        {move || {
            match binding() {
                Some(binding) => view! {
                    <WorkgroupBindingInfo binding/>
                }.into_any(),
                None => view! {
                    <div class="info-page-container">
                        <h1>"Workgroup Binding Not Found"</h1>

                        <p>"No binding with the given identifier found in the current module."</p>
                    </div>
                }.into_any(),
            }
        }}
    }
}

#[derive(PartialEq)]
struct Model {
    ty: slir::ty::Type,
}

#[component]
fn WorkgroupBindingInfo(binding: WorkgroupBinding) -> impl IntoView {
    let model = Memo::new(move |_| {
        let module_data = use_module_data();
        let binding_data = &module_data.module.read_value().workgroup_bindings[binding];

        Model {
            ty: binding_data.ty,
        }
    });

    view! {
        <div class="info-page-container">
            <h1>Workgroup Binding {workgroup_binding_name(binding)}</h1>

            <dl>
                <dt>Type</dt>
                <dd><Type ty=model.read().ty/></dd>
            </dl>
        </div>
    }
}
