use leptos::prelude::*;
use leptos::{component, IntoView, Params};
use leptos_router::hooks::use_params;
use leptos_router::params::Params;
use slir::StorageBinding;
use slotmap::KeyData;

use crate::module::module::use_module_data;
use crate::module::storage_bindings::storage_binding_name;
use crate::module::ty::Type;

#[derive(Params, PartialEq)]
struct StorageBindingParams {
    storage_binding_id: Option<u64>,
}

#[component]
pub fn Detail() -> impl IntoView {
    let module_data = use_module_data();
    let params = use_params::<StorageBindingParams>();
    let id = move || {
        params
            .read()
            .as_ref()
            .ok()
            .and_then(|params| params.storage_binding_id)
            .unwrap_or_default()
    };
    let binding = move || {
        let binding = StorageBinding::from(KeyData::from_ffi(id()));

        module_data
            .module
            .read_value()
            .storage_bindings
            .contains(binding)
            .then_some(binding)
    };

    view! {
        {move || {
            match binding() {
                Some(binding) => view! {
                    <StorageBindingInfo binding/>
                }.into_any(),
                None => view! {
                    <div class="info-page-container">
                        <h1>"Storage Binding Not Found"</h1>

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
    group: u32,
    binding: u32,
}

#[component]
fn StorageBindingInfo(binding: StorageBinding) -> impl IntoView {
    let model = Memo::new(move |_| {
        let module_data = use_module_data();
        let binding_data = &module_data.module.read_value().storage_bindings[binding];

        Model {
            ty: binding_data.ty,
            group: binding_data.resource_binding.group,
            binding: binding_data.resource_binding.binding,
        }
    });

    view! {
        <div class="info-page-container">
            <h1>Storage Binding {storage_binding_name(binding)}</h1>

            <dl>
                <dt>Type</dt>
                <dd><Type ty=model.read().ty/></dd>

                <dt>Group</dt>
                <dd>{model.read().group}</dd>

                <dt>Binding</dt>
                <dd>{model.read().binding}</dd>
            </dl>
        </div>
    }
}
