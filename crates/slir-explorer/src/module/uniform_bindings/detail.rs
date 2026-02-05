use leptos::prelude::*;
use leptos::{component, IntoView, Params};
use leptos_router::hooks::use_params;
use leptos_router::params::Params;
use slir::UniformBinding;
use slotmap::KeyData;

use crate::module::module::use_module_data;
use crate::module::ty::Type;
use crate::module::uniform_bindings::uniform_binding_name;

#[derive(Params, PartialEq)]
struct UniformBindingParams {
    uniform_binding_id: Option<u64>,
}

#[component]
pub fn Detail() -> impl IntoView {
    let module_data = use_module_data();
    let params = use_params::<UniformBindingParams>();
    let id = move || {
        params
            .read()
            .as_ref()
            .ok()
            .and_then(|params| params.uniform_binding_id)
            .unwrap_or_default()
    };
    let binding = move || {
        let binding = UniformBinding::from(KeyData::from_ffi(id()));

        module_data
            .module
            .read_value()
            .uniform_bindings
            .contains(binding)
            .then_some(binding)
    };

    view! {
        {move || {
            match binding() {
                Some(binding) => view! {
                    <UniformBindingInfo binding/>
                }.into_any(),
                None => view! {
                    <div class="info-page-container">
                        <h1>"Uniform Binding Not Found"</h1>

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
fn UniformBindingInfo(binding: UniformBinding) -> impl IntoView {
    let model = Memo::new(move |_| {
        let module_data = use_module_data();
        let binding_data = &module_data.module.read_value().uniform_bindings[binding];

        Model {
            ty: binding_data.ty,
            group: binding_data.resource_binding.group,
            binding: binding_data.resource_binding.binding,
        }
    });

    view! {
        <div class="info-page-container">
            <h1>Uniform Binding {uniform_binding_name(binding)}</h1>

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
