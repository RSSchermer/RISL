use leptos::prelude::*;
use leptos::Params;
use leptos_router::hooks::use_params;
use leptos_router::params::Params;
use slir::ty::TypeKind;
use thaw::*;

use crate::module::module::use_module_data;
use crate::module::ty::Type;

#[derive(Params, PartialEq)]
struct AdtParams {
    ty_id: Option<usize>,
}

#[component]
pub fn Adt() -> impl IntoView {
    let module_data = use_module_data();
    let params = use_params::<AdtParams>();
    let ty = move || {
        let ty_id = params
            .read()
            .as_ref()
            .ok()
            .and_then(|params| params.ty_id)
            .unwrap_or_default();
        slir::ty::Type::from_registration_id(ty_id)
    };

    view! {
        {move || {
            match *module_data.read_value().module.ty.kind(ty()) {
                TypeKind::Struct(_) => view! {
                    <Struct ty=ty()/>
                }.into_any(),
                TypeKind::Enum(_) => view! {
                    <Enum ty=ty()/>
                }.into_any(),
                _ => view! {
                    <p>"Type is not an ADT"</p>
                }.into_any(),
            }
        }}
    }
}

#[component]
pub fn Struct(ty: slir::ty::Type) -> impl IntoView {
    let id = ty.registration_id().unwrap_or_default();
    let module_data = use_module_data();

    view! {
        <div class="info-page-container">
            <h1>{format!("Struct S_{}", id)}</h1>

            <h2>"Fields:"</h2>

            <ul>
                {module_data.read_value().module.ty.kind(ty).expect_struct().fields.iter().map(|f| {
                    view! {
                        <li>
                            <Type ty=f.ty/>
                        </li>
                    }
                }).collect_view()}
            </ul>
        </div>
    }
}

#[component]
pub fn Enum(ty: slir::ty::Type) -> impl IntoView {
    let id = ty.registration_id().unwrap_or_default();
    let module_data = use_module_data();

    view! {
        <div class="info-page-container">
            <h1>{format!("Enum E_{}", id)}</h1>

            <h2>"Variants:"</h2>

            <ul>
                {module_data.read_value().module.ty.kind(ty).expect_enum().variants.iter().map(|s| {
                    let s = s.registration_id().unwrap_or_default();

                    view! {
                        <li>
                            <Link href=s.to_string()>
                                {format!("S_{}", s)}
                            </Link>
                        </li>
                    }
                }).collect_view()}
            </ul>
        </div>
    }
}
