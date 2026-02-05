use leptos::prelude::*;
use slir::ty::TypeKind;
use thaw::*;
use urlencoding::encode as urlencode;

use crate::module::use_module_data;

#[component]
pub fn Type(ty: slir::ty::Type) -> impl IntoView {
    let m = use_module_data();

    match &*m.module.read_value().ty.kind(ty) {
        TypeKind::Scalar(s) => view! {{s.to_string()}}.into_any(),
        TypeKind::Atomic(s) => view! {{format!("atomic<{}>", s)}}.into_any(),
        TypeKind::Vector(v) => view! {{v.to_string()}}.into_any(),
        TypeKind::Matrix(m) => view! {{m.to_string()}}.into_any(),
        TypeKind::Array {
            element_ty, count, ..
        } => view! { "array<" <Type ty=*element_ty/> ", " {*count} ">" }.into_any(),
        TypeKind::Slice { element_ty, .. } => {
            view! { "array<" <Type ty=*element_ty/> ">" }.into_any()
        }
        TypeKind::Struct(_) => {
            let id = ty.registration_id().unwrap_or_default();

            view! {
                <Link href=format!("/{}/adts/{}", urlencode(m.module.read_value().name.as_str()), id)>
                    {format!("S_{}", id)}
                </Link>
            }.into_any()
        }
        TypeKind::Enum(_) => {
            let id = ty.registration_id().unwrap_or_default();

            view! {
                <Link href=format!("/{}/adts/{}", urlencode(m.module.read_value().name.as_str()), id)>
                    {format!("E_{}", id)}
                </Link>
            }.into_any()
        }
        TypeKind::Ptr(pointee_ty) => view! {"ptr<" <Type ty=*pointee_ty/> ">"}.into_any(),
        TypeKind::Function(_) => view! {"fn"}.into_any(),
        TypeKind::Predicate => view! {"predicate"}.into_any(),
        TypeKind::Dummy => view! {"dummy"}.into_any(),
    }
}
