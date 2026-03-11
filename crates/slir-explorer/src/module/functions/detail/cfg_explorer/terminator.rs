use leptos::prelude::*;
use slir::cfg::BranchSelector;
use slotmap::Key;

use crate::module::functions::detail::cfg_explorer::value::Value;

#[component]
pub fn Terminator(terminator: slir::cfg::Terminator) -> impl IntoView {
    match terminator {
        slir::cfg::Terminator::Branch(branch) => {
            let targets = branch.targets().iter().copied().collect::<Vec<_>>();
            let targets = view! {
                    {move || {
                        let mut bb_views = Vec::new();
                        let mut is_first = true;

                        for bb in targets.iter().copied() {
                            if !is_first {
                                bb_views.push(view! {", "}.into_any());
                            }

                            bb_views.push(view!{
                                <a href=format!("#BB{}", bb.data().as_ffi())>
                                    {format!("BB{}", bb.data().as_ffi())}
                                </a>
                            }.into_any());

                            is_first = false;
                        }

                        bb_views
                    }}
            };

            match branch.selector() {
                BranchSelector::Single => view! { "branch " {targets}}.into_any(),
                BranchSelector::Bool(value) => view! {
                    "branch_bool("<Value value=value.into()/>")" {targets}
                }
                .into_any(),
                BranchSelector::Case { value, cases } => {
                    let cases = cases
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(", ");

                    view! {
                        "branch_case("<Value value=value.into()/>" : [" {cases} "])" {targets}
                    }
                    .into_any()
                }
                BranchSelector::U32(value) => view! {
                    "branch_u32("<Value value=value.into()/>")" {targets}
                }
                .into_any(),
            }
        }
        slir::cfg::Terminator::Return(None) => view! {
            "return"
        }
        .into_any(),
        slir::cfg::Terminator::Return(Some(value)) => view! {
            "return "<Value value/>
        }
        .into_any(),
        slir::cfg::Terminator::Unreachable => view! {
            "unreachable"
        }
        .into_any(),
    }
}
