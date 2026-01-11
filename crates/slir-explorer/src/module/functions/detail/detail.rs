use leptos::prelude::*;
use leptos::Params;
use leptos_router::hooks::{use_params, use_query};
use leptos_router::params::Params;
use slir::{Function, Symbol};
use urlencoding::decode as urldecode;

use crate::module::functions::detail::cfg_explorer::CfgExplorer;
use crate::module::functions::detail::rvsdg_explorer::{RvsdgExplorer, RvsdgStage};
use crate::module::functions::detail::scf_explorer::ScfExplorer;
use crate::module::module::use_module_data;
use crate::module::ty::Type;

const MODE_CFG: &'static str = "cfg";
const MODE_RVSDG_INITIAL: &'static str = "rvsdg-initial";
const MODE_RVSDG_TRANSFORMED: &'static str = "rvsdg-transformed";
const MODE_SCF: &'static str = "scf";

pub fn use_function() -> Function {
    use_context::<Function>().expect("can only be used inside a function-detail context")
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    Cfg,
    RvsdgInitial,
    RvsdgTransformed,
    Scf,
}

impl Mode {
    fn from_str(str: &str) -> Self {
        match str {
            MODE_RVSDG_INITIAL => Mode::RvsdgInitial,
            MODE_RVSDG_TRANSFORMED => Mode::RvsdgTransformed,
            MODE_SCF => Mode::Scf,
            _ => Mode::Cfg,
        }
    }
}

#[derive(Params, PartialEq)]
struct FunctionParams {
    function_name: Option<String>,
}

#[derive(Params, PartialEq)]
struct FunctionQuery {
    mode: Option<String>,
}

#[component]
pub fn Detail() -> impl IntoView {
    let module_data = use_module_data();
    let params = use_params::<FunctionParams>();
    let query = use_query::<FunctionQuery>();

    let function_string = move || {
        let param = params
            .read()
            .as_ref()
            .ok()
            .and_then(|params| params.function_name.clone())
            .unwrap_or_default();
        let decoded = urldecode(&param).unwrap_or_default();

        decoded.to_string()
    };

    let function = move || {
        let function_string = function_string();
        let (module, name) = function_string.split_once("--")?;

        let function = Function {
            module: Symbol::from_ref(module),
            name: Symbol::from_ref(name),
        };

        module_data
            .read_value()
            .module
            .fn_sigs
            .contains(function)
            .then_some(function)
    };

    let mode = move || {
        let mode_label = query
            .read()
            .as_ref()
            .ok()
            .and_then(|params| params.mode.clone())
            .unwrap_or_default();

        Mode::from_str(&mode_label)
    };

    view! {
        {move || {
            match function() {
                Some(function) => {
                    provide_context(function);

                    view! {
                        <div class="function-explorer-container">
                            <FnSig function/>
                            <ModeList selected_mode=mode()/>

                            <div class="function-body">
                                {move || {
                                    match mode() {
                                        Mode::Cfg => view! {
                                            <CfgExplorer/>
                                        }.into_any(),
                                        Mode::RvsdgInitial => view! {
                                            <RvsdgExplorer stage=RvsdgStage::Initial/>
                                        }.into_any(),
                                        Mode::RvsdgTransformed => view! {
                                            <RvsdgExplorer stage=RvsdgStage::Transformed/>
                                        }.into_any(),
                                        Mode::Scf => view! {
                                            <ScfExplorer/>
                                        }.into_any(),
                                    }
                                }}
                            </div>
                        </div>
                    }.into_any()
                }
                None => view! {
                    <h1>Not Found</h1>

                    <p>
                        {format!(
                            "No function `{}` found in the current module.",
                            function_string()
                        )}
                    </p>
                }.into_any()
            }
        }}
    }
}

#[component]
fn FnSig(function: Function) -> impl IntoView {
    let module_data = use_module_data();

    view! {
        <div class="function-sig">
            {function.name.to_string()}
            "("
            <span> // Without this span around the args, Leptos produces a hydration error...
                {move || {
                    let mut arg_views = Vec::new();
                    let mut is_first = true;

                    for arg in &module_data.read_value().module.fn_sigs[function].args {
                        if !is_first {
                            arg_views.push(view! {", "}.into_any());
                        }

                        arg_views.push(view!{<Type ty=arg.ty/>}.into_any());

                        is_first = false;
                    }

                    arg_views
                }}
            </span>
            ")"
            {move || {
                module_data.read_value().module.fn_sigs[function].ret_ty.map(|ty| {
                    view! {
                        " -> " <Type ty/>
                    }
                })
            }}
        </div>
    }
}

#[component]
fn ModeList(selected_mode: Mode) -> impl IntoView {
    view! {
        <div class="function-mode-list">
            <a href=format!("?mode={}", MODE_CFG)
                class="function-mode"
                class:active=move || selected_mode == Mode::Cfg
            >
                "CFG"
            </a>
            <a href=format!("?mode={}", MODE_RVSDG_INITIAL)
                class="function-mode"
                class:active=move || selected_mode == Mode::RvsdgInitial
            >
                "RVSDG-initial"
            </a>
            <a href=format!("?mode={}", MODE_RVSDG_TRANSFORMED)
                class="function-mode"
                class:active=move || selected_mode == Mode::RvsdgTransformed
            >
                "RVSDG-transformed"
            </a>
            <a href=format!("?mode={}", MODE_SCF)
                class="function-mode"
                class:active=move || selected_mode == Mode::Scf
            >
                "SCF"
            </a>
        </div>
    }
}
