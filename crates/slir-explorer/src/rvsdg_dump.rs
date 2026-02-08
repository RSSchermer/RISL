use gloo_file::futures::read_as_bytes;
use gloo_file::File;
use leptos::logging;
use leptos::prelude::*;
use leptos::web_sys::HtmlInputElement;
use slir::rvsdg::Rvsdg;
use slir::Function;
use thaw::*;
use wasm_bindgen_futures::spawn_local;

use crate::module::functions::detail::rvsdg_explorer::RvsdgRegionViewer;

#[component]
pub fn RvsdgDumpView() -> impl IntoView {
    let rvsdg = RwSignal::new(None::<StoredValue<Rvsdg>>);
    let selected_function = RwSignal::new(None::<Function>);

    let on_file_change = move |ev| {
        let target = event_target::<HtmlInputElement>(&ev);

        if let Some(files) = target.files() {
            if let Some(file) = files.get(0) {
                let file = File::from(file);

                spawn_local(async move {
                    let res = read_as_bytes(&file).await;

                    if let Ok(bytes) = res {
                        let decoded: Result<(Rvsdg, usize), _> =
                            bincode::serde::decode_from_slice(&bytes, bincode::config::standard());

                        match decoded {
                            Ok((decoded_rvsdg, _)) => {
                                rvsdg.set(Some(StoredValue::new(decoded_rvsdg)));
                                selected_function.set(None);
                            }
                            Err(err) => {
                                logging::error!("Failed to decode RVSDG dump: {}", err);
                            }
                        }
                    }
                });
            }
        }
    };

    let functions = move || {
        rvsdg.get().map(|rvsdg| {
            let mut fns = rvsdg
                .read_value()
                .registered_functions()
                .map(|(f, _)| f)
                .collect::<Vec<_>>();

            fns.sort_by(|a, b| a.name.cmp(&b.name));

            fns
        })
    };

    let selected_region = move || {
        if let (Some(rvsdg), Some(func)) = (rvsdg.get(), selected_function.get()) {
            let rvsdg = rvsdg.read_value();

            if let Some(node) = rvsdg.get_function_node(func) {
                return Some(rvsdg[node].expect_function().body_region());
            }
        }

        None
    };

    view! {
        <Layout has_sider=true class="rvsdg-dump-viewer-container">
            <LayoutSider class="rvsdg-dump-sider">
                <div class="rvsdg-dump-file-input-container">
                    <h3>"RVSDG Dump"</h3>

                    <input type="file" on:change=on_file_change />
                </div>

                <div class="function-list">
                    <h4>"Functions"</h4>

                    <ul>
                        {move || {
                            functions().map(|fns: Vec<Function>| {
                                fns.into_iter().map(|f| {
                                    let name = f.name.to_string();
                                    let is_selected = move || selected_function.get() == Some(f);

                                    view! {
                                        <li
                                            class:selected=is_selected
                                            class="rvsdg-dump-function-item"
                                            on:click=move |_| selected_function.set(Some(f))
                                        >
                                            {name}
                                        </li>
                                    }
                                }).collect_view()
                            })
                        }}
                    </ul>
                </div>
            </LayoutSider>

            <Layout class="rvsdg-dump-content-layout">
                <div class="rvsdg-dump-content-container">
                    {move || {
                        if let (Some(rvsdg), Some(region)) = (rvsdg.get(), selected_region()) {
                            view! {
                                <RvsdgRegionViewer
                                    rvsdg
                                    region
                                    gen_function_url=move |f| {
                                        format!("#{}", f.name.as_str())
                                    }
                                />
                            }.into_any()
                        } else {
                            view! {
                                <div class="rvsdg-dump-placeholder">
                                    <p>"Select an RVSDG dump file and then a function to visualize."</p>
                                </div>
                            }.into_any()
                        }
                    }}
                </div>
            </Layout>
        </Layout>
    }
}
