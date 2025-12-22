use leptos::prelude::*;
use leptos_meta::{provide_meta_context, Stylesheet, Title};
use leptos_router::components::{ParentRoute, Route, Router, Routes};
use leptos_router::path;
use thaw::*;
use urlencoding::encode as urlencode;

use crate::module;
use crate::no_module_selected::NoModuleSelected;
use crate::not_found::NotFound;

pub const MODULE_DIR: &'static str = "target/debug/deps/risl";

#[component]
pub fn App() -> impl IntoView {
    // Provides context that manages stylesheets, titles, meta tags, etc.
    provide_meta_context();

    let open_module_list = RwSignal::new(false);

    view! {
        // injects a stylesheet into the document <head>
        // id=leptos means cargo-leptos will hot-reload this stylesheet
        <Stylesheet id="leptos" href="/pkg/slir-explorer.css"/>

        <Title text="SLIR Explorer"/>

        <ConfigProvider>
            <Router>
                <div class="app-container">
                    <nav class="nav-main">
                        <Button on:click=move |_| open_module_list.set(true)><Icon icon=icondata::BiFileFindRegular/> "Select Module"</Button>
                    </nav>

                    <main>
                        <Routes fallback=NotFound>
                            <ParentRoute path=path!(":module_name") view=module::Module>
                                <ParentRoute path=path!("uniform_bindings") view=module::uniform_bindings::UniformBindings>
                                    <Route path=path!(":uniform_binding_id") view=module::uniform_bindings::Detail/>
                                    <Route path=path!("") view=module::uniform_bindings::List/>
                                </ParentRoute>
                                <ParentRoute path=path!("storage_bindings") view=module::storage_bindings::StorageBindings>
                                    <Route path=path!(":storage_binding_id") view=module::storage_bindings::Detail/>
                                    <Route path=path!("") view=module::storage_bindings::List/>
                                </ParentRoute>
                                <ParentRoute path=path!("workgroup_bindings") view=module::workgroup_bindings::WorkgroupBindings>
                                    <Route path=path!(":workgroup_binding_id") view=module::workgroup_bindings::Detail/>
                                    <Route path=path!("") view=module::workgroup_bindings::List/>
                                </ParentRoute>
                                <ParentRoute path=path!("functions") view=module::functions::Functions>
                                    <Route path=path!(":function_name") view=module::functions::Detail/>
                                    <Route path=path!("") view=module::functions::List/>
                                </ParentRoute>
                                <Route path=path!("adts/:ty_id") view=module::adts::Adt/>
                                <Route path=path!("wgsl") view=module::WgslExplorer/>
                                <Route path=path!("") view=module::NoItemSelected/>
                            </ParentRoute>
                            <Route path=path!("") view=NoModuleSelected/>
                        </Routes>
                    </main>
                </div>

                <ModuleList open=open_module_list/>
            </Router>
        </ConfigProvider>
    }
}

#[component]
pub fn ModuleList(open: RwSignal<bool>) -> impl IntoView {
    #[server]
    pub async fn get_module_list() -> Result<Vec<String>, ServerFnError> {
        match std::fs::read_dir(MODULE_DIR) {
            Ok(entries) => {
                let mut names = Vec::new();

                for entry in entries {
                    match entry {
                        Ok(entry) => {
                            let path = entry.path();

                            if path.extension().map(|e| e == "slir").unwrap_or(false) {
                                let module_name = path
                                    .as_path()
                                    .file_stem()
                                    .and_then(|n| n.to_str())
                                    .unwrap_or("")
                                    .to_string();

                                names.push(module_name);
                            }
                        }
                        Err(err) => {
                            return Err(ServerFnError::ServerError(err.to_string()));
                        }
                    }
                }

                names.sort();

                Ok(names)
            }
            Err(err) => Err(ServerFnError::ServerError(err.to_string())),
        }
    }

    let modules = Resource::new(|| (), |_| async move { get_module_list().await.unwrap() });

    view! {
        <OverlayDrawer open position={DrawerPosition::Left}>
            <DrawerHeader>
                <DrawerHeaderTitle>
                    <DrawerHeaderTitleAction slot>
                        <Button
                            appearance=ButtonAppearance::Subtle
                            on_click=move |_| open.set(false)
                        >
                            "x"
                        </Button>
                    </DrawerHeaderTitleAction>
                    {MODULE_DIR}
                </DrawerHeaderTitle>
            </DrawerHeader>
            <DrawerBody>
                <Suspense
                    fallback=move || view! { <p>"Loading..."</p> }
                >
                    <ul>
                        {move || {
                            modules.get().map(|modules| {
                                modules.into_iter().map(|module| view!{
                                    <li>
                                        <a on:click=move |_| open.set(false)
                                            href=format!("/{}", urlencode(&module))
                                        >
                                            {module.clone()}
                                        </a>
                                    </li>
                                }).collect_view()
                            })
                        }}
                    </ul>
                </Suspense>
            </DrawerBody>
        </OverlayDrawer>
    }
}
