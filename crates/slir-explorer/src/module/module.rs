use std::io::Read as IoRead;

use ar::Archive;
use leptos::prelude::*;
use leptos_router::hooks::use_params_map;
use leptos_router::nested_router::Outlet;
use slir::cfg::Cfg;
use slir::rvsdg::Rvsdg;
use slir::scf::Scf;
use slir::Module;
use slotmap::Key;
use thaw::*;
use thaw_utils::Model;
use urlencoding::{decode as urldecode, encode as urlencode};

use crate::module::url::function_url;

pub fn use_module_data() -> StoredValue<ModuleData> {
    use_context::<StoredValue<ModuleData>>().expect("should be used inside a module context")
}

pub struct ModuleData {
    pub module: Module,
    pub cfg: Cfg,
    pub rvsdg_initial: Option<Rvsdg>,
    pub rvsdg_transformed: Option<Rvsdg>,
    pub scf: Option<Scf>,
    pub wgsl: Option<String>,
}

impl ModuleData {
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        let mut archive = Archive::new(bytes);

        let mut module = None;
        let mut cfg = None;
        let mut rvsdg_initial = None;
        let mut rvsdg_transformed = None;
        let mut scf = None;
        let mut wgsl = None;

        while let Some(entry_result) = archive.next_entry() {
            let mut entry = entry_result.unwrap();

            if entry.header().identifier() == "module".as_bytes() {
                let decoded: slir::Module =
                    bincode::serde::decode_from_std_read(&mut entry, bincode::config::standard())
                        .map_err(|_| "module encoding was invalid".to_string())?;

                module = Some(decoded);
            }

            if entry.header().identifier() == "cfg".as_bytes() {
                let decoded: slir::cfg::CfgData =
                    bincode::serde::decode_from_std_read(&mut entry, bincode::config::standard())
                        .map_err(|_| "CFG encoding was invalid".to_string())?;

                cfg = Some(decoded);
            }

            if entry.header().identifier() == "rvsdg_initial".as_bytes() {
                let decoded: slir::rvsdg::RvsdgData =
                    bincode::serde::decode_from_std_read(&mut entry, bincode::config::standard())
                        .map_err(|_| "RSVDG-initial encoding was invalid".to_string())?;

                rvsdg_initial = Some(decoded);
            }

            if entry.header().identifier() == "rvsdg_transformed".as_bytes() {
                let decoded: slir::rvsdg::RvsdgData =
                    bincode::serde::decode_from_std_read(&mut entry, bincode::config::standard())
                        .map_err(|_| "RSVDG-transformed encoding was invalid".to_string())?;

                rvsdg_transformed = Some(decoded);
            }

            if entry.header().identifier() == "scf".as_bytes() {
                let decoded: slir::scf::ScfData =
                    bincode::serde::decode_from_std_read(&mut entry, bincode::config::standard())
                        .map_err(|_| "SCF encoding was invalid".to_string())?;

                scf = Some(decoded);
            }

            if entry.header().identifier() == "wgsl".as_bytes() {
                let mut decoded = String::new();

                entry
                    .read_to_string(&mut decoded)
                    .expect("could not read WGSL");

                wgsl = Some(decoded);
            }
        }

        let module =
            module.ok_or("SLIR arfifact should always contain a `module` entry".to_string())?;
        let cfg_data =
            cfg.ok_or("SLIR arfifact should always contain a `cfg` entry".to_string())?;
        let cfg = Cfg::from_ty_and_data(module.ty.clone(), cfg_data);
        let rvsdg_initial =
            rvsdg_initial.map(|data| Rvsdg::from_ty_and_data(module.ty.clone(), data));
        let rvsdg_transformed =
            rvsdg_transformed.map(|data| Rvsdg::from_ty_and_data(module.ty.clone(), data));
        let scf = scf.map(|data| Scf::from_ty_and_data(module.ty.clone(), data));

        Ok(ModuleData {
            module,
            cfg,
            rvsdg_initial,
            rvsdg_transformed,
            scf,
            wgsl,
        })
    }

    pub fn expect_scf(&self) -> &Scf {
        self.scf.as_ref().expect("SCF data is not present")
    }
}

#[server]
async fn get_module_bytes(module_name: String) -> Result<Vec<u8>, ServerFnError> {
    let filename = format!("{}/{}.slir", crate::app::MODULE_DIR, module_name);

    std::fs::read(filename).map_err(|e| ServerFnError::ServerError(e.to_string()))
}

#[component]
pub fn Module() -> impl IntoView {
    let params = use_params_map();
    let module_name = move || {
        urldecode(&params.read().get("module_name").unwrap_or_default())
            .unwrap_or_default()
            .to_string()
    };

    let module_bytes = Resource::new(module_name, |module_name| async {
        get_module_bytes(module_name).await
    });

    let module_data = move || {
        module_bytes.and_then(|bytes| ModuleData::decode(bytes).map(|d| StoredValue::new(d)))
    };

    view! {
        {move || match module_data() {
            Some(Ok(Ok(module_data))) => {
                provide_context(module_data);

                view! {
                    <div class="module-explorer-container">
                        <div class="module-items-container">
                            <ModuleNav module_data/>
                        </div>
                        <div class="item-explorer-container">
                            <Outlet/>
                        </div>
                    </div>
                }.into_any()
            }
            Some(Ok(Err(err))) => {
                view! {
                    <div class="info-page-container">
                        <h1>"Error decoding module"</h1>

                        <p>{err}</p>
                    </div>
                }.into_any()
            }
            Some(Err(err)) => {
                view! {
                    <div class="info-page-container">
                        <h1>"Error loading module"</h1>

                        <p>{err.to_string()}</p>
                    </div>
                }.into_any()
            }
            None => {
                view! { <p>"Loading..."</p> }.into_any()
            }
        }}
    }
}

#[component]
pub fn ModuleNav(module_data: StoredValue<ModuleData>) -> impl IntoView {
    let module_name = module_data.read_value().module.name;

    let functions = move || {
        let mut functions = module_data
            .read_value()
            .module
            .fn_sigs
            .keys()
            .collect::<Vec<_>>();

        functions.sort_by(|a, b| a.name.cmp(&b.name));

        functions
    };

    let open_categories: Model<Vec<String>> =
        Model::from(RwSignal::new(vec!["functions".to_string()]));

    view! {
        <NavDrawer multiple=true open_categories class="module-nav">
            <NavCategory value="uniform_bindings">
                <NavCategoryItem slot>
                    "Uniform Bindings"
                </NavCategoryItem>
                {module_data.read_value().module.uniform_bindings.keys().map(|b| view! {
                    <NavSubItem
                        value=format!("u{}", b.data().as_ffi())
                        href=format!("/{}/uniform_bindings/{}", urlencode(&module_name.as_str()), b.data().as_ffi())
                    >
                        {format!("U{}", b.data().as_ffi())}
                    </NavSubItem>
                }).collect_view()}
            </NavCategory>
            <NavCategory value="storage_bindings">
                <NavCategoryItem slot>
                    "Storage Bindings"
                </NavCategoryItem>
                {module_data.read_value().module.storage_bindings.keys().map(|b| view! {
                    <NavSubItem
                        value=format!("s{}", b.data().as_ffi())
                        href=format!("/{}/storage_bindings/{}", urlencode(&module_name.as_str()), b.data().as_ffi())
                    >
                        {format!("S{}", b.data().as_ffi())}
                    </NavSubItem>
                }).collect_view()}
            </NavCategory>
            <NavCategory value="workgroup_bindings">
                <NavCategoryItem slot>
                    "Workgroup Bindings"
                </NavCategoryItem>
                {module_data.read_value().module.workgroup_bindings.keys().map(|b| view! {
                    <NavSubItem
                        value=format!("w{}", b.data().as_ffi())
                        href=format!("/{}/workgroup_bindings/{}", urlencode(&module_name.as_str()), b.data().as_ffi())
                    >
                        {format!("W{}", b.data().as_ffi())}
                    </NavSubItem>
                }).collect_view()}
            </NavCategory>
            <NavCategory value="functions">
                <NavCategoryItem slot>
                    "Functions"
                </NavCategoryItem>
                {move || functions().into_iter().map(|f| view! {
                    <NavSubItem
                        value=format!("f{}--{}", f.module, f.name)
                        href=function_url(module_name, f)
                    >
                        {f.name.to_string()}
                    </NavSubItem>
                }).collect_view()}
            </NavCategory>
            <NavItem
                value="wgsl"
                href=format!("/{}/wgsl", urlencode(&module_name.as_str()))
            >
                "WGSL"
            </NavItem>
        </NavDrawer>
    }
}
