#![feature(iter_intersperse)]
#![recursion_limit = "256"]

pub mod app;
pub mod module;
pub mod no_module_selected;
pub mod not_found;
pub mod rvsdg_dump;

#[cfg(feature = "hydrate")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn hydrate() {
    use app::*;
    console_error_panic_hook::set_once();
    leptos::mount::hydrate_body(App);
}
