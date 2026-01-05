#![feature(proc_macro_tracked_path)]

mod request_smi;
mod request_wgsl;

use std::collections::HashMap;
use std::env;
use std::ffi::OsString;
use std::fs::File;
use std::path::PathBuf;

use base64::Engine;
use proc_macro::tracked;
use proc_macro2::Span;

extern crate proc_macro;

pub use risl_smi as smi;

pub use crate::request_smi::request_shader_module_interface;
pub use crate::request_wgsl::request_wgsl;

pub enum Request<T> {
    TokenStream(proc_macro::TokenStream),
    Resolution(T),
}

pub type Result<T> = std::result::Result<Request<T>, syn::Error>;

fn is_rislc_pass() -> bool {
    env::var("IS_RISLC_PASS").is_ok()
}

/// Generates a unique request identifier based on the combination of the current proc-macro
/// invocation's call-site and the path of target-module for the request.
///
/// Based on the file, line, and column of the macro's call-site span. Uses the OS-string encoding
/// of the file path as part of the ID, so it is not stable across different compiler versions,
/// though that should not affect rislc, as it only needs to be stable between the 2 compiler
/// passes (which it obviously will be).
fn request_id(mod_path: &proc_macro::TokenStream) -> String {
    let span = proc_macro::Span::call_site();

    // TODO: find out if this can fail.
    let file = span.local_file().unwrap();

    let mut bytes = file.as_os_str().as_encoded_bytes().to_vec();

    bytes.extend(span.line().to_ne_bytes());
    bytes.extend(span.column().to_ne_bytes());

    let path_str = mod_path.to_string();

    bytes.extend(path_str.as_bytes());

    base64::engine::general_purpose::STANDARD.encode(&bytes)
}

fn risl_shader_request_lookup() -> PathBuf {
    let target_dir = env::var("RISL_SHADER_REQUEST_LOOKUP").unwrap_or_default();

    PathBuf::from(target_dir)
}

fn resolve_artifact(request_id: &str) -> syn::Result<File> {
    let mut lookup_file = File::open(risl_shader_request_lookup()).map_err(|_| {
        syn::Error::new(
            Span::call_site(),
            "could not open RISL request info; are you building this with the RISL compiler?",
        )
    })?;

    let lookup: HashMap<String, OsString> =
        bincode::serde::decode_from_std_read(&mut lookup_file, bincode::config::standard())
            .unwrap();

    let artifact_path = lookup.get(request_id).ok_or_else(|| {
        syn::Error::new(
            Span::call_site(),
            "could not open RISL request info; are you building this with the RISL compiler?",
        )
    })?;

    let artifact_path = PathBuf::from(artifact_path);

    let artifact_file = File::open(&artifact_path).map_err(|_| {
        syn::Error::new(
            Span::call_site(),
            "could not find RISL artifact; are you building this with the RISL compiler?",
        )
    })?;

    tracked::path(
        artifact_path
            .to_str()
            .expect("cannot track non-unicode path"),
    );

    Ok(artifact_file)
}
