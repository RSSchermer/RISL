use std::collections::HashMap;
use std::ffi::OsString;
use std::fs::File;
use std::path::PathBuf;

use base64::Engine;
use proc_macro2::Span;

use crate::risl_shader_request_lookup;

pub mod smi;
pub mod wgsl;

/// Generates an identifier for the current macro invocation that is unique within the crate that is
/// currently being compiled and that is stable across multiple compilations of the same code with
/// the same compiler version.
///
/// Based on the file, line and column of the macro's call-site span. Uses the OS-string encoding
/// for the file path is part of the ID, so it is not stable across different compiler versions,
/// though that should not affect rislc.
fn macro_invocation_id() -> String {
    let span = proc_macro::Span::call_site();

    // TODO: find out if this can fail.
    let file = span.local_file().unwrap();

    let mut bytes = file.as_os_str().as_encoded_bytes().to_vec();

    bytes.extend(span.line().to_ne_bytes());
    bytes.extend(span.column().to_ne_bytes());

    base64::engine::general_purpose::STANDARD.encode(&bytes)
}

fn resolve_artifact(request_id: &str) -> syn::Result<PathBuf> {
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

    if !artifact_path.exists() {
        let err = syn::Error::new(
            Span::call_site(),
            "could not find RISL artifact; are you building this with the RISL compiler?",
        );

        return Err(err);
    }

    Ok(artifact_path)
}
