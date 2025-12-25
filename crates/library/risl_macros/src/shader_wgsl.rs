use std::collections::HashMap;
use std::ffi::OsString;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use ar::Archive;
use base64::Engine;
use proc_macro::{TokenStream, tracked};
use proc_macro2::Span;
use quote::{ToTokens, quote};
use syn::{LitStr, Path, parse_macro_input};

use crate::{is_rislc_pass, risl_shader_request_lookup};

pub fn expand_shader_wgsl(input: TokenStream) -> TokenStream {
    let shader_module_path = parse_macro_input!(input as Path);
    let request_id = macro_invocation_id();

    if is_rislc_pass() {
        let request_id = LitStr::new(&request_id, Span::call_site());

        quote! {
            #[rislc::shader_wgsl(#request_id)]
            const {
                use #shader_module_path;

                ""
            }
        }
        .into()
    } else {
        let Ok(mut lookup_file) = File::open(risl_shader_request_lookup()) else {
            return quote! {
                compile_error!(
                    "could not open RISL request info; are you building this with the RISL \
                    compiler?"
                )
            }
            .into();
        };

        let lookup: HashMap<String, OsString> =
            bincode::serde::decode_from_std_read(&mut lookup_file, bincode::config::standard())
                .unwrap();

        let Some(artifact_path) = lookup.get(&request_id) else {
            return quote! {
                compile_error!(
                    "could not find RISL request info; are you building this with the RISL \
                    compiler?"
                )
            }
            .into();
        };

        let artifact_path = PathBuf::from(artifact_path);

        if !artifact_path.exists() {
            return quote! {
                compile_error!(
                    "could not find RISL artifact; are you building this with the RISL compiler?"
                )
            }
            .into();
        }

        let mut archive = Archive::new(File::open(&artifact_path).unwrap());

        let mut wgsl = None;

        while let Some(entry) = archive.next_entry() {
            let mut entry = entry.unwrap();

            if entry.header().identifier() == "wgsl".as_bytes() {
                let mut decoded = String::new();

                entry
                    .read_to_string(&mut decoded)
                    .expect("WGSL archive entry is not a valid UTF-8 encoding");

                wgsl = Some(decoded);
            }
        }

        let Some(wgsl) = wgsl else {
            return quote! {
                compile_error!(
                    "could not find a WGSL entry in the RISL artifact; are you building this with \
                    the RISL compiler?"
                )
            }
            .into();
        };

        tracked::path(
            artifact_path
                .to_str()
                .expect("cannot track non-unicode path"),
        );

        LitStr::new(&wgsl, Span::call_site())
            .to_token_stream()
            .into()
    }
}

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
