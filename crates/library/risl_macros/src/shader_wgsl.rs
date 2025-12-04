use std::ffi::OsString;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use ar::Archive;
use base64::Engine;
use proc_macro::{TokenStream, tracked_path};
use proc_macro2::Span;
use quote::{ToTokens, quote};
use syn::{LitStr, Path, parse_macro_input};

use crate::{IS_RISLC_PASS, TARGET_DIR};

pub fn expand_shader_wgsl(input: TokenStream) -> TokenStream {
    let shader_module_path = parse_macro_input!(input as Path);
    let request_db_path = TARGET_DIR.join("shader_wgsl_requests");
    let request_db = sled::open(request_db_path).unwrap();
    let request_id = macro_invocation_id();

    if *IS_RISLC_PASS {
        // Overwrite current entry (if any) so that we error rather than load a stale path in case
        // something goes wrong.
        request_db.insert(&request_id, "").unwrap();

        let request_id = LitStr::new(&request_id, Span::call_site());

        quote! {
            #[rislc::shader_wgsl(#request_id)]
            const {
                #shader_module_path

                ""
            }
        }
        .into()
    } else {
        let Some(artifact_path) = request_db.get(&request_id).unwrap() else {
            return quote! {
                compile_error!(
                    "could not find RISL request info; are you building this with the RISL \
                    compiler?"
                )
            }
            .into();
        };

        // SAFETY: should have just been set by the rislc pass, which should use the same rustc
        // version and thus the same encoding.
        let artifact_path =
            unsafe { OsString::from_encoded_bytes_unchecked(artifact_path.to_vec()) };
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

        tracked_path::path(
            artifact_path
                .to_str()
                .expect("cannot track non-unicode path"),
        );

        LitStr::new(&wgsl, Span::call_site())
            .to_token_stream()
            .into()
    }
}

/// Generates an identifier unique to the current macro invocation that is stable across multiple
/// compilations of the same code with the same compiler version.
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
