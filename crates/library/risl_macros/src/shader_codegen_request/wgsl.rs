use std::fs::File;
use std::io::Read;

use ar::Archive;
use proc_macro::{TokenStream, tracked};
use proc_macro2::Span;
use quote::{ToTokens, quote};
use syn::{LitStr, Path, parse_macro_input};

use crate::is_rislc_pass;
use crate::shader_codegen_request::{macro_invocation_id, resolve_artifact};

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
        let artifact_path = match resolve_artifact(&request_id) {
            Ok(path) => path,
            Err(err) => return err.into_compile_error().into(),
        };

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
