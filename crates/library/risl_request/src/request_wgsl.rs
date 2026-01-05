use std::io::Read;

use ar::Archive;
use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::LitStr;

use crate::{Request, is_rislc_pass, request_id, resolve_artifact};

pub fn request_wgsl(mod_path: TokenStream) -> crate::Result<String> {
    let request_id = request_id(&mod_path);
    let mod_path = proc_macro2::TokenStream::from(mod_path);

    if is_rislc_pass() {
        let request_id = LitStr::new(&request_id, Span::call_site());

        let request_expr = quote! {
            #[rislc::shader_wgsl(#request_id)]
            const {
                use #mod_path;

                ""
            }
        };

        Ok(Request::TokenStream(request_expr.into()))
    } else {
        let artifact_file = resolve_artifact(&request_id)?;

        let mut archive = Archive::new(artifact_file);
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
            return Err(syn::Error::new(
                Span::call_site(),
                "could not find a WGSL entry in the RISL artifact; are you building this with \
                the RISL compiler?",
            ));
        };

        Ok(Request::Resolution(wgsl))
    }
}
