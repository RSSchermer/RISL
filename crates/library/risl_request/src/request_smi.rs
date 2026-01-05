use ar::Archive;
use proc_macro2::Span;
use quote::quote;
use syn::LitStr;

use crate::smi::ShaderModuleInterface;
use crate::{Request, is_rislc_pass, request_id, resolve_artifact};

pub fn request_shader_module_interface(
    mod_path: proc_macro::TokenStream,
) -> crate::Result<ShaderModuleInterface> {
    let request_id = request_id(&mod_path);
    let mod_path = proc_macro2::TokenStream::from(mod_path);

    if is_rislc_pass() {
        let request_id = LitStr::new(&request_id, Span::call_site());

        let request_expr = quote! {
            #[rislc::shader_module_interface(#request_id)]
            const {
                use #mod_path;

                risl::smi::ShaderModuleInterface {
                    overridable_constants: &[],
                    resource_bindings: &[],
                    entry_points: &[],
                }
            }
        };

        Ok(Request::TokenStream(request_expr.into()))
    } else {
        let artifact_file = resolve_artifact(&request_id)?;

        let mut archive = Archive::new(artifact_file);
        let mut smi = None;

        while let Some(entry) = archive.next_entry() {
            let mut entry = entry.unwrap();

            if entry.header().identifier() == "smi".as_bytes() {
                let decoded = bincode::serde::decode_from_std_read::<ShaderModuleInterface, _, _>(
                    &mut entry,
                    bincode::config::standard(),
                )
                .expect("SMI archive entry is not a valid encoding");

                smi = Some(decoded);
            }
        }

        let Some(smi) = smi else {
            return Err(syn::Error::new(
                Span::call_site(),
                "could not find a shader-module-interface entry in the RISL artifact; are you \
                    building this with the RISL compiler?",
            ));
        };

        Ok(Request::Resolution(smi))
    }
}
