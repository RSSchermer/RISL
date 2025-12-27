use std::fs::File;

use ar::Archive;
use empa_smi::{ShaderModuleInterface, smi_to_token_stream};
use proc_macro::{TokenStream, tracked};
use proc_macro2::Span;
use quote::{ToTokens, quote};
use syn::{LitStr, Path, parse_macro_input};

use crate::is_rislc_pass;
use crate::shader_codegen_request::{macro_invocation_id, resolve_artifact};

pub fn expand_shader_module_interface(input: TokenStream) -> TokenStream {
    let shader_module_path = parse_macro_input!(input as Path);
    let request_id = macro_invocation_id();

    if is_rislc_pass() {
        let request_id = LitStr::new(&request_id, Span::call_site());

        quote! {
            #[rislc::shader_module_interface(#request_id)]
            const {
                use #shader_module_path;

                risl::smi::ShaderModuleInterface {
                    overridable_constants: std::borrow::Cow::Borrowed(&[]),
                    resource_bindings: std::borrow::Cow::Borrowed(&[]),
                    entry_points: std::borrow::Cow::Borrowed(&[]),
                }
            }
        }
        .into()
    } else {
        let artifact_path = match resolve_artifact(&request_id) {
            Ok(path) => path,
            Err(err) => return err.into_compile_error().into(),
        };

        let mut archive = Archive::new(File::open(&artifact_path).unwrap());

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
            return quote! {
                compile_error!(
                    "could not find a shader-module-interface entry in the RISL artifact; are you \
                    building this with the RISL compiler?"
                )
            }
            .into();
        };

        tracked::path(
            artifact_path
                .to_str()
                .expect("cannot track non-unicode path"),
        );

        smi_to_token_stream(&smi, &quote!(risl::smi)).into()
    }
}
