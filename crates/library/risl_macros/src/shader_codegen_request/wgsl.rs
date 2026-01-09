use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{ToTokens, quote};
use risl_request::{Request, request_wgsl};
use syn::LitStr;

pub fn expand_shader_wgsl(input: TokenStream) -> TokenStream {
    match request_wgsl(input) {
        Ok(Request::TokenStream(request)) => {
            let request = proc_macro2::TokenStream::from(request);

            quote! {
                const {
                    #request;

                    ""
                }
            }
            .into()
        }
        Ok(Request::Resolution(wgsl)) => LitStr::new(&wgsl, Span::call_site())
            .to_token_stream()
            .into(),
        Err(err) => err.into_compile_error().into(),
    }
}
