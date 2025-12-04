use proc_macro::TokenStream;
use proc_macro2::{Ident, Span};
use quote::quote;
use syn::token::Paren;
use syn::{
    Field, Item, MacroDelimiter, Meta, MetaList, Path, PathArguments, PathSegment,
    parse_macro_input,
};

pub fn expand_attribute(attr: TokenStream, item: TokenStream) -> TokenStream {
    if !attr.is_empty() {
        return quote! {
            compile_error!("the `shader_io` attribute does not accept any arguments");
        }
        .into();
    }

    let item = parse_macro_input!(item as Item);

    let Item::Struct(mut struct_decl) = item else {
        return quote! {
            compile_error!("the `shader_io` attribute can only be applied to `struct` items");
        }
        .into();
    };

    struct_decl.fields.iter_mut().for_each(adjust_field);

    quote! {
        #[cfg_attr(rislc, rislc::shader_io)]
        #struct_decl
    }
    .into()
}

fn adjust_field(field: &mut Field) {
    for attr in &mut field.attrs {
        if is_shader_io_path(attr.meta.path()) {
            adjust_meta(&mut attr.meta);
        }
    }
}

/// Wraps the meta in a `cfg_attr` outer meta and prepends the tool path.
///
/// For example:
///
/// ```pseudocode
/// location(0)
/// ```
///
/// Becomes:
///
/// ```pseudocode
/// cfg_attr(rislc, rislc::location(0))
/// ```
fn adjust_meta(meta: &mut Meta) {
    let mut tokens = quote! {
        rislc, rislc::#meta
    };

    let cfg_attr_path = PathSegment {
        ident: Ident::new("cfg_attr", Span::call_site()),
        arguments: PathArguments::None,
    };

    *meta = Meta::List(MetaList {
        path: cfg_attr_path.into(),
        delimiter: MacroDelimiter::Paren(Paren::default()),
        tokens,
    });
}

fn is_shader_io_path(path: &Path) -> bool {
    path.is_ident("location")
        || path.is_ident("builtin")
        || path.is_ident("invariant")
        || path.is_ident("interpolate")
        || path.is_ident("blend_src")
}
