#![feature(track_path)]

use std::path::PathBuf;
use std::sync::LazyLock;
use std::{env, fs};

use proc_macro::TokenStream;

mod compute;
mod fragment;
mod gpu;
mod impl_mat_mul;
mod resource;
mod shader_io;
mod shader_module;
mod shader_wgsl;
mod vertex;
mod workgroup_shared;

static IS_RISLC_PASS: LazyLock<bool> = LazyLock::new(|| env::var("IS_RISLC_PASS").is_ok());

static TARGET_DIR: LazyLock<PathBuf> = LazyLock::new(|| {
    let target_dir = env::var("CARGO_TARGET_DIR").unwrap_or_default();

    PathBuf::from(target_dir)
});

#[proc_macro_attribute]
pub fn compute(attr: TokenStream, item: TokenStream) -> TokenStream {
    compute::expand_attribute(attr, item)
}

#[proc_macro_attribute]
pub fn fragment(attr: TokenStream, item: TokenStream) -> TokenStream {
    fragment::expand_attribute(attr, item)
}

#[proc_macro_attribute]
pub fn gpu(attr: TokenStream, item: TokenStream) -> TokenStream {
    gpu::expand_attribute(attr, item)
}

#[proc_macro_attribute]
pub fn resource(attr: TokenStream, item: TokenStream) -> TokenStream {
    resource::expand_attribute(attr, item)
}

#[proc_macro_attribute]
pub fn shader_io(attr: TokenStream, item: TokenStream) -> TokenStream {
    shader_io::expand_attribute(attr, item)
}

#[proc_macro_attribute]
pub fn shader_module(attr: TokenStream, item: TokenStream) -> TokenStream {
    shader_module::expand_attribute(attr, item)
}

#[proc_macro]
pub fn shader_wgsl(input: TokenStream) -> TokenStream {
    shader_wgsl::expand_shader_wgsl(input)
}

#[proc_macro_attribute]
pub fn vertex(attr: TokenStream, item: TokenStream) -> TokenStream {
    vertex::expand_attribute(attr, item)
}

#[proc_macro_attribute]
pub fn workgroup_shared(attr: TokenStream, item: TokenStream) -> TokenStream {
    workgroup_shared::expand_attribute(attr, item)
}

/// Helper macro for generating matrix multiplication implementations.
///
/// This is a utility macro used to generate `mat * mat`, `mat * vec` and `vec * mat`
/// implementations of `core::ops::Mul` inside the `risl` crate, as macro-rules based macros don't
/// allow is to cleanly do this. It is not useful outside of that and not intended for public use.
///
/// It expects a token-stream formatted as follows:
///
/// ```psuedocode
/// // For matrix-matrix multiplication
/// mat<2, 3> * mat<4, 2>
///
/// // For matrix-vector multiplication
/// mat<2, 3> * vec<2>
/// ```
#[doc(hidden)]
#[proc_macro]
pub fn impl_mat_mul(token_stream: TokenStream) -> TokenStream {
    impl_mat_mul::expand_macro(token_stream)
}
