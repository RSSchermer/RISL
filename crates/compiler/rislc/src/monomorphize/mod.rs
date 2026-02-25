//! This is modified from `rustc_monomorphize`. There are two main reasons for replicating mono-item
//! collection here:
//!
//! 1. We collect different roots (we are only interested in "gpu" items).
//! 2. `rustc`'s mono-item collection is also what triggers const-eval and replaces `const`
//!    expressions with the evaluation's result; we want to preserve `const` expressions for
//!    overridable/specializable constants.
//!

mod collect;
mod errors;

pub use collect::collect_shader_module_codegen_units;
