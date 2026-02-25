//! Shims for whitelisted APIs in Rust's `core`/`std` libraries.
//!
//! This is an implementation detail of the RISL compiler. You should never use anything defined in
//! this module directly; instead use the corresponding type, trait, or function in `core`/`std`.

pub mod ops;
pub mod slice;
