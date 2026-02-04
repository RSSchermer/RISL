#![feature(core_intrinsics, custom_inner_attributes)]
#![cfg_attr(rislc, rislc::core_shim_crate)]

pub mod mem;
pub mod primitive;
pub mod resource;
pub mod smi;
pub mod workgroup;

#[doc(hidden)]
pub mod core_shim;

mod intrinsic;

pub use risl_macros::gpu;

pub mod prelude {
    pub use super::gpu;
    pub use super::mem::{Storage, StorageMut, Uniform, Workgroup};
    pub use super::primitive::*;
    pub use super::resource::{Resource, resource};
    pub use super::shader::{compute, fragment, shader_io, shader_module, vertex};
    pub use super::workgroup::workgroup_shared;
}

pub mod shader {
    pub use risl_macros::{
        compute, fragment, shader_io, shader_module, shader_module_interface, shader_wgsl, vertex,
    };
}
