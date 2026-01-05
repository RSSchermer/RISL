#![feature(core_intrinsics)]

pub mod mem;
pub mod primitive;
pub mod resource;
pub mod smi;
pub mod workgroup;

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
