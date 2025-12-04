#![feature(core_intrinsics)]

pub mod mem;
pub mod primitive;
pub mod resource;
pub mod workgroup;

pub use risl_macros::{gpu};

pub mod prelude {
    pub use super::mem::{Storage, StorageMut, Uniform, Workgroup};
    pub use super::primitive::*;
    pub use super::resource::{Resource, resource};
    pub use super::shader::{compute, fragment, shader_io, shader_module, vertex};
    pub use super::workgroup::workgroup_shared;
    pub use super::{gpu};
}

pub mod shader {
    pub use risl_macros::{compute, fragment, shader_io, shader_module, shader_wgsl, vertex};
}

pub mod smi {
    pub use empa_smi::{
        ArrayLayout, EntryPoint, Interpolate, InterpolationType, IoBinding, IoBindingType,
        MemoryUnit, MemoryUnitLayout, OverridableConstant, OverridableConstantType,
        ResourceBinding, ResourceType, Sampling, ShaderModuleInterface, ShaderStage,
        SizedBufferLayout, StorageTextureFormat, TexelType, UnsizedBufferLayout, UnsizedTailLayout,
    };
}
