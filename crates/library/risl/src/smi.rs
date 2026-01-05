//! Defines types that may be used to describe the interface of a RISL shader module.

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub enum OverridableConstantType {
    Float,
    Bool,
    SignedInteger,
    UnsignedInteger,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub enum TexelType {
    Float,
    UnfilterableFloat,
    Integer,
    UnsignedInteger,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[allow(non_camel_case_types)]
#[non_exhaustive]
pub enum StorageTextureFormat {
    rgba8unorm,
    rgba8snorm,
    rgba8uint,
    rgba8sint,
    rgba16uint,
    rgba16sint,
    rgba16float,
    r32uint,
    r32sint,
    r32float,
    rg32uint,
    rg32sint,
    rg32float,
    rgba32uint,
    rgba32sint,
    rgba32float,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub enum IoBindingType {
    SignedInteger,
    SignedIntegerVector2,
    SignedIntegerVector3,
    SignedIntegerVector4,
    UnsignedInteger,
    UnsignedIntegerVector2,
    UnsignedIntegerVector3,
    UnsignedIntegerVector4,
    Float,
    FloatVector2,
    FloatVector3,
    FloatVector4,
    HalfFloat,
    HalfFloatVector2,
    HalfFloatVector3,
    HalfFloatVector4,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub enum InterpolationType {
    Perspective,
    Linear,
    Flat,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub enum Sampling {
    Center,
    Centroid,
    Sample,
    First,
    Either,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Interpolate {
    pub interpolation_type: InterpolationType,
    pub sampling: Option<Sampling>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct IoBinding {
    pub location: u32,
    pub binding_type: IoBindingType,
    pub interpolate: Option<Interpolate>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct OverridableConstant {
    pub id: u16,
    pub constant_type: OverridableConstantType,
    pub required: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ArrayLayout {
    pub element_layout: &'static [MemoryUnit],
    pub stride: u64,
    pub len: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub enum MemoryUnitLayout {
    Float,
    FloatVector2,
    FloatVector3,
    FloatVector4,
    Integer,
    IntegerVector2,
    IntegerVector3,
    IntegerVector4,
    UnsignedInteger,
    UnsignedIntegerVector2,
    UnsignedIntegerVector3,
    UnsignedIntegerVector4,
    Matrix2x2,
    Matrix2x3,
    Matrix2x4,
    Matrix3x2,
    Matrix3x3,
    Matrix3x4,
    Matrix4x2,
    Matrix4x3,
    Matrix4x4,
    Array(ArrayLayout),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MemoryUnit {
    pub offset: u64,
    pub layout: MemoryUnitLayout,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SizedBufferLayout {
    pub memory_units: &'static [MemoryUnit],
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct UnsizedTailLayout {
    pub offset: u64,
    pub element_layout: &'static [MemoryUnit],
    pub stride: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct UnsizedBufferLayout {
    pub sized_head: &'static [MemoryUnit],
    pub unsized_tail: Option<UnsizedTailLayout>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub enum ResourceType {
    Texture1D(TexelType),
    Texture2D(TexelType),
    Texture3D(TexelType),
    Texture2DArray(TexelType),
    TextureCube(TexelType),
    TextureCubeArray(TexelType),
    TextureMultisampled2D(TexelType),
    TextureDepth2D,
    TextureDepth2DArray,
    TextureDepthCube,
    TextureDepthCubeArray,
    TextureDepthMultisampled2D,
    StorageTexture1D(StorageTextureFormat),
    StorageTexture2D(StorageTextureFormat),
    StorageTexture2DArray(StorageTextureFormat),
    StorageTexture3D(StorageTextureFormat),
    FilteringSampler,
    NonFilteringSampler,
    ComparisonSampler,
    Uniform(SizedBufferLayout),
    StorageRead(UnsizedBufferLayout),
    StorageReadWrite(UnsizedBufferLayout),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ResourceBinding {
    pub group: u32,
    pub binding: u32,
    pub resource_type: ResourceType,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct EntryPoint {
    pub name: &'static str,
    pub stage: ShaderStage,
    pub input_bindings: &'static [IoBinding],
    pub output_bindings: &'static [IoBinding],
    pub overridable_constants: &'static [usize],
    pub resource_bindings: &'static [usize],
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ShaderModuleInterface {
    pub overridable_constants: &'static [OverridableConstant],
    pub resource_bindings: &'static [ResourceBinding],
    pub entry_points: &'static [EntryPoint],
}
