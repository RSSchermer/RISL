use proc_macro::TokenStream;
use quote::quote;
use risl_request::smi::{
    ArrayLayout, EntryPoint, Interpolate, InterpolationType, IoBinding, IoBindingType, MemoryUnit,
    MemoryUnitLayout, OverridableConstant, OverridableConstantType, ResourceBinding, ResourceType,
    Sampling, ShaderModuleInterface, ShaderStage, SizedBufferLayout, StorageTextureFormat,
    TexelType, UnsizedBufferLayout, UnsizedTailLayout,
};
use risl_request::{Request, request_shader_module_interface};

pub fn expand_shader_module_interface(input: TokenStream) -> TokenStream {
    match request_shader_module_interface(input) {
        Ok(Request::TokenStream(request)) => {
            let request = proc_macro2::TokenStream::from(request);

            quote! {
                const {
                    #request;

                    risl::smi::ShaderModuleInterface {
                        resource_bindings: &[],
                        overridable_constants: &[],
                        entry_points: &[],
                    }
                }
            }
            .into()
        }
        Ok(Request::Resolution(smi)) => smi_to_token_stream(&smi, &quote!(risl::smi)).into(),
        Err(err) => err.into_compile_error().into(),
    }
}

fn smi_to_token_stream(
    smi: &ShaderModuleInterface,
    mod_path: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let resource_bindings = smi
        .resource_bindings
        .iter()
        .map(|b| resource_binding_to_token_stream(b, mod_path));
    let overridable_constants = smi
        .overridable_constants
        .iter()
        .map(|c| overridable_constant_to_token_stream(c, mod_path));
    let entry_points = smi
        .entry_points
        .iter()
        .map(|ep| entry_point_to_token_stream(ep, mod_path));

    quote! {
        #mod_path::ShaderModuleInterface {
            resource_bindings: &[#(#resource_bindings),*],
            overridable_constants: &[#(#overridable_constants),*],
            entry_points: &[#(#entry_points),*],
        }
    }
}

fn resource_binding_to_token_stream(
    resource_binding: &ResourceBinding,
    mod_path: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let group = resource_binding.group;
    let binding = resource_binding.binding;
    let resource_type = resource_type_to_token_stream(&resource_binding.resource_type, mod_path);

    quote! {
        #mod_path::ResourceBinding {
            group: #group,
            binding: #binding,
            resource_type: #resource_type
        }
    }
}

fn overridable_constant_to_token_stream(
    overridable_constant: &OverridableConstant,
    mod_path: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let id = overridable_constant.id;
    let constant_type = constant_type_to_token_stream(overridable_constant.constant_type, mod_path);
    let required = overridable_constant.required;

    quote! {
        #mod_path::OverridableConstant {
            id: #id,
            constant_type: #constant_type,
            required: #required,
        }
    }
}

fn entry_point_to_token_stream(
    entry_point: &EntryPoint,
    mod_path: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let name = entry_point.name.as_str();
    let stage = shader_stage_to_token_stream(entry_point.stage, mod_path);
    let input_bindings = entry_point
        .input_bindings
        .iter()
        .map(|b| io_binding_to_token_stream(b, mod_path));
    let output_bindings = entry_point
        .output_bindings
        .iter()
        .map(|b| io_binding_to_token_stream(b, mod_path));
    let overridable_constants = entry_point.overridable_constants.iter();
    let resource_bindings = entry_point.resource_bindings.iter();

    quote! {
        #mod_path::EntryPoint {
            name: #name,
            stage: #stage,
            input_bindings: &[#(#input_bindings),*],
            output_bindings: &[#(#output_bindings),*],
            overridable_constants: &[#(#overridable_constants),*],
            resource_bindings: &[#(#resource_bindings),*],
        }
    }
}

fn resource_type_to_token_stream(
    resource_type: &ResourceType,
    mod_path: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    match resource_type {
        ResourceType::Texture1D(texel_type) => {
            let texel_type = texel_type_to_token_stream(*texel_type, mod_path);

            quote!(#mod_path::ResourceType::Texture1D(#texel_type))
        }
        ResourceType::Texture2D(texel_type) => {
            let texel_type = texel_type_to_token_stream(*texel_type, mod_path);

            quote!(#mod_path::ResourceType::Texture2D(#texel_type))
        }
        ResourceType::Texture3D(texel_type) => {
            let texel_type = texel_type_to_token_stream(*texel_type, mod_path);

            quote!(#mod_path::ResourceType::Texture3D(#texel_type))
        }
        ResourceType::Texture2DArray(texel_type) => {
            let texel_type = texel_type_to_token_stream(*texel_type, mod_path);

            quote!(#mod_path::ResourceType::Texture2DArray(#texel_type))
        }
        ResourceType::TextureCube(texel_type) => {
            let texel_type = texel_type_to_token_stream(*texel_type, mod_path);

            quote!(#mod_path::ResourceType::TextureCube(#texel_type))
        }
        ResourceType::TextureCubeArray(texel_type) => {
            let texel_type = texel_type_to_token_stream(*texel_type, mod_path);

            quote!(#mod_path::ResourceType::TextureCubeArray(#texel_type))
        }
        ResourceType::TextureMultisampled2D(texel_type) => {
            let texel_type = texel_type_to_token_stream(*texel_type, mod_path);

            quote!(#mod_path::ResourceType::TextureMultisampled2D(#texel_type))
        }
        ResourceType::TextureDepth2D => {
            quote!(#mod_path::ResourceType::TextureDepth2D)
        }
        ResourceType::TextureDepth2DArray => {
            quote!(#mod_path::ResourceType::TextureDepth2DArray)
        }
        ResourceType::TextureDepthCube => {
            quote!(#mod_path::ResourceType::TextureDepthCube)
        }
        ResourceType::TextureDepthCubeArray => {
            quote!(#mod_path::ResourceType::TextureDepthCubeArray)
        }
        ResourceType::TextureDepthMultisampled2D => {
            quote!(#mod_path::ResourceType::TextureDepthMultisampled2D)
        }
        ResourceType::StorageTexture1D(storage_format) => {
            let storage_format = storage_texture_format_to_token_stream(*storage_format, mod_path);

            quote!(#mod_path::ResourceType::StorageTexture1D(#storage_format))
        }
        ResourceType::StorageTexture2D(storage_format) => {
            let storage_format = storage_texture_format_to_token_stream(*storage_format, mod_path);

            quote!(#mod_path::ResourceType::StorageTexture2D(#storage_format))
        }
        ResourceType::StorageTexture2DArray(storage_format) => {
            let storage_format = storage_texture_format_to_token_stream(*storage_format, mod_path);

            quote!(#mod_path::ResourceType::StorageTexture2DArray(#storage_format))
        }
        ResourceType::StorageTexture3D(storage_format) => {
            let storage_format = storage_texture_format_to_token_stream(*storage_format, mod_path);

            quote!(#mod_path::ResourceType::StorageTexture3D(#storage_format))
        }
        ResourceType::FilteringSampler => {
            quote!(#mod_path::ResourceType::FilteringSampler)
        }
        ResourceType::NonFilteringSampler => {
            quote!(#mod_path::ResourceType::NonFilteringSampler)
        }
        ResourceType::ComparisonSampler => {
            quote!(#mod_path::ResourceType::ComparisonSampler)
        }
        ResourceType::Uniform(layout) => {
            let layout = sized_buffer_layout_to_token_stream(layout, mod_path);

            quote!(#mod_path::ResourceType::Uniform(#layout))
        }
        ResourceType::StorageRead(layout) => {
            let layout = unsized_buffer_layout_to_token_stream(layout, mod_path);

            quote!(#mod_path::ResourceType::StorageRead(#layout))
        }
        ResourceType::StorageReadWrite(layout) => {
            let layout = unsized_buffer_layout_to_token_stream(layout, mod_path);

            quote!(#mod_path::ResourceType::StorageReadWrite(#layout))
        }
    }
}

fn texel_type_to_token_stream(
    texel_type: TexelType,
    mod_path: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    match texel_type {
        TexelType::Float => {
            quote!(#mod_path::TexelType::Float)
        }
        TexelType::UnfilterableFloat => {
            quote!(#mod_path::TexelType::UnfilterableFloat)
        }
        TexelType::Integer => {
            quote!(#mod_path::TexelType::Integer)
        }
        TexelType::UnsignedInteger => {
            quote!(#mod_path::TexelType::UnsignedInteger)
        }
    }
}

fn storage_texture_format_to_token_stream(
    storage_format: StorageTextureFormat,
    mod_path: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    match storage_format {
        StorageTextureFormat::rgba8unorm => {
            quote!(#mod_path::StorageTextureFormat::rgba8unorm)
        }
        StorageTextureFormat::rgba8snorm => {
            quote!(#mod_path::StorageTextureFormat::rgba8snorm)
        }
        StorageTextureFormat::rgba8uint => {
            quote!(#mod_path::StorageTextureFormat::rgba8uint)
        }
        StorageTextureFormat::rgba8sint => {
            quote!(#mod_path::StorageTextureFormat::rgba8sint)
        }
        StorageTextureFormat::rgba16uint => {
            quote!(#mod_path::StorageTextureFormat::rgba16uint)
        }
        StorageTextureFormat::rgba16sint => {
            quote!(#mod_path::StorageTextureFormat::rgba16sint)
        }
        StorageTextureFormat::rgba16float => {
            quote!(#mod_path::StorageTextureFormat::rgba16float)
        }
        StorageTextureFormat::r32uint => {
            quote!(#mod_path::StorageTextureFormat::r32uint)
        }
        StorageTextureFormat::r32sint => {
            quote!(#mod_path::StorageTextureFormat::r32sint)
        }
        StorageTextureFormat::r32float => {
            quote!(#mod_path::StorageTextureFormat::r32float)
        }
        StorageTextureFormat::rg32uint => {
            quote!(#mod_path::StorageTextureFormat::rg32uint)
        }
        StorageTextureFormat::rg32sint => {
            quote!(#mod_path::StorageTextureFormat::rg32sint)
        }
        StorageTextureFormat::rg32float => {
            quote!(#mod_path::StorageTextureFormat::rg32float)
        }
        StorageTextureFormat::rgba32uint => {
            quote!(#mod_path::StorageTextureFormat::rgba32uint)
        }
        StorageTextureFormat::rgba32sint => {
            quote!(#mod_path::StorageTextureFormat::rgba32sint)
        }
        StorageTextureFormat::rgba32float => {
            quote!(#mod_path::StorageTextureFormat::rgba32float)
        }
    }
}

fn sized_buffer_layout_to_token_stream(
    layout: &SizedBufferLayout,
    mod_path: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let memory_units = layout
        .memory_units
        .iter()
        .map(|unit| memory_unit_to_token_stream(unit, mod_path));

    quote! {
        #mod_path::SizedBufferLayout {
            memory_units: &[#(#memory_units),*],
        }
    }
}

fn unsized_buffer_layout_to_token_stream(
    layout: &UnsizedBufferLayout,
    mod_path: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let sized_head = layout
        .sized_head
        .iter()
        .map(|unit| memory_unit_to_token_stream(unit, mod_path));
    let unsized_tail = if let Some(unsized_tail) = &layout.unsized_tail {
        let unsized_tail = unsized_tail_layout_to_token_stream(unsized_tail, mod_path);

        quote!(Some(#unsized_tail))
    } else {
        quote!(None)
    };

    quote! {
        #mod_path::UnsizedBufferLayout {
            sized_head: &[#(#sized_head),*],
            unsized_tail: #unsized_tail,
        }
    }
}

fn unsized_tail_layout_to_token_stream(
    layout: &UnsizedTailLayout,
    mod_path: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let offset = layout.offset;
    let element_layout = layout
        .element_layout
        .iter()
        .map(|unit| memory_unit_to_token_stream(unit, mod_path));
    let stride = layout.stride;

    quote! {
        #mod_path::UnsizedTailLayout {
            offset: #offset,
            element_layout: &[#(#element_layout),*],
            stride: #stride,
        }
    }
}

fn memory_unit_to_token_stream(
    memory_unit: &MemoryUnit,
    mod_path: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let offset = memory_unit.offset;
    let layout = memory_unit_layout_to_token_stream(&memory_unit.layout, mod_path);

    quote! {
        #mod_path::MemoryUnit {
            offset: #offset,
            layout: #layout
        }
    }
}

fn memory_unit_layout_to_token_stream(
    memory_unit_layout: &MemoryUnitLayout,
    mod_path: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    match memory_unit_layout {
        MemoryUnitLayout::Float => {
            quote!(#mod_path::MemoryUnitLayout::Float)
        }
        MemoryUnitLayout::FloatVector2 => {
            quote!(#mod_path::MemoryUnitLayout::FloatVector2)
        }
        MemoryUnitLayout::FloatVector3 => {
            quote!(#mod_path::MemoryUnitLayout::FloatVector3)
        }
        MemoryUnitLayout::FloatVector4 => {
            quote!(#mod_path::MemoryUnitLayout::FloatVector4)
        }
        MemoryUnitLayout::Integer => {
            quote!(#mod_path::MemoryUnitLayout::Integer)
        }
        MemoryUnitLayout::IntegerVector2 => {
            quote!(#mod_path::MemoryUnitLayout::IntegerVector2)
        }
        MemoryUnitLayout::IntegerVector3 => {
            quote!(#mod_path::MemoryUnitLayout::IntegerVector3)
        }
        MemoryUnitLayout::IntegerVector4 => {
            quote!(#mod_path::MemoryUnitLayout::IntegerVector4)
        }
        MemoryUnitLayout::UnsignedInteger => {
            quote!(#mod_path::MemoryUnitLayout::UnsignedInteger)
        }
        MemoryUnitLayout::UnsignedIntegerVector2 => {
            quote!(#mod_path::MemoryUnitLayout::UnsignedIntegerVector2)
        }
        MemoryUnitLayout::UnsignedIntegerVector3 => {
            quote!(#mod_path::MemoryUnitLayout::UnsignedIntegerVector3)
        }
        MemoryUnitLayout::UnsignedIntegerVector4 => {
            quote!(#mod_path::MemoryUnitLayout::UnsignedIntegerVector4)
        }
        MemoryUnitLayout::Matrix2x2 => {
            quote!(#mod_path::MemoryUnitLayout::Matrix2x2)
        }
        MemoryUnitLayout::Matrix2x3 => {
            quote!(#mod_path::MemoryUnitLayout::Matrix2x3)
        }
        MemoryUnitLayout::Matrix2x4 => {
            quote!(#mod_path::MemoryUnitLayout::Matrix2x4)
        }
        MemoryUnitLayout::Matrix3x2 => {
            quote!(#mod_path::MemoryUnitLayout::Matrix3x2)
        }
        MemoryUnitLayout::Matrix3x3 => {
            quote!(#mod_path::MemoryUnitLayout::Matrix3x3)
        }
        MemoryUnitLayout::Matrix3x4 => {
            quote!(#mod_path::MemoryUnitLayout::Matrix3x4)
        }
        MemoryUnitLayout::Matrix4x2 => {
            quote!(#mod_path::MemoryUnitLayout::Matrix4x2)
        }
        MemoryUnitLayout::Matrix4x3 => {
            quote!(#mod_path::MemoryUnitLayout::Matrix4x3)
        }
        MemoryUnitLayout::Matrix4x4 => {
            quote!(#mod_path::MemoryUnitLayout::Matrix4x4)
        }
        MemoryUnitLayout::Array(array_layout) => {
            let array_layout = array_layout_to_token_stream(array_layout, mod_path);

            quote!(#mod_path::MemoryUnitLayout::Array(#array_layout))
        }
    }
}

fn array_layout_to_token_stream(
    array_layout: &ArrayLayout,
    mod_path: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let element_layout = array_layout
        .element_layout
        .iter()
        .map(|unit| memory_unit_to_token_stream(unit, mod_path));
    let stride = array_layout.stride;
    let len = array_layout.len;

    quote! {
        #mod_path::ArrayLayout {
            element_layout: &[#(#element_layout),*],
            stride: #stride,
            len: #len,
        }
    }
}

fn constant_type_to_token_stream(
    constant_type: OverridableConstantType,
    mod_path: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    match constant_type {
        OverridableConstantType::Float => {
            quote!(#mod_path::OverridableConstantType::Float)
        }
        OverridableConstantType::Bool => {
            quote!(#mod_path::OverridableConstantType::Bool)
        }
        OverridableConstantType::SignedInteger => {
            quote!(#mod_path::OverridableConstantType::SignedInteger)
        }
        OverridableConstantType::UnsignedInteger => {
            quote!(#mod_path::OverridableConstantType::UnsignedInteger)
        }
    }
}

fn shader_stage_to_token_stream(
    shader_stage: ShaderStage,
    mod_path: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    match shader_stage {
        ShaderStage::Vertex => {
            quote!(#mod_path::ShaderStage::Vertex)
        }
        ShaderStage::Fragment => {
            quote!(#mod_path::ShaderStage::Fragment)
        }
        ShaderStage::Compute => {
            quote!(#mod_path::ShaderStage::Compute)
        }
    }
}

fn io_binding_to_token_stream(
    io_binding: &IoBinding,
    mod_path: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let location = io_binding.location;
    let binding_type = io_binding_type_to_token_stream(io_binding.binding_type, mod_path);

    let interpolate = if let Some(interpolate) = &io_binding.interpolate {
        let interpolate = interpolate_to_token_stream(interpolate, mod_path);

        quote!(Some(#interpolate))
    } else {
        quote!(None)
    };

    quote! {
        #mod_path::IoBinding {
            location: #location,
            binding_type: #binding_type,
            interpolate: #interpolate,
        }
    }
}

fn io_binding_type_to_token_stream(
    binding_type: IoBindingType,
    mod_path: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    match binding_type {
        IoBindingType::SignedInteger => {
            quote!(#mod_path::IoBindingType::SignedInteger)
        }
        IoBindingType::SignedIntegerVector2 => {
            quote!(#mod_path::IoBindingType::SignedIntegerVector2)
        }
        IoBindingType::SignedIntegerVector3 => {
            quote!(#mod_path::IoBindingType::SignedIntegerVector3)
        }
        IoBindingType::SignedIntegerVector4 => {
            quote!(#mod_path::IoBindingType::SignedIntegerVector4)
        }
        IoBindingType::UnsignedInteger => {
            quote!(#mod_path::IoBindingType::UnsignedInteger)
        }
        IoBindingType::UnsignedIntegerVector2 => {
            quote!(#mod_path::IoBindingType::UnsignedIntegerVector2)
        }
        IoBindingType::UnsignedIntegerVector3 => {
            quote!(#mod_path::IoBindingType::UnsignedIntegerVector3)
        }
        IoBindingType::UnsignedIntegerVector4 => {
            quote!(#mod_path::IoBindingType::UnsignedIntegerVector4)
        }
        IoBindingType::Float => {
            quote!(#mod_path::IoBindingType::Float)
        }
        IoBindingType::FloatVector2 => {
            quote!(#mod_path::IoBindingType::FloatVector2)
        }
        IoBindingType::FloatVector3 => {
            quote!(#mod_path::IoBindingType::FloatVector3)
        }
        IoBindingType::FloatVector4 => {
            quote!(#mod_path::IoBindingType::FloatVector4)
        }
        IoBindingType::HalfFloat => {
            quote!(#mod_path::IoBindingType::HalfFloat)
        }
        IoBindingType::HalfFloatVector2 => {
            quote!(#mod_path::IoBindingType::HalfFloatVector2)
        }
        IoBindingType::HalfFloatVector3 => {
            quote!(#mod_path::IoBindingType::HalfFloatVector3)
        }
        IoBindingType::HalfFloatVector4 => {
            quote!(#mod_path::IoBindingType::HalfFloatVector4)
        }
    }
}

fn interpolate_to_token_stream(
    interpolate: &Interpolate,
    mod_path: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let interpolation_type =
        interpolation_type_to_token_stream(interpolate.interpolation_type, mod_path);
    let sampling = if let Some(sampling) = interpolate.sampling {
        let sampling = sampling_to_token_stream(sampling, mod_path);

        quote!(Some(#sampling))
    } else {
        quote!(None)
    };

    quote! {
        #mod_path::Interpolate {
            interpolation_type: #interpolation_type,
            sampling: #sampling,
        }
    }
}

fn interpolation_type_to_token_stream(
    interpolation: InterpolationType,
    mod_path: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    match interpolation {
        InterpolationType::Perspective => {
            quote!(#mod_path::InterpolationType::Perspective)
        }
        InterpolationType::Linear => {
            quote!(#mod_path::InterpolationType::Linear)
        }
        InterpolationType::Flat => {
            quote!(#mod_path::InterpolationType::Flat)
        }
    }
}

fn sampling_to_token_stream(
    sampling: Sampling,
    mod_path: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    match sampling {
        Sampling::Center => {
            quote!(#mod_path::Sampling::Center)
        }
        Sampling::Centroid => {
            quote!(#mod_path::Sampling::Centroid)
        }
        Sampling::Sample => {
            quote!(#mod_path::Sampling::Sample)
        }
        Sampling::First => {
            quote!(#mod_path::Sampling::First)
        }
        Sampling::Either => {
            quote!(#mod_path::Sampling::Either)
        }
    }
}
