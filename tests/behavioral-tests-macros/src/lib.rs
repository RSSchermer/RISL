use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{Error, Ident, Result, Token, Type, braced, parse_macro_input};

struct Input {
    name: Ident,
    ty: Type,
    binding: Binding,
}

enum Binding {
    Uniform(Type),
    Storage(Type),
    StorageMut(Type),
}

impl Parse for Binding {
    fn parse(input: ParseStream) -> Result<Self> {
        let lookahead = input.lookahead1();

        if lookahead.peek(Ident) {
            let ident: Ident = input.parse()?;
            let binding_kind = ident.to_string();

            input.parse::<Token![<]>()?;
            let gpu_ty: Type = input.parse()?;
            input.parse::<Token![>]>()?;

            match binding_kind.as_str() {
                "Uniform" => Ok(Binding::Uniform(gpu_ty)),
                "Storage" => Ok(Binding::Storage(gpu_ty)),
                "StorageMut" => Ok(Binding::StorageMut(gpu_ty)),
                _ => Err(Error::new(
                    ident.span(),
                    "expected Uniform, Storage, or StorageMut",
                )),
            }
        } else {
            Err(lookahead.error())
        }
    }
}

impl Parse for Input {
    fn parse(input: ParseStream) -> Result<Self> {
        let name: Ident = input.parse()?;
        input.parse::<Token![:]>()?;
        let ty: Type = input.parse()?;
        let _: Token![as] = input.parse()?;
        let binding: Binding = input.parse()?;

        Ok(Input { name, ty, binding })
    }
}

struct MacroInput {
    name: Ident,
    inputs: Punctuated<Input, Token![,]>,
    result_ty: Type,
    shader_body: syn::Block,
}

impl Parse for MacroInput {
    fn parse(input: ParseStream) -> Result<Self> {
        // name: <Ident>
        let _: Ident = {
            let ident: Ident = input.parse()?;
            if ident != "name" {
                return Err(Error::new(ident.span(), "expected 'name'"));
            }
            ident
        };
        input.parse::<Token![:]>()?;
        let name: Ident = input.parse()?;
        input.parse::<Token![,]>()?;

        // inputs: { <FIELD_SPEC>, ... }
        let _: Ident = {
            let ident: Ident = input.parse()?;
            if ident != "inputs" {
                return Err(Error::new(ident.span(), "expected 'inputs'"));
            }
            ident
        };
        input.parse::<Token![:]>()?;
        let content;
        braced!(content in input);
        let inputs = content.parse_terminated(Input::parse, Token![,])?;
        input.parse::<Token![,]>()?;

        // result: <ResultTy>
        let _: Ident = {
            let ident: Ident = input.parse()?;
            if ident != "result" {
                return Err(Error::new(ident.span(), "expected 'result'"));
            }
            ident
        };
        input.parse::<Token![:]>()?;
        let result_ty: Type = input.parse()?;
        input.parse::<Token![,]>()?;

        // shader: { /* tokens */ }
        let _: Ident = {
            let ident: Ident = input.parse()?;
            if ident != "shader" {
                return Err(Error::new(ident.span(), "expected 'shader'"));
            }
            ident
        };
        input.parse::<Token![:]>()?;
        let shader_body: syn::Block = input.parse()?;

        // Optional trailing comma after shader block
        let _ = input.parse::<Token![,]>();

        Ok(MacroInput {
            name,
            inputs,
            result_ty,
            shader_body,
        })
    }
}

/// Generates a GPU behavioral test runner.
///
/// This procedural macro reduces boilerplate when writing behavioral tests for RISL shaders.
///
/// # Example
///
/// ```rust
/// use behavioral_tests_macros::test_runner;
///
/// test_runner! {
///     name: SliceRunner,
///     inputs: {
///         INDEX: u32 as Uniform<u32>,
///         VALUES: [u32] as Storage<[u32]>,
///     },
///     result: u32,
///     shader: {
///         let values = VALUES.as_ref();
///         let value = if let Some(v) = values.get(*INDEX as usize) { *v } else { 99 };
///         unsafe { *RESULT.as_mut_unchecked() = value; }
///     },
/// }
///
/// # async fn _demo() -> Result<(), Box<dyn std::error::Error>> {
/// let runner = SliceRunner::init().await?;
///
/// assert_eq!(runner.run(0u32, vec![10u32, 20u32]).await?, 10u32);
/// assert_eq!(runner.run(1u32, vec![10u32, 20u32]).await?, 20u32);
/// assert_eq!(runner.run(3u32, vec![10u32, 20u32]).await?, 99u32);
/// # Ok(())
/// # }
/// ```
///
/// Macro input (order matters):
///
/// - `name`: name of the generated runner struct (e.g.: `SliceRunner`).
/// - `inputs`: ordered set of test inputs. Each input must match one of:
///
///   - `<IDENT>: <CpuTy> as Uniform<GpuTy>`: creates the input as a uniform binding.
///   - `<IDENT>: <CpuTy> as Storage<GpuTy>`: creates the input as a read-only storage binding.
///   - `<IDENT>: <CpuTy> as StorageMut<GpuTy>`: creates the input as a read-write storage binding.
///
///   For all binding types, the buffer is created from `CpuTy`, while the shader binding uses the
///   binding type parameterized by `GpuTy`. The `CpuTy` and `GpuTy` must be "binding compatible".
/// - `result`: the type of the test-result. Creates a read-write storage binding called `RESULT`
///   of the specified type. The type must implement [Clone] and must be storage binding compatible.
/// - `shader`: the test code. May access any of the input bindings by their `<IDENT>` or the result
///   as `RESULT`.
///
/// The macro generates a test-runner struct with the specified `name`. This test-runner maybe
/// instantiated via its async `init` method, which takes no arguments. It returns a `[Result] which
/// contains the test-runner, or an error if the test-runner could not be instantiated (for example,
/// because of a shader compilation error).
///
/// The test-runner holds an Empa compute-pipeline. It has a `run` method which may be invoked to
/// run the pipeline with a `1x1x1` dispatch size and a `1x1x1` workgroup size. The `run` method
/// takes a number of arguments equal to the number of `inputs` specified for the macro. Each
/// argument value's type must be - in order - [empa::buffer::AsBuffer] compatible with the type of
/// the corresponding input. `run` is an async function that returns a [Result] that holds a clone
/// of the value of the `RESULT` binding after the pipeline has successfully finished executing, or
/// an error if there was as problem executing the pipeline.
#[proc_macro]
pub fn test_runner(macro_input: TokenStream) -> TokenStream {
    let macro_input = parse_macro_input!(macro_input as MacroInput);
    let name = &macro_input.name;
    let result_ty = &macro_input.result_ty;
    let shader_body = &macro_input.shader_body;

    let mut shader_resources = quote! {};
    let mut input_resource_fields = quote! {};
    let mut input_resource_init = quote! {};
    let mut run_args = quote! {};
    let mut input_buffer_creations = quote! {};

    for (i, input) in macro_input.inputs.iter().enumerate() {
        let input_name = &input.name;
        let input_ty = &input.ty;
        let binding = i as u32;

        let binding_ty = match &input.binding {
            Binding::Uniform(gpu_ty) => {
                quote! { Uniform<#gpu_ty> }
            }
            Binding::Storage(gpu_ty) => {
                quote! { Storage<#gpu_ty> }
            }
            Binding::StorageMut(gpu_ty) => {
                quote! { StorageMut<#gpu_ty> }
            }
        };

        shader_resources.extend(quote! {
            #[resource(group = 0, binding = #binding)]
            static #input_name: #binding_ty;
        });

        let buffer_view_ty = match &input.binding {
            Binding::Uniform(_) => {
                quote! { empa::buffer::Uniform<'a, #input_ty> }
            }
            Binding::Storage(_) => {
                quote! { empa::buffer::Storage<'a, #input_ty> }
            }
            Binding::StorageMut(_) => {
                quote! {
                    empa::buffer::Storage<'a, #input_ty, empa::access_mode::ReadWrite>
                }
            }
        };

        input_resource_fields.extend(quote! {
            #[resource(binding = #binding, visibility = "COMPUTE")]
            #input_name: #buffer_view_ty,
        });

        let arg_name = Ident::new(&format!("input_{}", i), Span::call_site());
        run_args.extend(quote! {
            #arg_name: impl empa::buffer::AsBuffer<#input_ty>,
        });

        let usage = match &input.binding {
            Binding::Uniform(_) => {
                quote! { empa::buffer::Usages::uniform_binding() }
            }
            Binding::Storage(_) => {
                quote! { empa::buffer::Usages::storage_binding() }
            }
            Binding::StorageMut(_) => {
                quote! {
                    empa::buffer::Usages::storage_binding().and_copy_dst()
                }
            }
        };

        let buffer_name = Ident::new(&format!("buffer_{}", i), Span::call_site());
        input_buffer_creations.extend(quote! {
            let #buffer_name = self.device.create_buffer(#arg_name, #usage);
        });

        let buffer_view = match &input.binding {
            Binding::Uniform(_) => quote! { #buffer_name.uniform() },
            Binding::Storage(_) => {
                quote! { #buffer_name.storage::<empa::access_mode::Read>() }
            }
            Binding::StorageMut(_) => {
                quote! { #buffer_name.storage::<empa::access_mode::ReadWrite>() }
            }
        };

        input_resource_init.extend(quote! {
            #input_name: #buffer_view,
        });
    }

    // We explicitly set the span of the generated `shader_risl!` macro call to `Span::call_site()`.
    // This is done because `risl_request` currently relies on the span of the artifact-request
    // macro being globally unique in order to generate unique request IDs. If the `shader_risl!`
    // macro was assigned a `def_site` span, then multiple invocations of the `test_runner!` macro
    // would produce the same request ID, leading to incorrect shader-artifact resolution.
    //
    // This constraint is the primary reason this macro is implemented as a proc-macro; otherwise
    // it could be implemented as a "macro-rules" declarative macro. If we ever come up with a
    // better way to make and satisfy shader-artifact requests, then we may want to look into a
    // "macro-rules"-based implementation.
    let shader_risl_ident = Ident::new("shader_risl", Span::call_site());

    let expanded = quote! {
        #[risl::shader::shader_module]
        mod shader_mod {
            use risl::prelude::*;
            use super::*;

            #shader_resources

            #[resource(group = 1, binding = 0)]
            static RESULT: StorageMut<#result_ty>;

            #[compute]
            fn main() {
                #shader_body
            }
        }

        const SHADER: empa::shader_module::ShaderSource =
            empa::shader_module::#shader_risl_ident!(shader_mod);

        #[derive(empa::resource_binding::Resources)]
        struct InputResources<'a> {
            #input_resource_fields
        }

        type InputLayout =
            <InputResources<'static> as empa::resource_binding::Resources>::Layout;

        #[derive(empa::resource_binding::Resources)]
        struct ResultResources<'a> {
            #[resource(binding = 0, visibility = "COMPUTE")]
            result: empa::buffer::Storage<'a, #result_ty, empa::access_mode::ReadWrite>,
        }

        type ResultLayout =
            <ResultResources<'static> as empa::resource_binding::Resources>::Layout;

        struct #name {
            _instance: empa::native::Instance,
            device: empa::device::Device,
            queue: empa::device::Queue,
            input_bind_group_layout: empa::resource_binding::BindGroupLayout<InputLayout>,
            result_bind_group_layout: empa::resource_binding::BindGroupLayout<ResultLayout>,
            pipeline: empa::compute_pipeline::ComputePipeline<(InputLayout, ResultLayout)>,
        }

        impl #name {
            async fn init() -> Result<Self, Box<dyn std::error::Error>> {
                let instance = empa::native::Instance::default();
                let adapter = instance.get_adapter(Default::default())?;
                let device = adapter
                    .request_device(&empa::device::DeviceDescriptor::default())
                    .await?;
                let queue = device.queue();

                let shader_module = device.create_shader_module(&SHADER);

                let input_bind_group_layout =
                    device.create_bind_group_layout::<InputLayout>();
                let result_bind_group_layout =
                    device.create_bind_group_layout::<ResultLayout>();

                let pipeline_layout = device.create_pipeline_layout((
                    &input_bind_group_layout,
                    &result_bind_group_layout,
                ));

                let pipeline = device
                    .create_compute_pipeline(
                        &empa::compute_pipeline::ComputePipelineDescriptorBuilder::begin()
                            .layout(&pipeline_layout)
                            .compute(
                                empa::compute_pipeline::ComputeStageBuilder::begin(
                                    &shader_module,
                                    "main",
                                )
                                .finish(),
                            )
                            .finish(),
                    )
                    .await;

                Ok(Self {
                    _instance: instance,
                    device,
                    queue,
                    input_bind_group_layout,
                    result_bind_group_layout,
                    pipeline,
                })
            }

            async fn run(
                &self,
                #run_args
            ) -> Result<#result_ty, Box<dyn std::error::Error>> {
                use empa::command::ResourceBindingCommandEncoder as _;
                use empa::command::CommandEncoder as _;

                #input_buffer_creations

                let result_buffer = self.device.create_buffer(
                    <#result_ty>::default(),
                    empa::buffer::Usages::storage_binding().and_copy_src(),
                );
                let readback_buffer = self.device.create_buffer(
                    <#result_ty>::default(),
                    empa::buffer::Usages::map_read().and_copy_dst(),
                );

                let input_bind_group = self.device.create_bind_group(
                    &self.input_bind_group_layout,
                    InputResources {
                        #input_resource_init
                    },
                );

                let result_bind_group = self.device.create_bind_group(
                    &self.result_bind_group_layout,
                    ResultResources {
                        result: result_buffer.storage::<empa::access_mode::ReadWrite>(),
                    },
                );

                let command_buffer = self.device
                    .create_command_encoder()
                    .begin_compute_pass()
                    .set_pipeline(&self.pipeline)
                    .set_bind_groups((&input_bind_group, &result_bind_group))
                    .dispatch_workgroups(empa::command::DispatchWorkgroups {
                        count_x: 1,
                        count_y: 1,
                        count_z: 1,
                    })
                    .end()
                    .copy_buffer_to_buffer(result_buffer.view(), readback_buffer.view())
                    .finish();

                self.queue.submit(command_buffer);

                readback_buffer.map_read().await?;
                let result = readback_buffer.mapped().clone();
                readback_buffer.unmap();

                Ok(result)
            }
        }
    };

    expanded.into()
}
