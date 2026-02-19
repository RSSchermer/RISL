#![feature(
    stmt_expr_attributes,
    maybe_uninit_array_assume_init,
    maybe_uninit_uninit_array_transpose,
    macro_metavar_expr
)]

#[macro_export]
macro_rules! gen_test_runner {
    (
        name: $name:ident,
        shader: { $($shader:tt)* },
        $(entry: $entry:expr,)?
        result: $res_ident:ident: $res_ty:ty,
        inputs: [
            $($in_ident:ident: $in_ty:ty as $in_kind:ident),* $(,)?
        ] $(,)?
    ) => {
        #[risl::shader::shader_module]
        mod shader_mod {
            use risl::prelude::*;
            // Bring parent items like user-defined GPU structs into scope
            use super::*;

            $(
                gen_test_runner!(@shader_resource $in_kind, $in_ident, $in_ty, ${index()});
            )*

            #[resource(group = 1, binding = 0)]
            static $res_ident: StorageMut<$res_ty>;

            #[compute]
            fn main() {
                $($shader)*
            }
        }

        const SHADER: empa::shader_module::ShaderSource = empa::shader_module::shader_risl!(shader_mod);

        type InputLayout = <InputResources<'static> as empa::resource_binding::Resources>::Layout;
        type ResultLayout = <ResultResources<'static> as empa::resource_binding::Resources>::Layout;

        #[derive(empa::resource_binding::Resources)]
        struct InputResources<'a> {
            $(
                #[resource(binding = ${index()}, visibility = "COMPUTE")]
                $in_ident: gen_test_runner!(@resource_ty $in_kind, 'a, $in_ty),
            )*
        }

        #[derive(empa::resource_binding::Resources)]
        struct ResultResources<'a> {
            #[resource(binding = 0, visibility = "COMPUTE")]
            __result: empa::buffer::Storage<'a, $res_ty, empa::access_mode::ReadWrite>,
        }

        struct $name {
            _instance: empa::native::Instance,
            device: empa::device::Device,
            queue: empa::device::Queue,
            input_bind_group_layout: empa::resource_binding::BindGroupLayout<InputLayout>,
            result_bind_group_layout: empa::resource_binding::BindGroupLayout<ResultLayout>,
            pipeline: empa::compute_pipeline::ComputePipeline<(InputLayout, ResultLayout)>,
        }

        impl $name {
            async fn new() -> Result<Self, Box<dyn std::error::Error>> {
                let instance = empa::native::Instance::default();
                let adapter = instance.get_adapter(Default::default())?;
                let device = adapter.request_device(&empa::device::DeviceDescriptor::default()).await?;
                let queue = device.queue();

                println!("{:#?}", SHADER);

                let shader = device.create_shader_module(&SHADER);
                let input_bind_group_layout = device.create_bind_group_layout::<InputLayout>();
                let result_bind_group_layout = device.create_bind_group_layout::<ResultLayout>();
                let pipeline_layout = device.create_pipeline_layout((&input_bind_group_layout, &result_bind_group_layout));

                let pipeline = device
                    .create_compute_pipeline(
                        &empa::compute_pipeline::ComputePipelineDescriptorBuilder::begin()
                            .layout(&pipeline_layout)
                            .compute(empa::compute_pipeline::ComputeStageBuilder::begin(&shader, "main").finish())
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

            async fn run(&self, $($in_ident: impl empa::buffer::AsBuffer<$in_ty>),*) -> Result<$res_ty, Box<dyn std::error::Error>> {
                use empa::command::ResourceBindingCommandEncoder as _;
                use empa::command::CommandEncoder as _;
                $(
                    let $in_ident: empa::buffer::Buffer<$in_ty, _> = self.device.create_buffer($in_ident, gen_test_runner!(@usage $in_kind));
                )*

                let result_buffer: empa::buffer::Buffer<$res_ty, _> = self.device.create_buffer(<$res_ty>::default(), empa::buffer::Usages::storage_binding().and_copy_src());
                let readback_buffer: empa::buffer::Buffer<$res_ty, _> = self.device.create_buffer(<$res_ty>::default(), empa::buffer::Usages::map_read().and_copy_dst());

                let input_bind_group = self.device.create_bind_group(
                    &self.input_bind_group_layout,
                    InputResources {
                        $(
                            $in_ident: gen_test_runner!(@resource_view $in_kind, $in_ident),
                        )*
                    },
                );

                let result_bind_group = self.device.create_bind_group(
                    &self.result_bind_group_layout,
                    ResultResources {
                        __result: result_buffer.storage::<empa::access_mode::ReadWrite>(),
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
                let res = readback_buffer.mapped().clone();
                readback_buffer.unmap();

                Ok(res)
            }
        }
    };

    (@shader_resource Uniform, $ident:ident, $ty:ty, $index:expr) => {
        #[resource(group = 0, binding = $index)]
        static $ident: Uniform<$ty>;
    };
    (@shader_resource Storage, $ident:ident, [$elem:ty], $index:expr) => {
        #[resource(group = 0, binding = $index)]
        static $ident: Storage<[$elem]>;
    };
    (@shader_resource Storage, $ident:ident, $ty:ty, $index:expr) => {
        #[resource(group = 0, binding = $index)]
        static $ident: Storage<$ty>;
    };
    (@shader_resource StorageMut, $ident:ident, [$elem:ty], $index:expr) => {
        #[resource(group = 0, binding = $index)]
        static $ident: StorageMut<[$elem]>;
    };
    (@shader_resource StorageMut, $ident:ident, $ty:ty, $index:expr) => {
        #[resource(group = 0, binding = $index)]
        static $ident: StorageMut<$ty>;
    };

    (@resource_ty Uniform, $lt:lifetime, $ty:ty) => { empa::buffer::Uniform<$lt, $ty> };
    (@resource_ty Storage, $lt:lifetime, [$elem:ty]) => { empa::buffer::Storage<$lt, [$elem]> };
    (@resource_ty Storage, $lt:lifetime, $ty:ty) => { empa::buffer::Storage<$lt, $ty> };
    (@resource_ty StorageMut, $lt:lifetime, [$elem:ty]) => { empa::buffer::Storage<$lt, [$elem], empa::access_mode::ReadWrite> };
    (@resource_ty StorageMut, $lt:lifetime, $ty:ty) => { empa::buffer::Storage<$lt, $ty, empa::access_mode::ReadWrite> };

    (@usage Uniform) => { empa::buffer::Usages::uniform_binding() };
    (@usage Storage) => { empa::buffer::Usages::storage_binding() };
    (@usage StorageMut) => { empa::buffer::Usages::storage_binding().and_copy_dst() };

    (@resource_view Uniform, $buf:ident) => { $buf.uniform() };
    (@resource_view Storage, $buf:ident) => { $buf.storage::<empa::access_mode::Read>() };
    (@resource_view StorageMut, $buf:ident) => { $buf.storage::<empa::access_mode::ReadWrite>() };
}

mod enum_result;
mod slice_get_index;
mod variable_pointer_if_else;
