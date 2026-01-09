use std::error::Error;

use empa::access_mode::ReadWrite;
use empa::buffer::{Buffer, Storage, Uniform};
use empa::command::{DispatchWorkgroups, ResourceBindingCommandEncoder};
use empa::compute_pipeline::{ComputePipelineDescriptorBuilder, ComputeStageBuilder};
use empa::device::DeviceDescriptor;
use empa::native::{AdapterOptions, Instance, PowerPreference};
use empa::shader_module::{ShaderSource, shader_risl};
use empa::{abi, buffer};
use futures::FutureExt;
use risl::gpu;

#[risl::shader::shader_module]
pub mod shader {
    use risl::prelude::*;

    #[resource(group = 0, binding = 0)]
    static VALUE: StorageMut<u32>;

    fn test(v: u32) -> Result<u32, u32> {
        if v > 10 { Ok(1) } else { Err(0) }
    }

    #[compute]
    fn main() {
        unsafe {
            *VALUE.as_mut_unchecked() = match test(*VALUE.as_ref_unchecked()) {
                Ok(v) => v,
                Err(v) => v,
            };
        }
    }
}

const SHADER: ShaderSource = shader_risl!(shader);

#[derive(empa::resource_binding::Resources)]
struct Resources<'a> {
    #[resource(binding = 0, visibility = "COMPUTE")]
    value: Storage<'a, u32, ReadWrite>,
}

type ResourceLayout = <Resources<'static> as empa::resource_binding::Resources>::Layout;

async fn run() -> Result<(), Box<dyn Error>> {
    let instance = Instance::default();

    let adapter = instance.get_adapter(Default::default())?;
    let device = adapter.request_device(&DeviceDescriptor::default()).await?;

    let shader = device.create_shader_module(&SHADER);

    let bind_group_layout = device.create_bind_group_layout::<ResourceLayout>();
    let pipeline_layout = device.create_pipeline_layout(&bind_group_layout);

    let pipeline = device
        .create_compute_pipeline(
            &ComputePipelineDescriptorBuilder::begin()
                .layout(&pipeline_layout)
                .compute(ComputeStageBuilder::begin(&shader, "main").finish())
                .finish(),
        )
        .await;

    let value = device.create_buffer(
        0,
        buffer::Usages::storage_binding()
            .and_copy_src()
            .and_copy_dst(),
    );
    let readback = device.create_buffer(999u32, buffer::Usages::map_read().and_copy_dst());

    let bind_group = device.create_bind_group(
        &bind_group_layout,
        Resources {
            value: value.storage(),
        },
    );

    let queue = device.queue();

    let test_case = async move |case, expected| -> Result<(), Box<dyn Error>> {
        queue.write_buffer(value.view(), &case);

        let command_buffer = device
            .create_command_encoder()
            .begin_compute_pass()
            .set_pipeline(&pipeline)
            .set_bind_groups(&bind_group)
            .dispatch_workgroups(DispatchWorkgroups {
                count_x: 1,
                count_y: 1,
                count_z: 1,
            })
            .end()
            .copy_buffer_to_buffer(value.view(), readback.view())
            .finish();

        queue.submit(command_buffer);

        readback.map_read().await?;

        let result = *readback.mapped();

        readback.unmap();

        assert_eq!(result, expected);

        Ok(())
    };

    test_case(10, 0).await?;
    test_case(11, 1).await?;

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
