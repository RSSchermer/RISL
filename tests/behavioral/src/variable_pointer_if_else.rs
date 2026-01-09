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

#[derive(Clone, Copy, abi::Sized)]
#[gpu]
struct Values {
    a: u32,
    b: u32,
}

#[risl::shader::shader_module]
mod shader {
    use risl::prelude::*;

    use super::Values;

    #[resource(group = 0, binding = 0)]
    static CONDITION: Uniform<u32>;

    #[resource(group = 0, binding = 1)]
    static VALUES: Storage<Values>;

    #[resource(group = 0, binding = 2)]
    static RESULT: StorageMut<u32>;

    #[compute]
    fn main() {
        let values = VALUES.as_ref();

        let value = if *CONDITION == 1 {
            &values.a
        } else {
            &values.b
        };

        unsafe {
            *RESULT.as_mut_unchecked() = *value;
        }
    }
}

const SHADER: ShaderSource = shader_risl!(shader);

#[derive(empa::resource_binding::Resources)]
struct Resources<'a> {
    #[resource(binding = 0, visibility = "COMPUTE")]
    condition: Uniform<'a, u32>,
    #[resource(binding = 1, visibility = "COMPUTE")]
    values: Storage<'a, Values>,
    #[resource(binding = 2, visibility = "COMPUTE")]
    result: Storage<'a, u32, ReadWrite>,
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

    let condition = device.create_buffer(0u32, buffer::Usages::uniform_binding().and_copy_dst());
    let values = device.create_buffer(Values { a: 0, b: 1 }, buffer::Usages::storage_binding());
    let result = device.create_buffer(999u32, buffer::Usages::storage_binding().and_copy_src());
    let readback = device.create_buffer(999u32, buffer::Usages::map_read().and_copy_dst());

    let bind_group = device.create_bind_group(
        &bind_group_layout,
        Resources {
            condition: condition.uniform(),
            values: values.storage(),
            result: result.storage(),
        },
    );

    let queue = device.queue();

    let test_case = async move |case, expected| -> Result<(), Box<dyn Error>> {
        queue.write_buffer(condition.view(), &case);

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
            .copy_buffer_to_buffer(result.view(), readback.view())
            .finish();

        queue.submit(command_buffer);

        readback.map_read().await?;

        let result = *readback.mapped();

        readback.unmap();

        assert_eq!(result, expected);

        Ok(())
    };

    test_case(0, 1).await?;
    test_case(1, 0).await?;

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
