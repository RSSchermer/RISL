#![cfg(feature = "test_adt_abi_struct_array_field_layout")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    before: u32,
    values: [u32; 2],
    after: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        BASE: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                before: *BASE,
                values: [*BASE + 1, *BASE + 2],
                after: *BASE + 3,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(
        runner.run(10).await?,
        Results {
            before: 10,
            values: [11, 12],
            after: 13,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
