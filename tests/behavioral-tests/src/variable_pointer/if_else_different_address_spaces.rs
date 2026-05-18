#![cfg(feature = "test_variable_pointer_if_else_different_address_spaces")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

test_runner! {
    name: Runner,
    inputs: {
        CONDITION: u32 as Uniform<u32>,
        A: u32 as Uniform<u32>,
        B: u32 as Storage<u32>,
    },
    result: u32,
    shader: {
        let value = if *CONDITION == 1 {
            A.as_ref()
        } else {
            B.as_ref()
        };

        unsafe {
            *RESULT.as_mut_unchecked() = *value;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run(0, 0, 1).await?, 1);
    assert_eq!(runner.run(1, 0, 1).await?, 0);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
