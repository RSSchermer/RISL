#![cfg(feature = "test_enum_result")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;
use risl::gpu;

#[gpu]
fn validate_value(v: u32) -> Result<u32, u32> {
    if v > 10 { Ok(1) } else { Err(0) }
}

test_runner! {
    name: Runner,
    inputs: {
        VALUE: u32 as StorageMut<u32>,
    },
    result: u32,
    shader: {
        unsafe {
            *RESULT.as_mut_unchecked() = match validate_value(*VALUE.as_ref_unchecked()) {
                Ok(v) => v,
                Err(v) => v,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run(10).await?, 0);
    assert_eq!(runner.run(11).await?, 1);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
