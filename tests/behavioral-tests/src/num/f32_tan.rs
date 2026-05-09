#![cfg(feature = "test_num_f32_tan")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;

test_runner! {
    name: Runner,
    inputs: {
        A: f32 as Uniform<f32>,
    },
    result: f32,
    shader: {
        let res = (*A).tan();

        unsafe {
            *RESULT.as_mut_unchecked() = res;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    // We use a small epsilon for comparisons as floating point results can vary slightly
    let result = runner.run(0.0).await?;
    assert!((result - 0.0).abs() < 1e-6);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
