#![cfg(feature = "test_num_f32_powf")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

test_runner! {
    name: Runner,
    inputs: {
        A: f32 as Uniform<f32>,
        B: f32 as Uniform<f32>,
    },
    result: f32,
    shader: {
        let res = (*A).powf(*B);

        unsafe {
            *RESULT.as_mut_unchecked() = res;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run(2.0, 3.0).await?, 8.0);
    assert_eq!(runner.run(9.0, 0.5).await?, 3.0);
    assert_eq!(runner.run(5.0, 0.0).await?, 1.0);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
