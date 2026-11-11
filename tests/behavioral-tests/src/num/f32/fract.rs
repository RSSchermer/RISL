#![cfg(feature = "test_num_f32_fract")]

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
        let res = (*A).fract();

        unsafe {
            *RESULT.as_mut_unchecked() = res;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run(1.5).await?, 0.5);
    assert_eq!(runner.run(1.0).await?, 0.0);
    assert_eq!(runner.run(0.5).await?, 0.5);
    assert_eq!(runner.run(0.0).await?, 0.0);
    // WGSL fract(-1.5) is -1.5 - floor(-1.5) = -1.5 - (-2.0) = 0.5
    // Rust f32::fract(-1.5) is -1.5 - trunc(-1.5) = -1.5 - (-1.0) = -0.5
    assert_eq!(runner.run(-1.5).await?, -0.5);
    assert_eq!(runner.run(-1.0).await?, 0.0);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
