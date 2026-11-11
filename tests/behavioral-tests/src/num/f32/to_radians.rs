#![cfg(feature = "test_num_f32_to_radians")]

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
        let res = (*A).to_radians();

        unsafe {
            *RESULT.as_mut_unchecked() = res;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    let deg = 180.0f32;
    let rad = deg.to_radians();
    assert!((runner.run(deg).await? - rad).abs() < 1e-6);

    let deg = 90.0f32;
    let rad = deg.to_radians();
    assert!((runner.run(deg).await? - rad).abs() < 1e-6);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
