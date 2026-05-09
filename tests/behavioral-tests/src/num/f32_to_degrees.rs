#![cfg(feature = "test_num_f32_to_degrees")]

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
        let res = (*A).to_degrees();

        unsafe {
            *RESULT.as_mut_unchecked() = res;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    let rad = std::f32::consts::PI;
    let deg = rad.to_degrees();
    assert!((runner.run(rad).await? - deg).abs() < 1e-5);

    let rad = std::f32::consts::FRAC_PI_2;
    let deg = rad.to_degrees();
    assert!((runner.run(rad).await? - deg).abs() < 1e-5);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
