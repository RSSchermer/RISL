#![cfg(feature = "test_num_f32_min")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;

test_runner! {
    name: Runner,
    inputs: {
        A: f32 as Uniform<f32>,
        B: f32 as Uniform<f32>,
    },
    result: f32,
    shader: {
        let res = f32::min(*A, *B);

        unsafe {
            *RESULT.as_mut_unchecked() = res;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run(10.0, 20.0).await?, 10.0);
    assert_eq!(runner.run(20.0, 10.0).await?, 10.0);
    assert_eq!(runner.run(-10.0, -20.0).await?, -20.0);
    assert_eq!(runner.run(-20.0, -10.0).await?, -20.0);
    assert_eq!(runner.run(-5.0, 5.0).await?, -5.0);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
