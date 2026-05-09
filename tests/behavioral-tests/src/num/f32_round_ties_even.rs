#![cfg(feature = "test_num_f32_round_ties_even")]

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
        let res = (*A).round_ties_even();

        unsafe {
            *RESULT.as_mut_unchecked() = res;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run(0.4).await?, 0.0);
    assert_eq!(runner.run(0.6).await?, 1.0);
    assert_eq!(runner.run(-0.4).await?, 0.0);
    assert_eq!(runner.run(-0.6).await?, -1.0);

    // Test halfway cases to see what it does (WGSL is ties-to-even)
    // 0.5 -> 0.0
    // 1.5 -> 2.0
    assert_eq!(runner.run(0.5).await?, 0.0);
    assert_eq!(runner.run(1.5).await?, 2.0);
    assert_eq!(runner.run(2.5).await?, 2.0);
    assert_eq!(runner.run(-0.5).await?, 0.0);
    assert_eq!(runner.run(-1.5).await?, -2.0);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
