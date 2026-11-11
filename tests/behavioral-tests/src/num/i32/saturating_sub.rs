#![cfg(feature = "test_num_i32_saturating_sub")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: Runner,
    inputs: {
        A: i32 as Uniform<i32>,
        B: i32 as Uniform<i32>,
    },
    result: i32,
    shader: {
        let res = (*A).saturating_sub(*B);

        unsafe {
            *RESULT.as_mut_unchecked() = res;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    // Ordinary differences don't saturate for all four operand-sign combinations.
    assert_eq!(runner.run(4, 3).await?, 1);
    assert_eq!(runner.run(-4, -3).await?, -1);
    assert_eq!(runner.run(-4, 3).await?, -7);
    assert_eq!(runner.run(4, -3).await?, 7);

    // Ordinary numbers that overflow saturates rather than wrapping.
    assert_eq!(runner.run(i32::MAX, -1).await?, i32::MAX);
    assert_eq!(runner.run(i32::MIN, 1).await?, i32::MIN);
    assert_eq!(runner.run(0, i32::MIN).await?, i32::MAX);
    assert_eq!(runner.run(-1, i32::MIN).await?, i32::MAX);

    // Extreme numbers that overflow saturates rather than wrapping.
    assert_eq!(runner.run(i32::MAX, i32::MIN).await?, i32::MAX);
    assert_eq!(runner.run(i32::MIN, i32::MAX).await?, i32::MIN);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
