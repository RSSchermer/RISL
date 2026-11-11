#![cfg(feature = "test_num_i32_overflowing_sub")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: Runner,
    inputs: {
        A: i32 as Uniform<i32>,
        B: i32 as Uniform<i32>,
    },
    result: [i32; 2],
    shader: {
        let (value, overflow) = (*A).overflowing_sub(*B);

        unsafe {
            *RESULT.as_mut_unchecked() = [value, overflow as i32];
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    // All four operand-sign combinations without overflow.
    assert_eq!(runner.run(3, 4).await?, [-1, 0]);
    assert_eq!(runner.run(-3, 4).await?, [-7, 0]);
    assert_eq!(runner.run(3, -4).await?, [7, 0]);
    assert_eq!(runner.run(-3, -4).await?, [1, 0]);

    // Exact boundaries fit.
    assert_eq!(runner.run(i32::MAX, 0).await?, [i32::MAX, 0]);
    assert_eq!(runner.run(i32::MIN, 0).await?, [i32::MIN, 0]);
    assert_eq!(runner.run(i32::MAX - 1, -1).await?, [i32::MAX, 0]);
    assert_eq!(runner.run(i32::MIN + 1, 1).await?, [i32::MIN, 0]);

    // Crossing either boundary returns the wrapped value and reports overflow.
    assert_eq!(runner.run(i32::MIN, 1).await?, [i32::MAX, 1]);
    assert_eq!(runner.run(i32::MAX, -1).await?, [i32::MIN, 1]);
    assert_eq!(runner.run(0, i32::MIN).await?, [i32::MIN, 1]);

    // Subtracting the largest positive value from MIN wraps to one.
    assert_eq!(runner.run(i32::MIN, i32::MAX).await?, [1, 1]);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
