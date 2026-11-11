#![cfg(feature = "test_num_i32_checked_sub")]

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
        let (value, is_some) = match (*A).checked_sub(*B) {
            Some(value) => (value, 1),
            None => (0, 0),
        };

        unsafe {
            *RESULT.as_mut_unchecked() = [value, is_some];
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    // Successful differences for all four operand-sign combinations.
    assert_eq!(runner.run(3, 4).await?, [-1, 1]);
    assert_eq!(runner.run(-3, 4).await?, [-7, 1]);
    assert_eq!(runner.run(3, -4).await?, [7, 1]);
    assert_eq!(runner.run(-3, -4).await?, [1, 1]);

    // Exact boundaries fit.
    assert_eq!(runner.run(i32::MAX, 0).await?, [i32::MAX, 1]);
    assert_eq!(runner.run(i32::MIN, 0).await?, [i32::MIN, 1]);
    assert_eq!(runner.run(i32::MAX - 1, -1).await?, [i32::MAX, 1]);
    assert_eq!(runner.run(i32::MIN + 1, 1).await?, [i32::MIN, 1]);

    // Crossing either boundary returns None, including negating MIN through subtraction.
    assert_eq!(runner.run(i32::MIN, 1).await?, [0, 0]);
    assert_eq!(runner.run(i32::MAX, -1).await?, [0, 0]);
    assert_eq!(runner.run(0, i32::MIN).await?, [0, 0]);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
