#![cfg(feature = "test_num_i32_checked_add")]

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
        let (value, is_some) = match (*A).checked_add(*B) {
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

    // Successful sums for all four operand-sign combinations.
    assert_eq!(runner.run(3, 4).await?, [7, 1]);
    assert_eq!(runner.run(-3, 4).await?, [1, 1]);
    assert_eq!(runner.run(3, -4).await?, [-1, 1]);
    assert_eq!(runner.run(-3, -4).await?, [-7, 1]);

    // Both exact boundaries are Some, whether preserved or reached by addition.
    assert_eq!(runner.run(i32::MAX, 0).await?, [i32::MAX, 1]);
    assert_eq!(runner.run(i32::MIN, 0).await?, [i32::MIN, 1]);
    assert_eq!(runner.run(i32::MAX - 1, 1).await?, [i32::MAX, 1]);
    assert_eq!(runner.run(i32::MIN + 1, -1).await?, [i32::MIN, 1]);

    // Overflow in either direction returns None rather than a wrapped value.
    assert_eq!(runner.run(i32::MAX, 1).await?, [0, 0]);
    assert_eq!(runner.run(i32::MIN, -1).await?, [0, 0]);
    assert_eq!(runner.run(-1, i32::MIN).await?, [0, 0]);
    assert_eq!(runner.run(i32::MAX, i32::MAX).await?, [0, 0]);
    assert_eq!(runner.run(i32::MIN, i32::MIN).await?, [0, 0]);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
