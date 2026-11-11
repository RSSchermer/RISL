#![cfg(feature = "test_num_i32_overflowing_add")]

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
        let (value, overflow) = (*A).overflowing_add(*B);

        unsafe {
            *RESULT.as_mut_unchecked() = [value, overflow as i32];
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    // All four operand-sign combinations without overflow.
    assert_eq!(runner.run(3, 4).await?, [7, 0]);
    assert_eq!(runner.run(-3, 4).await?, [1, 0]);
    assert_eq!(runner.run(3, -4).await?, [-1, 0]);
    assert_eq!(runner.run(-3, -4).await?, [-7, 0]);

    // Exact boundaries, both unchanged and reached by addition, still fit.
    assert_eq!(runner.run(i32::MAX, 0).await?, [i32::MAX, 0]);
    assert_eq!(runner.run(i32::MIN, 0).await?, [i32::MIN, 0]);
    assert_eq!(runner.run(i32::MAX - 1, 1).await?, [i32::MAX, 0]);
    assert_eq!(runner.run(i32::MIN + 1, -1).await?, [i32::MIN, 0]);

    // Overflow in either direction returns the wrapped value and sets the flag.
    assert_eq!(runner.run(i32::MAX, 1).await?, [i32::MIN, 1]);
    assert_eq!(runner.run(i32::MIN, -1).await?, [i32::MAX, 1]);
    assert_eq!(runner.run(-1, i32::MIN).await?, [i32::MAX, 1]);
    assert_eq!(runner.run(i32::MAX, i32::MAX).await?, [-2, 1]);
    assert_eq!(runner.run(i32::MIN, i32::MIN).await?, [0, 1]);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
