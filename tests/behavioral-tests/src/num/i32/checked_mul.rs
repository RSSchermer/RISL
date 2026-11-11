#![cfg(feature = "test_num_i32_checked_mul")]

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
        let (value, is_some) = match (*A).checked_mul(*B) {
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

    // Successful products for all four operand-sign combinations.
    assert_eq!(runner.run(3, 4).await?, [12, 1]);
    assert_eq!(runner.run(-3, 4).await?, [-12, 1]);
    assert_eq!(runner.run(3, -4).await?, [-12, 1]);
    assert_eq!(runner.run(-3, -4).await?, [12, 1]);

    // Multiplication by one preserves either boundary in either operand position.
    assert_eq!(runner.run(i32::MAX, 1).await?, [i32::MAX, 1]);
    assert_eq!(runner.run(1, i32::MAX).await?, [i32::MAX, 1]);
    assert_eq!(runner.run(i32::MIN, 1).await?, [i32::MIN, 1]);
    assert_eq!(runner.run(1, i32::MIN).await?, [i32::MIN, 1]);

    // Negating MIN and multiplying large extremes must all return None.
    assert_eq!(runner.run(i32::MIN, -1).await?, [0, 0]);
    assert_eq!(runner.run(-1, i32::MIN).await?, [0, 0]);
    assert_eq!(runner.run(i32::MIN, i32::MIN).await?, [0, 0]);
    assert_eq!(runner.run(i32::MAX, i32::MAX).await?, [0, 0]);
    assert_eq!(runner.run(i32::MIN, i32::MAX).await?, [0, 0]);
    assert_eq!(runner.run(i32::MAX, i32::MIN).await?, [0, 0]);

    // The exact negative boundary fits; its positive counterpart and smaller negatives do not.
    assert_eq!(runner.run(-1073741824, 2).await?, [i32::MIN, 1]);
    assert_eq!(runner.run(2, -1073741824).await?, [i32::MIN, 1]);
    assert_eq!(runner.run(-2, 1073741824).await?, [i32::MIN, 1]);
    assert_eq!(runner.run(1073741824, 2).await?, [0, 0]);
    assert_eq!(runner.run(-1073741824, -2).await?, [0, 0]);
    assert_eq!(runner.run(-1073741825, 2).await?, [0, 0]);

    // Just below MAX succeeds; larger products return None.
    assert_eq!(runner.run(46340, 46340).await?, [2147395600, 1]);
    assert_eq!(runner.run(46341, 46341).await?, [0, 0]);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
