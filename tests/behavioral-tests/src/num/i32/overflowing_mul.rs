#![cfg(feature = "test_num_i32_overflowing_mul")]

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
        let (value, overflow) = (*A).overflowing_mul(*B);

        unsafe {
            *RESULT.as_mut_unchecked() = [value, overflow as i32];
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    // All four operand-sign combinations without overflow.
    assert_eq!(runner.run(3, 4).await?, [12, 0]);
    assert_eq!(runner.run(-3, 4).await?, [-12, 0]);
    assert_eq!(runner.run(3, -4).await?, [-12, 0]);
    assert_eq!(runner.run(-3, -4).await?, [12, 0]);

    // Multiplication by one preserves either boundary in either operand position.
    assert_eq!(runner.run(i32::MAX, 1).await?, [i32::MAX, 0]);
    assert_eq!(runner.run(1, i32::MAX).await?, [i32::MAX, 0]);
    assert_eq!(runner.run(i32::MIN, 1).await?, [i32::MIN, 0]);
    assert_eq!(runner.run(1, i32::MIN).await?, [i32::MIN, 0]);

    // Negating MIN wraps and reports overflow.
    assert_eq!(runner.run(i32::MIN, -1).await?, [i32::MIN, 1]);
    assert_eq!(runner.run(-1, i32::MIN).await?, [i32::MIN, 1]);

    // Large products wrap and report overflow.
    assert_eq!(runner.run(i32::MIN, i32::MIN).await?, [0, 1]);
    assert_eq!(runner.run(i32::MAX, i32::MAX).await?, [1, 1]);
    assert_eq!(runner.run(i32::MIN, i32::MAX).await?, [i32::MIN, 1]);
    assert_eq!(runner.run(i32::MAX, i32::MIN).await?, [i32::MIN, 1]);

    // The exact negative boundary fits; its positive counterpart and smaller negatives do not.
    assert_eq!(runner.run(-1073741824, 2).await?, [i32::MIN, 0]);
    assert_eq!(runner.run(2, -1073741824).await?, [i32::MIN, 0]);
    assert_eq!(runner.run(-2, 1073741824).await?, [i32::MIN, 0]);
    assert_eq!(runner.run(1073741824, 2).await?, [i32::MIN, 1]);
    assert_eq!(runner.run(-1073741824, -2).await?, [i32::MIN, 1]);
    assert_eq!(runner.run(-1073741825, 2).await?, [2147483646, 1]);

    // Products on either side of the MAX boundary.
    assert_eq!(runner.run(46340, 46340).await?, [2147395600, 0]);
    assert_eq!(runner.run(46341, 46341).await?, [-2147479015, 1]);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
