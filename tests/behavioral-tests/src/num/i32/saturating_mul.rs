#![cfg(feature = "test_num_i32_saturating_mul")]

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
        let res = (*A).saturating_mul(*B);

        unsafe {
            *RESULT.as_mut_unchecked() = res;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    // Ordinary products don't saturate for all four operand-sign combinations.
    assert_eq!(runner.run(3, 4).await?, 12);
    assert_eq!(runner.run(-3, 4).await?, -12);
    assert_eq!(runner.run(3, -4).await?, -12);
    assert_eq!(runner.run(-3, -4).await?, 12);

    // Multiplication by one preserves either limit in either operand position.
    assert_eq!(runner.run(i32::MAX, 1).await?, i32::MAX);
    assert_eq!(runner.run(1, i32::MAX).await?, i32::MAX);
    assert_eq!(runner.run(i32::MIN, 1).await?, i32::MIN);
    assert_eq!(runner.run(1, i32::MIN).await?, i32::MIN);

    // Negating MIN saturates at MAX instead of wrapping back to MIN.
    assert_eq!(runner.run(i32::MIN, -1).await?, i32::MAX);
    assert_eq!(runner.run(-1, i32::MIN).await?, i32::MAX);

    // Extreme products saturate.
    assert_eq!(runner.run(i32::MIN, i32::MIN).await?, i32::MAX);
    assert_eq!(runner.run(i32::MAX, i32::MAX).await?, i32::MAX);
    assert_eq!(runner.run(i32::MIN, i32::MAX).await?, i32::MIN);
    assert_eq!(runner.run(i32::MAX, i32::MIN).await?, i32::MIN);

    // Ordinary products that overflow saturate.
    assert_eq!(runner.run(-1073741824, 2).await?, i32::MIN);
    assert_eq!(runner.run(1073741824, 2).await?, i32::MAX);
    assert_eq!(runner.run(-1073741824, -2).await?, i32::MAX);
    assert_eq!(runner.run(-1073741825, 2).await?, i32::MIN);
    assert_eq!(runner.run(46340, 46340).await?, 2147395600);
    assert_eq!(runner.run(46341, 46341).await?, i32::MAX);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
