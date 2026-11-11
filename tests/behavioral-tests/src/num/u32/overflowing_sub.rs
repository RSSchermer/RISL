#![cfg(feature = "test_num_u32_overflowing_sub")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: Runner,
    inputs: {
        A: u32 as Uniform<u32>,
        B: u32 as Uniform<u32>,
    },
    result: [u32; 2],
    shader: {
        let (value, overflow) = (*A).overflowing_sub(*B);

        unsafe {
            *RESULT.as_mut_unchecked() = [value, overflow as u32];
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    // Ordinary subtraction and the exact maximum fit.
    assert_eq!(runner.run(9, 4).await?, [5, 0]);
    assert_eq!(runner.run(u32::MAX, 0).await?, [4294967295, 0]);

    // Equal operands reach zero without underflow.
    assert_eq!(runner.run(0, 0).await?, [0, 0]);
    assert_eq!(runner.run(65536, 65536).await?, [0, 0]);

    // Underflow wraps at ordinary and extreme and reports underflow.
    assert_eq!(runner.run(3, 4).await?, [4294967295, 1]);
    assert_eq!(runner.run(0, 1).await?, [4294967295, 1]);
    assert_eq!(runner.run(0, u32::MAX).await?, [1, 1]);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
