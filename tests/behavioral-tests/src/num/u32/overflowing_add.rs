#![cfg(feature = "test_num_u32_overflowing_add")]

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
        let (value, overflow) = (*A).overflowing_add(*B);

        unsafe {
            *RESULT.as_mut_unchecked() = [value, overflow as u32];
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    // Ordinary addition and zero do not overflow.
    assert_eq!(runner.run(3, 4).await?, [7, 0]);
    assert_eq!(runner.run(0, 0).await?, [0, 0]);

    // Reaching MAX exactly is not overflow.
    assert_eq!(runner.run(u32::MAX, 0).await?, [4294967295, 0]);
    assert_eq!(runner.run(0, u32::MAX).await?, [4294967295, 0]);
    assert_eq!(runner.run(4294967294, 1).await?, [4294967295, 0]);

    // Overflow wraps and reports overflow.
    assert_eq!(runner.run(u32::MAX, 1).await?, [0, 1]);
    assert_eq!(runner.run(u32::MAX, 2).await?, [1, 1]);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
