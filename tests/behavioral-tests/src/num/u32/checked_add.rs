#![cfg(feature = "test_num_u32_checked_add")]

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

    // Ordinary addition and zero return Some.
    assert_eq!(runner.run(3, 4).await?, [7, 1]);
    assert_eq!(runner.run(0, 0).await?, [0, 1]);

    // Exact MAX results are still Some.
    assert_eq!(runner.run(u32::MAX, 0).await?, [4294967295, 1]);
    assert_eq!(runner.run(0, u32::MAX).await?, [4294967295, 1]);
    assert_eq!(runner.run(4294967294, 1).await?, [4294967295, 1]);

    // Overflow returns None.
    assert_eq!(runner.run(u32::MAX, 1).await?, [0, 0]);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
