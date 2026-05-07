#![cfg(feature = "test_iter_iterator_nth")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;
use risl::gpu;

use crate::iter::iterator::SimpleIter;

test_runner! {
    name: Runner,
    inputs: {
        START: u32 as Uniform<u32>,
        N: u32 as Uniform<u32>,
        VALUES: [u32; 4] as Storage<[u32; 4]>,
    },
    result: u32,
    shader: {
        let mut iter = SimpleIter {
            values: VALUES.clone(),
            index: *START as usize,
        };

        let res = iter.nth(*N as usize);

        unsafe {
            *RESULT.as_mut_unchecked() = res.unwrap_or(u32::MAX);
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    let values = [10, 20, 30, 40];

    // nth(0): next_val = 10
    assert_eq!(runner.run(0, 0, values).await?, 10);

    // nth(2): next_val = 30
    assert_eq!(runner.run(0, 2, values).await?, 30);

    // nth(3): next_val = 40
    assert_eq!(runner.run(0, 3, values).await?, 40);

    // nth(4): next_val = None (u32::MAX)
    assert_eq!(runner.run(0, 4, values).await?, u32::MAX);

    // nth(10) from start: next_val = None (u32::MAX)
    assert_eq!(runner.run(0, 10, values).await?, u32::MAX);

    // nth(2) from index 1: next_val = 40
    assert_eq!(runner.run(1, 2, values).await?, 40);

    // nth(1) from index 2: next_val = 40
    assert_eq!(runner.run(2, 1, values).await?, 40);

    // nth(2) from index 2: next_val = None (u32::MAX)
    assert_eq!(runner.run(2, 2, values).await?, u32::MAX);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
