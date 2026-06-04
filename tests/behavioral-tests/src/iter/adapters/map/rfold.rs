#![cfg(feature = "test_iter_adapters_map_rfold")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: Runner,
    inputs: {
        VALUES: [u32; 4] as Storage<[u32; 4]>,
    },
    result: u32,
    shader: {
        let iter = VALUES.iter().map(|x| *x * 2);

        let result = iter.rfold(0, |acc, x| acc * 10 + x);

        unsafe {
            *RESULT.as_mut_unchecked() = result;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    // values: [1, 2, 3, 4] -> mapped: [2, 4, 6, 8]
    // rfold(0, ...)
    // 1. x=8, acc=0 -> 8
    // 2. x=6, acc=8 -> 86
    // 3. x=4, acc=86 -> 864
    // 4. x=2, acc=864 -> 8642
    assert_eq!(runner.run([1, 2, 3, 4]).await?, 8642);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
