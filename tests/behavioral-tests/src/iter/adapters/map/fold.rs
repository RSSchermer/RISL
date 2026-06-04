#![cfg(feature = "test_iter_adapters_map_fold")]

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

        let result = iter.fold(0, |acc, x| acc + x);

        unsafe {
            *RESULT.as_mut_unchecked() = result;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    // values: [1, 2, 3, 4] -> mapped: [2, 4, 6, 8] -> sum: 20
    assert_eq!(runner.run([1, 2, 3, 4]).await?, 20);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
