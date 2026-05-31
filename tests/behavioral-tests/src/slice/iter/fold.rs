#![cfg(feature = "test_slice_iter_fold")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: Runner,
    inputs: {
        VALUES: [u32] as Storage<[u32]>,
    },
    result: u32,
    shader: {
        let result = VALUES.iter().fold(0u32, |acc, x| acc + *x);

        unsafe {
            *RESULT.as_mut_unchecked() = result;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run(vec![10, 20, 30, 40]).await?, 100);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
