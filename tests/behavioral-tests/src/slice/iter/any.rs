#![cfg(feature = "test_slice_iter_any")]

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
        let has_even = VALUES.iter().any(|value| {
            *value % 2 == 0
        });

        unsafe {
            *RESULT.as_mut_unchecked() = has_even as u32;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run(vec![1, 3, 5, 8]).await?, 1);
    assert_eq!(runner.run(vec![1, 3, 5, 7]).await?, 0);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
