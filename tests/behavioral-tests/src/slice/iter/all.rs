#![cfg(feature = "test_slice_iter_all")]

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
        let all_even = VALUES.iter().all(|value| {
            *value % 2 == 0
        });

        unsafe {
            *RESULT.as_mut_unchecked() = all_even as u32;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run(vec![2, 4, 6, 8]).await?, 1);
    assert_eq!(runner.run(vec![2, 4, 7, 8]).await?, 0);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
