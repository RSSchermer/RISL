#![cfg(feature = "test_slice_iter_position")]

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
        let result = VALUES.iter().position(|value| {
            *value == 7
        });

        unsafe {
            *RESULT.as_mut_unchecked() = match result {
                Some(v) => v as u32,
                None => u32::MAX,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run(vec![1, 3, 5, 7, 9, 7, 2]).await?, 3);
    assert_eq!(runner.run(vec![1, 3, 5, 8, 9]).await?, u32::MAX);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
