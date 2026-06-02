#![cfg(feature = "test_slice_iter_find_map")]

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
        let result = VALUES.iter().find_map(|value| {
            if *value > 5 {
                Some(*value * 2)
            } else {
                None
            }
        });

        unsafe {
            *RESULT.as_mut_unchecked() = match result {
                Some(v) => v,
                None => u32::MAX,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run(vec![1, 3, 5, 7, 9]).await?, 14);
    assert_eq!(runner.run(vec![1, 2, 3, 4, 5]).await?, u32::MAX);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
