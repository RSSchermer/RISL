#![cfg(feature = "test_slice_iter_mut_find")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: Runner,
    inputs: {
        VALUES: [u32] as StorageMut<[u32]>,
    },
    result: u32,
    shader: {
        unsafe {
            let result = VALUES.as_mut_unchecked().iter_mut().find(|value| {
                **value == 7
            });

            *RESULT.as_mut_unchecked() = match result {
                Some(v) => {
                    *v += 1;
                    *v
                },
                None => u32::MAX,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run(vec![1, 3, 5, 7, 9]).await?, 8);
    assert_eq!(runner.run(vec![1, 3, 5, 8, 9]).await?, u32::MAX);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
