#![cfg(feature = "test_slice_iter_mut_for_each")]

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
            VALUES.as_mut_unchecked().iter_mut().for_each(|value| {
                *value *= 2;
            });

            let mut sum = 0;

            for value in VALUES.as_ref_unchecked() {
                sum += *value;
            }

            *RESULT.as_mut_unchecked() = sum;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run(vec![10, 20, 30, 40]).await?, 200);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
