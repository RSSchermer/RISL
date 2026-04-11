#![cfg(feature = "test_slice_iter_iter_mut")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: SliceRangeRunner,
    inputs: {
        VALUES: [u32] as StorageMut<[u32]>,
    },
    result: u32,
    shader: {
        unsafe {
            for value in VALUES.as_mut_unchecked() {
                *value += 1;
            }

            let mut sum = 0u32;

            for value in VALUES.as_ref_unchecked() {
                sum += value;
            }

            *RESULT.as_mut_unchecked() = sum;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = SliceRangeRunner::init().await?;

    assert_eq!(runner.run(vec![10u32, 20u32, 30u32, 40u32]).await?, 104u32);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
