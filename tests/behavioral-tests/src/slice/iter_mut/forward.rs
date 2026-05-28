#![cfg(feature = "test_slice_iter_mut_forward")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: Runner,
    inputs: {
        VALUES: [u32] as StorageMut<[u32]>,
    },
    result: [u32; 5],
    shader: {
        unsafe {
            for value in VALUES.as_mut_unchecked() {
                *value += 1;
            }

            let res = RESULT.as_mut_unchecked();
            let mut i = 0;

            for value in VALUES.as_ref_unchecked() {
                *res.get_unchecked_mut(i) = *value;
                i += 1;
            }
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run(vec![10, 20, 30, 40]).await?, [11, 21, 31, 41, 0]);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
