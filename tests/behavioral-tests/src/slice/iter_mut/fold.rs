#![cfg(feature = "test_slice_iter_mut_fold")]

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
        let result = unsafe {
            VALUES.as_mut_unchecked().iter_mut().fold(0u32, |acc, x| {
                *x += 1;
                acc + *x
            })
        };

        unsafe {
            *RESULT.as_mut_unchecked() = result;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    let values = vec![10, 20, 30, 40];
    // result should be (10+1) + (20+1) + (30+1) + (40+1) = 11 + 21 + 31 + 41 = 104
    assert_eq!(runner.run(values).await?, 104);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
