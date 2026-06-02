#![cfg(feature = "test_slice_iter_mut_all")]

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
            let all_even = VALUES.as_mut_unchecked().iter_mut().all(|value| {
                *value % 2 == 0
            });

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
