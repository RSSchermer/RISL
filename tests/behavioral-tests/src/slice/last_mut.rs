#![cfg(feature = "test_slice_last_mut")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

test_runner! {
    name: Runner,
    inputs: {
        END: u32 as Uniform<u32>,
        VALUES: [u32] as StorageMut<[u32]>,
    },
    result: u32,
    shader: {
        let mut values = unsafe { VALUES.as_mut_unchecked() };
        let end = *END as usize;

        let slice = if let Some(slice) = values.get_mut(..end) {
            slice
        } else {
            &mut []
        };

        let result = if let Some(last) = slice.last_mut() {
            *last += 1;
            *last
        } else {
            99
        };

        unsafe {
            *RESULT.as_mut_unchecked() = result;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let values = vec![10u32, 20u32, 30u32, 40u32];

    // Case 1: Sub-slice [] -> last: 99
    assert_eq!(runner.run(0, values.clone()).await?, 99);

    // Case 2: Sub-slice [10] -> last: 10 + 1 = 11
    assert_eq!(runner.run(1, values.clone()).await?, 11);

    // Case 3: Sub-slice [10, 20, 30] -> last: 30 + 1 = 31
    assert_eq!(runner.run(3, values.clone()).await?, 31);

    // Case 4: Sub-slice [10, 20, 30, 40] -> last: 40 + 1 = 41
    assert_eq!(runner.run(4, values.clone()).await?, 41);

    // Case 5: Out of bounds -> last: 99
    assert_eq!(runner.run(5, values.clone()).await?, 99);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
