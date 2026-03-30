#![cfg(feature = "test_slice_index_range_to_get_mut")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: SliceRangeToRunner,
    inputs: {
        END: u32 as Uniform<u32>,
        VALUES: [u32] as StorageMut<[u32]>,
    },
    result: u32,
    shader: {
        let values = unsafe {
            VALUES.as_mut_unchecked()
        };

        let slice = if let Some(slice) = values.get_mut(..(*END as usize)) {
            slice
        } else {
            &mut []
        };

        let sum = if slice.is_empty() {
            99
        } else {
            let mut s = 0;
            let mut i = 0;

            while i < slice.len() {
                unsafe {
                    s += slice.get_unchecked(i);
                }

                i += 1;
            }

            s
        };

        unsafe {
            *RESULT.as_mut_unchecked() = sum;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = SliceRangeToRunner::init().await?;

    let values = vec![10u32, 20u32, 30u32, 40u32];

    // Valid range: [..3] -> 10 + 20 + 30 = 60
    assert_eq!(runner.run(3u32, values.clone()).await?, 60u32);
    // Valid range: [..4] -> 10 + 20 + 30 + 40 = 100
    assert_eq!(runner.run(4u32, values.clone()).await?, 100u32);
    // Valid range: [..0] -> empty -> 99
    assert_eq!(runner.run(0u32, values.clone()).await?, 99u32);
    // Invalid range (out of bounds): [..5] -> 99
    assert_eq!(runner.run(5u32, values.clone()).await?, 99u32);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
