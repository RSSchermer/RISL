#![cfg(feature = "test_slice_index_range_to_inclusive_get_unchecked")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: SliceRangeToInclusiveRunner,
    inputs: {
        END: u32 as Uniform<u32>,
        VALUES: [u32] as Storage<[u32]>,
    },
    result: u32,
    shader: {
        let values = VALUES.as_ref();

        let slice = unsafe {
            values.get_unchecked(..=(*END as usize))
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
    let runner = SliceRangeToInclusiveRunner::init().await?;

    let values = vec![10u32, 20u32, 30u32, 40u32];

    // Valid range: [..=2] -> 10 + 20 + 30 = 60
    assert_eq!(runner.run(2u32, values.clone()).await?, 60u32);
    // Valid range: [..=3] -> 10 + 20 + 30 + 40 = 100
    assert_eq!(runner.run(3u32, values.clone()).await?, 100u32);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
