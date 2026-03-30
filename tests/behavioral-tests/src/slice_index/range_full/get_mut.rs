#![cfg(feature = "test_slice_index_range_full_get_mut")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: SliceRangeFullRunner,
    inputs: {
        VALUES: [u32] as StorageMut<[u32]>,
    },
    result: u32,
    shader: {
        let values = unsafe {
            VALUES.as_mut_unchecked()
        };

        let slice = if let Some(slice) = values.get_mut(..) {
            slice
        } else {
            &mut []
        };

        let mut sum = 0;
        let mut i = 0;

        while i < slice.len() {
            unsafe {
                sum += slice.get_unchecked(i);
            }

            i += 1;
        }

        unsafe {
            *RESULT.as_mut_unchecked() = sum;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = SliceRangeFullRunner::init().await?;

    let values = vec![10u32, 20u32, 30u32, 40u32];

    // RangeFull should return the entire slice: 10 + 20 + 30 + 40 = 100
    assert_eq!(runner.run(values.clone()).await?, 100u32);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
