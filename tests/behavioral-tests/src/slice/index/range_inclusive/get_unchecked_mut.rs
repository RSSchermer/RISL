#![cfg(feature = "test_slice_index_range_inclusive_get_unchecked_mut")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: Runner,
    inputs: {
        START: u32 as Uniform<u32>,
        END: u32 as Uniform<u32>,
        VALUES: [u32] as StorageMut<[u32]>,
    },
    result: u32,
    shader: {
        let values = unsafe {
            VALUES.as_mut_unchecked()
        };

        let slice = unsafe {
            values.get_unchecked_mut((*START as usize)..=(*END as usize))
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
    let runner = Runner::init().await?;

    let values = vec![10, 20, 30, 40];

    // Valid range: [1..=2] -> 20 + 30 = 50
    assert_eq!(runner.run(1, 2, values.clone()).await?, 50);
    // Single element range: [0..=0] -> 10
    assert_eq!(runner.run(0, 0, values.clone()).await?, 10);
    // Valid range (end of slice): [1..=3] -> 20 + 30 + 40 = 90
    assert_eq!(runner.run(1, 3, values.clone()).await?, 90);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
