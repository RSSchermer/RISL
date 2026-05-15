#![cfg(feature = "test_slice_index_range_get")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: Runner,
    inputs: {
        START: u32 as Uniform<u32>,
        END: u32 as Uniform<u32>,
        VALUES: [u32] as Storage<[u32]>,
    },
    result: u32,
    shader: {
        let values = VALUES.as_ref();

        let slice = if let Some(slice) = values.get((*START as usize)..(*END as usize)) {
            slice
        } else {
            &[]
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

    // Valid range: [1..3] -> 20 + 30 = 50
    assert_eq!(runner.run(1, 3, values.clone()).await?, 50);
    // Single element range: [0..1] -> 10
    assert_eq!(runner.run(0, 1, values.clone()).await?, 10);
    // Invalid range (out of bounds): [1..5] -> 99
    assert_eq!(runner.run(1, 5, values.clone()).await?, 99);
    // Invalid range (start > end): [3..1] -> 99
    assert_eq!(runner.run(3, 1, values.clone()).await?, 99);
    // Empty range: [2..2] -> 99 (since empty returns &[] which gives 99)
    assert_eq!(runner.run(2, 2, values.clone()).await?, 99);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
