#![cfg(feature = "test_slice_index_range_from_get")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: Runner,
    inputs: {
        START: u32 as Uniform<u32>,
        VALUES: [u32] as Storage<[u32]>,
    },
    result: u32,
    shader: {
        let values = VALUES.as_ref();

        let slice = if let Some(slice) = values.get((*START as usize)..) {
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

    // Valid range: [1..] -> 20 + 30 + 40 = 90
    assert_eq!(runner.run(1, values.clone()).await?, 90);
    // Valid range: [0..] -> 10 + 20 + 30 + 40 = 100
    assert_eq!(runner.run(0, values.clone()).await?, 100);
    // Valid range (end of slice): [4..] -> empty -> 99
    assert_eq!(runner.run(4, values.clone()).await?, 99);
    // Invalid range (out of bounds): [5..] -> 99
    assert_eq!(runner.run(5, values.clone()).await?, 99);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
