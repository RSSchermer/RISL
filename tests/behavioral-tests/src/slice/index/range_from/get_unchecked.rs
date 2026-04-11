#![cfg(feature = "test_slice_index_range_from_get_unchecked")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: SliceRangeFromRunner,
    inputs: {
        START: u32 as Uniform<u32>,
        VALUES: [u32] as Storage<[u32]>,
    },
    result: u32,
    shader: {
        let values = VALUES.as_ref();

        let slice = unsafe {
            values.get_unchecked((*START as usize)..)
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
    let runner = SliceRangeFromRunner::init().await?;

    let values = vec![10u32, 20u32, 30u32, 40u32];

    // Valid range: [1..] -> 20 + 30 + 40 = 90
    assert_eq!(runner.run(1u32, values.clone()).await?, 90u32);
    // Valid range: [0..] -> 10 + 20 + 30 + 40 = 100
    assert_eq!(runner.run(0u32, values.clone()).await?, 100u32);
    // Valid range (end of slice): [4..] -> empty -> 99
    assert_eq!(runner.run(4u32, values.clone()).await?, 99u32);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
