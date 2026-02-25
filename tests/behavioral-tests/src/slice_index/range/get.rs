use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: SliceRangeRunner,
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
    let runner = SliceRangeRunner::init().await?;

    let values = vec![10u32, 20u32, 30u32, 40u32];

    // Valid range: [1..3] -> 20 + 30 = 50
    assert_eq!(runner.run(1u32, 3u32, values.clone()).await?, 50u32);
    // Single element range: [0..1] -> 10
    assert_eq!(runner.run(0u32, 1u32, values.clone()).await?, 10u32);
    // Invalid range (out of bounds): [1..5] -> 99
    assert_eq!(runner.run(1u32, 5u32, values.clone()).await?, 99u32);
    // Invalid range (start > end): [3..1] -> 99
    assert_eq!(runner.run(3u32, 1u32, values.clone()).await?, 99u32);
    // Empty range: [2..2] -> 99 (since empty returns &[] which gives 99)
    assert_eq!(runner.run(2u32, 2u32, values.clone()).await?, 99u32);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
