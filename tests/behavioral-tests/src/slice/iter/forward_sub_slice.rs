#![cfg(feature = "test_slice_iter_forward_sub_slice")]

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

        let res = if let Some(sub_slice) = values.get((*START as usize)..(*END as usize)) {
            let mut sum = 0;

            for &value in sub_slice {
                sum += value;
            }

            sum
        } else {
            99
        };

        unsafe {
            *RESULT.as_mut_unchecked() = res;
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
    // Empty range: [2..2] -> 0
    assert_eq!(runner.run(2, 2, values.clone()).await?, 0);
    // Invalid range (out of bounds): [1..5] -> 99
    assert_eq!(runner.run(1, 5, values.clone()).await?, 99);
    // Invalid range (start > end): [3..1] -> 99
    assert_eq!(runner.run(3, 1, values.clone()).await?, 99);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
