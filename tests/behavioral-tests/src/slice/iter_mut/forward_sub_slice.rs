#![cfg(feature = "test_slice_iter_mut_forward_sub_slice")]

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
    result: [u32; 5],
    shader: {
        unsafe {
            let values = VALUES.as_mut_unchecked();

            if let Some(sub_slice) = values.get_mut((*START as usize)..(*END as usize)) {
                for value in sub_slice.iter_mut() {
                    *value += 1;
                }

                let res = RESULT.as_mut_unchecked();
                let mut i = 0;

                for value in sub_slice.iter() {
                    *res.get_unchecked_mut(i) = *value;
                    i += 1;
                }
            }
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    let values = vec![10, 20, 30, 40];

    // Valid full range
    assert_eq!(runner.run(0, 4, values.clone()).await?, [11, 21, 31, 41, 0]);
    // Valid multi-element range
    assert_eq!(runner.run(1, 3, values.clone()).await?, [21, 31, 0, 0, 0]);
    // Valid single element range
    assert_eq!(runner.run(0, 1, values.clone()).await?, [11, 0, 0, 0, 0]);
    // Empty range: [2..2] -> 0
    assert_eq!(runner.run(2, 2, values.clone()).await?, [0, 0, 0, 0, 0]);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
