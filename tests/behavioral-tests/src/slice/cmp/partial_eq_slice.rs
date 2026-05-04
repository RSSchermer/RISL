#![cfg(feature = "test_slice_cmp_partial_eq_slice")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    eq: u32,
    ne: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        START_A: u32 as Uniform<u32>,
        B: [u32] as Storage<[u32]>,
        START_B: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let a = A.as_ref();
        let b = B.as_ref();

        let start_a = *START_A as usize;
        let start_b = *START_B as usize;

        unsafe {
            let a = a.get_unchecked(start_a..);
            let b = b.get_unchecked(start_b..);

            let eq = a == b;
            let ne = a != b;

            *RESULT.as_mut_unchecked() = Results {
                eq: eq as u32,
                ne: ne as u32,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    // Case 1: Equal
    assert_eq!(
        runner.run(vec![1, 2, 3], 0, vec![1, 2, 3], 0).await?,
        Results { eq: 1, ne: 0 }
    );

    // Case 2: Not equal (different elements)
    assert_eq!(
        runner.run(vec![1, 2, 3], 0, vec![1, 2, 4], 0).await?,
        Results { eq: 0, ne: 1 }
    );

    // Case 3: Not equal (different length)
    assert_eq!(
        runner.run(vec![1, 2, 3], 0, vec![1, 2], 0).await?,
        Results { eq: 0, ne: 1 }
    );

    // Case 4: Equal (different offsets)
    assert_eq!(
        runner
            .run(vec![0, 1, 2, 3], 1, vec![9, 9, 9, 1, 2, 3], 3)
            .await?,
        Results { eq: 1, ne: 0 }
    );

    // Case 5: Empty (both empty)
    // The WebGPU spec does not allow us to create empty buffers, so we'll use the START offset to
    // create empty slices from non-empty buffers.
    assert_eq!(
        runner.run(vec![1], 1, vec![2, 3], 2).await?,
        Results { eq: 1, ne: 0 }
    );

    // Case 6: Empty (A empty)
    assert_eq!(
        runner.run(vec![1], 1, vec![1], 0).await?,
        Results { eq: 0, ne: 1 }
    );

    // Case 7: Empty (B empty)
    assert_eq!(
        runner.run(vec![1], 0, vec![1], 1).await?,
        Results { eq: 0, ne: 1 }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
