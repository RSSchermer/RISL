#![cfg(feature = "test_iter_adapters_fuse_last")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

use super::{NONE, NonFusedIter};

test_runner! {
    name: Runner,
    inputs: {
        A: [u32; 5] as Storage<[u32; 5]>,
        FRONT: u32 as Uniform<u32>,
        BACK: u32 as Uniform<u32>,
    },
    result: u32,
    shader: {
        let mut iter = NonFusedIter::new(A.as_ref()).fuse();
        let mut i = 0;

        while i < *FRONT {
            iter.next();
            i += 1;
        }

        i = 0;

        while i < *BACK {
            iter.next_back();
            i += 1;
        }

        let result = match iter.last() {
            Some(value) => value,
            None => NONE,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = result;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32; 5] = &[1, 2, NONE, 3, 4];
    let full: &[u32; 5] = &[1, 2, 3, 4, 5];
    let first_none: &[u32; 5] = &[NONE, 1, 2, 3, 4];

    // Last stops at the gap rather than returning the backmost Some.
    assert_eq!(runner.run(a, 0, 0).await?, 2);
    // Forward preconsumption preserves the final item before the gap.
    assert_eq!(runner.run(a, 1, 0).await?, 2);
    // Backward preconsumption that does not cross the gap does not change the last item.
    assert_eq!(runner.run(a, 0, 1).await?, 2);
    // Mixed preconsumption that does not cross the gap does not change the last item.
    assert_eq!(runner.run(a, 1, 1).await?, 2);
    // Starting at the gap returns None.
    assert_eq!(runner.run(a, 3, 0).await?, NONE);
    // If back preconsumption reaches the gap, then the iterator fuses and returns None, even
    // though there are still front items left.
    assert_eq!(runner.run(a, 0, 3).await?, NONE);
    // Without gaps, last reaches the final slot.
    assert_eq!(runner.run(full, 0, 0).await?, 5);
    // An initial None immediately fuses and finds no last item.
    assert_eq!(runner.run(first_none, 0, 0).await?, NONE);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
