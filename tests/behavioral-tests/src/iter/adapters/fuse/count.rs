#![cfg(feature = "test_iter_adapters_fuse_count")]

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

        let result = iter.count() as u32;

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

    // A fresh adapter counts only up to the first gap.
    assert_eq!(runner.run(a, 0, 0).await?, 2);
    // Forward preconsumption removes an item before the gap.
    assert_eq!(runner.run(a, 1, 0).await?, 1);
    // Backward preconsumption leaves the forward items intact.
    assert_eq!(runner.run(a, 0, 1).await?, 2);
    // Both ends can consume items without reaching the gap.
    assert_eq!(runner.run(a, 1, 1).await?, 1);
    // A front None prevents counting the later Some values.
    assert_eq!(runner.run(a, 3, 0).await?, 0);
    // A back None also discards the remaining forward items.
    assert_eq!(runner.run(a, 0, 3).await?, 0);
    // Without gaps, every item is counted.
    assert_eq!(runner.run(full, 0, 0).await?, 5);
    // An initial None yields zero despite later Some values.
    assert_eq!(runner.run(first_none, 0, 0).await?, 0);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
