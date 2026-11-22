#![cfg(feature = "test_iter_adapters_fuse_fold")]

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

        let value = iter.fold(9, |acc, value| acc * 10 + value);

        unsafe {
            *RESULT.as_mut_unchecked() = value;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32; 5] = &[1, 2, NONE, 3, 4];
    let full: &[u32; 5] = &[1, 2, 3, 4, 5];
    let first_none: &[u32; 5] = &[NONE, 1, 2, 3, 4];

    // Folding stops at the first gap.
    assert_eq!(runner.run(a, 0, 0).await?, 912);
    // Resume folding after a consumed front item.
    assert_eq!(runner.run(a, 1, 0).await?, 92);
    // Backward preconsumption leaves the forward prefix intact.
    assert_eq!(runner.run(a, 0, 1).await?, 912);
    // A front None leaves the accumulator unchanged despite later Some values.
    assert_eq!(runner.run(a, 3, 0).await?, 9);
    // A back None prevents folding the remaining forward items.
    assert_eq!(runner.run(a, 0, 3).await?, 9);
    // Without gaps, every item is folded.
    assert_eq!(runner.run(full, 0, 0).await?, 912345);
    // An initial None never calls the fold closure.
    assert_eq!(runner.run(first_none, 0, 0).await?, 9);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
