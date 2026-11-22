#![cfg(feature = "test_iter_adapters_fuse_next_back")]

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
    result: [u32; 7],
    shader: {
        let mut iter = NonFusedIter::new(A.as_ref()).fuse();
        let mut i = 0;

        while i < *FRONT {
            iter.next();
            i += 1;
        }

        let mut i = 0;

        while i < *BACK {
            iter.next_back();
            i += 1;
        }

        unsafe {
            let res = RESULT.as_mut_unchecked();
            let mut i = 0;

            while i < 7 {
                *res.get_unchecked_mut(i) = match iter.next_back() {
                    Some(value) => value,
                    None => NONE,
                };

                i += 1;
            }
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a = &[1, 2, NONE, 3, 4];
    let full = &[1, 2, 3, 4, 5];

    // After the first None item, further reverse iteration never returns Some.
    assert_eq!(
        runner.run(a, 0, 0).await?,
        [4, 3, NONE, NONE, NONE, NONE, NONE]
    );
    // Resume at the second item from the back.
    assert_eq!(
        runner.run(a, 0, 1).await?,
        [3, NONE, NONE, NONE, NONE, NONE, NONE]
    );
    // Resume after consuming an item from both ends.
    assert_eq!(
        runner.run(full, 1, 1).await?,
        [4, 3, 2, NONE, NONE, NONE, NONE]
    );
    // If the first item is a None, reverse iteration never returns Some later.
    assert_eq!(
        runner.run(a, 0, 3).await?,
        [NONE, NONE, NONE, NONE, NONE, NONE, NONE]
    );
    // If forward iteration reaches a None, the iterator fuses and never returns Some later, even
    // when resuming iteration from the back where there are still items.
    assert_eq!(
        runner.run(a, 3, 0).await?,
        [NONE, NONE, NONE, NONE, NONE, NONE, NONE]
    );
    // Without a gap, iteration consumes every item until the inner iterator is exhausted.
    assert_eq!(runner.run(full, 0, 0).await?, [5, 4, 3, 2, 1, NONE, NONE]);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
