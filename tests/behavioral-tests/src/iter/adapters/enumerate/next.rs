#![cfg(feature = "test_iter_adapters_enumerate_next")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

const NONE: u32 = u32::MAX;

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        SOURCE_SKIP: u32 as Uniform<u32>,
        FRONT: u32 as Uniform<u32>,
        BACK: u32 as Uniform<u32>,
    },
    result: [[u32; 2]; 6],
    shader: {
        let mut source = A.iter();
        let mut i = 0;

        while i < *SOURCE_SKIP {
            source.next();
            i += 1;
        }

        let mut iter = source.enumerate();
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

            while i < 6 {
                *res.get_unchecked_mut(i) = match iter.next() {
                    Some((index, value)) => [index as u32, *value],
                    None => [NONE, NONE],
                };

                i += 1;
            }
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3, 4];

    // Number unchanged values from zero, then explicitly report repeated None.
    assert_eq!(
        runner.run(a, 0, 0, 0).await?,
        [[0, 1], [1, 2], [2, 3], [3, 4], [NONE, NONE], [NONE, NONE]]
    );
    // Front consumption advances the index rather than restarting it.
    assert_eq!(
        runner.run(a, 0, 2, 0).await?,
        [
            [2, 3],
            [3, 4],
            [NONE, NONE],
            [NONE, NONE],
            [NONE, NONE],
            [NONE, NONE]
        ]
    );
    // Back consumption must not advance the front index.
    assert_eq!(
        runner.run(a, 0, 0, 2).await?,
        [
            [0, 1],
            [1, 2],
            [NONE, NONE],
            [NONE, NONE],
            [NONE, NONE],
            [NONE, NONE]
        ]
    );
    // Alternating ends preserves the original indices of the middle items.
    assert_eq!(
        runner.run(a, 0, 1, 1).await?,
        [
            [1, 2],
            [2, 3],
            [NONE, NONE],
            [NONE, NONE],
            [NONE, NONE],
            [NONE, NONE]
        ]
    );
    // Only items consumed after adapter construction contribute to its index.
    assert_eq!(
        runner.run(a, 2, 0, 0).await?,
        [
            [0, 3],
            [1, 4],
            [NONE, NONE],
            [NONE, NONE],
            [NONE, NONE],
            [NONE, NONE]
        ]
    );
    // Exhaustion before probing does not resurrect any indexed items.
    assert_eq!(runner.run(a, 0, 4, 0).await?, [[NONE, NONE]; 6]);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
