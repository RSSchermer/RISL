#![cfg(feature = "test_iter_adapters_enumerate_next_back")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

const NONE: u32 = u32::MAX;

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        FRONT: u32 as Uniform<u32>,
        BACK: u32 as Uniform<u32>,
    },
    result: [[u32; 2]; 6],
    shader: {
        let mut iter = A.iter().enumerate();
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
                *res.get_unchecked_mut(i) = match iter.next_back() {
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

    // Reverse visitation retains forward indices and explicitly reports repeated None.
    assert_eq!(
        runner.run(a, 0, 0).await?,
        [[3, 4], [2, 3], [1, 2], [0, 1], [NONE, NONE], [NONE, NONE]]
    );
    // Backward preconsumption shortens the source without shifting its indices.
    assert_eq!(
        runner.run(a, 0, 2).await?,
        [
            [1, 2],
            [0, 1],
            [NONE, NONE],
            [NONE, NONE],
            [NONE, NONE],
            [NONE, NONE]
        ]
    );
    // Front consumption must be included when computing indices from remaining length.
    assert_eq!(
        runner.run(a, 2, 0).await?,
        [
            [3, 4],
            [2, 3],
            [NONE, NONE],
            [NONE, NONE],
            [NONE, NONE],
            [NONE, NONE]
        ]
    );
    // Mixed consumption leaves the original indices of the middle items intact.
    assert_eq!(
        runner.run(a, 1, 1).await?,
        [
            [2, 3],
            [1, 2],
            [NONE, NONE],
            [NONE, NONE],
            [NONE, NONE],
            [NONE, NONE]
        ]
    );
    // Meeting in the middle exhausts both ends without duplicating an item.
    assert_eq!(runner.run(a, 2, 2).await?, [[NONE, NONE]; 6]);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
