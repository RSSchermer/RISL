#![cfg(feature = "test_iter_adapters_flat_map_next")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

const NONE: u32 = u32::MAX;

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        B: [u32] as Storage<[u32]>,
        FRONT: u32 as Uniform<u32>,
        BACK: u32 as Uniform<u32>,
    },
    result: [u32; 7],
    shader: {
        let mut iter = A
            .iter()
            .flat_map(|len| unsafe { B.get_unchecked(..(*len as usize)) }.iter());
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
                *res.get_unchecked_mut(i) = match iter.next() {
                    Some(value) => *value,
                    None => NONE,
                };

                i += 1;
            }
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[0, 2, 0, 3, 0];
    let b: &[u32] = &[1, 2, 3];
    let empty: &[u32] = &[0, 0, 0];

    assert_eq!(runner.run(a, b, 0, 0).await?, [1, 2, 1, 2, 3, NONE, NONE]);
    // Resuming after a pre-consuming an item from the front removes the head item.
    assert_eq!(
        runner.run(a, b, 1, 0).await?,
        [2, 1, 2, 3, NONE, NONE, NONE]
    );
    // Resuming after a pre-consuming an item from the back removes the tail item.
    assert_eq!(
        runner.run(a, b, 0, 1).await?,
        [1, 2, 1, 2, NONE, NONE, NONE]
    );
    assert_eq!(
        runner.run(a, b, 3, 1).await?,
        [2, NONE, NONE, NONE, NONE, NONE, NONE]
    );
    // After exhaustion, iteration does not produce items.
    assert_eq!(runner.run(a, b, 6, 0).await?, [NONE; 7]);
    // All-empty inners don't produce items.
    assert_eq!(runner.run(empty, b, 0, 0).await?, [NONE; 7]);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
