#![cfg(feature = "test_iter_adapters_filter_map_next_back")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

const NONE: u32 = u32::MAX;

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        MIN: u32 as Uniform<u32>,
        FRONT: u32 as Uniform<u32>,
        BACK: u32 as Uniform<u32>,
    },
    result: [u32; 5],
    shader: {
        let mut iter = A.iter().filter_map(|value| {
            if *value % 2 == 0 && *value >= *MIN {
                Some(*value * 10 + 1)
            } else {
                None
            }
        });

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

            while i < 5 {
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
    let a: &[u32] = &[1, 2, 3, 4, 5, 6];

    // Map selected evens in reverse order and explicitly report repeated None.
    assert_eq!(runner.run(a, 0, 0, 0).await?, [61, 41, 21, NONE, NONE]);
    // The threshold rejects earlier evens as well as odd values.
    assert_eq!(runner.run(a, 5, 0, 0).await?, [61, NONE, NONE, NONE, NONE]);
    // Rejecting every item exhausts the source without yielding a value.
    assert_eq!(runner.run(a, 7, 0, 0).await?, [NONE; 5]);
    // Backward preconsumption skips a mapped item, not just one source item.
    assert_eq!(runner.run(a, 0, 0, 1).await?, [41, 21, NONE, NONE, NONE]);
    // Forward preconsumption removes the first mapped item from reverse traversal.
    assert_eq!(runner.run(a, 0, 1, 0).await?, [61, 41, NONE, NONE, NONE]);
    // Consuming both ends leaves only the middle mapped item.
    assert_eq!(runner.run(a, 0, 1, 1).await?, [41, NONE, NONE, NONE, NONE]);
    // Meeting and passing the other end cannot duplicate or resurrect items.
    assert_eq!(runner.run(a, 0, 1, 3).await?, [NONE; 5]);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
