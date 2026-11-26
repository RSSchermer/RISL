#![cfg(feature = "test_iter_adapters_map_while_next")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

const NONE: u32 = u32::MAX;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    values: [u32; 5],
    visited: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        STOP: u32 as Uniform<u32>,
        PRECONSUME: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut visited = 0;
        let mut iter = A.iter().copied().map_while(|value| {
            visited = visited * 10 + value;

            if value == *STOP {
                None
            } else {
                Some(value * 2)
            }
        });

        let mut i = 0;

        while i < *PRECONSUME {
            iter.next();
            i += 1;
        }

        unsafe {
            let result = RESULT.as_mut_unchecked();
            let mut finished = false;

            let mut i = 0;

            while i < 5 {
                // MapWhile is not fused: pad the output without probing after None.
                *result.values.get_unchecked_mut(i) = if finished {
                    NONE
                } else {
                    match iter.next() {
                        Some(value) => value,
                        None => {
                            finished = true;

                            NONE
                        }
                    }
                };

                i += 1;
            }

            result.visited = visited;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3];

    // Accepted values are mapped in forward order until the source is exhausted.
    assert_eq!(
        runner.run(a, 0, 0).await?,
        Results {
            values: [2, 4, 6, NONE, NONE],
            visited: 123,
        }
    );
    // Rejecting the first item returns None immediately rather than searching for a match.
    assert_eq!(
        runner.run(a, 1, 0).await?,
        Results {
            values: [NONE, NONE, NONE, NONE, NONE],
            visited: 1,
        }
    );
    // A middle rejection stops before visiting the following source item.
    assert_eq!(
        runner.run(a, 2, 0).await?,
        Results {
            values: [2, NONE, NONE, NONE, NONE],
            visited: 12,
        }
    );
    // Rejecting the final item excludes it from the mapped outputs.
    assert_eq!(
        runner.run(a, 3, 0).await?,
        Results {
            values: [2, 4, NONE, NONE, NONE],
            visited: 123,
        }
    );
    // An exhausted source returns None without invoking the predicate again.
    assert_eq!(
        runner.run(a, 0, 3).await?,
        Results {
            values: [NONE, NONE, NONE, NONE, NONE],
            visited: 123,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
