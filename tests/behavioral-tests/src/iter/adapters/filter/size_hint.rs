#![cfg(feature = "test_iter_adapters_filter_size_hint")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

const NONE: u32 = u32::MAX;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    lower_bound: u32,
    upper_bound: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        MIN: u32 as Uniform<u32>,
        FRONT: u32 as Uniform<u32>,
        BACK: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut iter = A.iter().filter(|value| **value % 2 == 0 && **value >= *MIN);
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

        let (lower_bound, upper_bound) = iter.size_hint();
        let upper_bound = match upper_bound {
            Some(value) => value as u32,
            None => NONE,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                lower_bound: lower_bound as u32,
                upper_bound,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3, 4, 5, 6];

    // The upper bound counts source items, not the three matching evens.
    assert_eq!(
        runner.run(a, 0, 0, 0).await?,
        Results {
            lower_bound: 0,
            upper_bound: 6,
        }
    );
    // One forward match consumes both the rejected 1 and the accepted 2.
    assert_eq!(
        runner.run(a, 0, 1, 0).await?,
        Results {
            lower_bound: 0,
            upper_bound: 4,
        }
    );
    // One backward match consumes only the accepted 6.
    assert_eq!(
        runner.run(a, 0, 0, 1).await?,
        Results {
            lower_bound: 0,
            upper_bound: 5,
        }
    );
    // Mixed consumption combines the actual source consumption at both ends.
    assert_eq!(
        runner.run(a, 0, 1, 1).await?,
        Results {
            lower_bound: 0,
            upper_bound: 3,
        }
    );
    // An all-rejecting predicate does not change an untouched source's bound.
    assert_eq!(
        runner.run(a, 7, 0, 0).await?,
        Results {
            lower_bound: 0,
            upper_bound: 6,
        }
    );
    // Searching once with no matches consumes the entire source.
    assert_eq!(
        runner.run(a, 7, 1, 0).await?,
        Results {
            lower_bound: 0,
            upper_bound: 0,
        }
    );
    // Consuming every match and probing the other end leaves zero bounds.
    assert_eq!(
        runner.run(a, 0, 3, 1).await?,
        Results {
            lower_bound: 0,
            upper_bound: 0,
        }
    );

    unknown_upper::run().await?;

    Ok(())
}

mod unknown_upper {
    use std::error::Error;

    use behavioral_tests_macros::test_runner;

    use super::{NONE, Results};

    test_runner! {
        name: Runner,
        inputs: {
            A: [u32] as Storage<[u32]>,
        },
        result: Results,
        shader: {
            let (lower_bound, upper_bound) = A.iter().cycle().filter(|value| **value % 2 == 0).size_hint();
            let upper_bound = match upper_bound {
                Some(value) => value as u32,
                None => NONE,
            };

            unsafe {
                *RESULT.as_mut_unchecked() = Results {
                    lower_bound: lower_bound as u32,
                    upper_bound,
                };
            }
        },
    }

    pub(super) async fn run() -> Result<(), Box<dyn Error>> {
        let runner = Runner::init().await?;
        let a: &[u32] = &[1, 2, 3, 4, 5, 6];

        // Filtering an unconsumed cycle discards its lower bound but preserves None.
        assert_eq!(
            runner.run(a).await?,
            Results {
                lower_bound: 0,
                upper_bound: NONE,
            }
        );

        Ok(())
    }
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
