#![cfg(feature = "test_iter_adapters_filter_map_size_hint")]

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
    mapper_calls: u32,
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
        let mut calls = 0;
        let mut iter = A.iter().filter_map(|value| {
            calls += 1;

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

        let (lower_bound, upper_bound) = iter.size_hint();
        let mapper_calls = calls;
        let upper_bound = match upper_bound {
            Some(value) => value as u32,
            None => NONE,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                lower_bound: lower_bound as u32,
                upper_bound,
                mapper_calls,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3, 4, 5, 6];

    // The upper bound counts source items without calling the mapper.
    assert_eq!(
        runner.run(a, 0, 0, 0).await?,
        Results {
            lower_bound: 0,
            upper_bound: 6,
            mapper_calls: 0,
        }
    );
    // One forward match maps the rejected 1 and accepted 2, with no hint calls.
    assert_eq!(
        runner.run(a, 0, 1, 0).await?,
        Results {
            lower_bound: 0,
            upper_bound: 4,
            mapper_calls: 2,
        }
    );
    // One backward match calls the mapper only for the accepted 6.
    assert_eq!(
        runner.run(a, 0, 0, 1).await?,
        Results {
            lower_bound: 0,
            upper_bound: 5,
            mapper_calls: 1,
        }
    );
    // Mixed consumption counts the actual source visits at both ends.
    assert_eq!(
        runner.run(a, 0, 1, 1).await?,
        Results {
            lower_bound: 0,
            upper_bound: 3,
            mapper_calls: 3,
        }
    );
    // An all-rejecting mapper stays unevaluated for an untouched source.
    assert_eq!(
        runner.run(a, 7, 0, 0).await?,
        Results {
            lower_bound: 0,
            upper_bound: 6,
            mapper_calls: 0,
        }
    );
    // Searching once with no matches maps and consumes the entire source.
    assert_eq!(
        runner.run(a, 7, 1, 0).await?,
        Results {
            lower_bound: 0,
            upper_bound: 0,
            mapper_calls: 6,
        }
    );
    // Exhaustion and an opposite-end probe do not add mapper calls.
    assert_eq!(
        runner.run(a, 0, 3, 1).await?,
        Results {
            lower_bound: 0,
            upper_bound: 0,
            mapper_calls: 6,
        }
    );

    unknown_upper::run().await?;

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
