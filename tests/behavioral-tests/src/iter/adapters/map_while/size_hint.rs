#![cfg(feature = "test_iter_adapters_map_while_size_hint")]

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
    next_value: u32,
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

        let (lower_bound, upper_bound) = iter.size_hint();

        let upper_bound = match upper_bound {
            Some(value) => value as u32,
            None => NONE,
        };

        let next_value = match iter.next() {
            Some(value) => value,
            None => NONE,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                lower_bound: lower_bound as u32,
                upper_bound,
                next_value,
                visited,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3];

    // The lower bound is zero; the source's upper bound is forwarded.
    assert_eq!(
        runner.run(a, 0, 0).await?,
        Results {
            lower_bound: 0,
            upper_bound: 3,
            next_value: 2,
            visited: 1,
        }
    );
    // Even an immediately rejecting predicate leaves the untouched upper bound unchanged.
    assert_eq!(
        runner.run(a, 1, 0).await?,
        Results {
            lower_bound: 0,
            upper_bound: 3,
            next_value: NONE,
            visited: 1,
        }
    );
    // Partial consumption reduces the upper bound.
    assert_eq!(
        runner.run(a, 0, 1).await?,
        Results {
            lower_bound: 0,
            upper_bound: 2,
            next_value: 4,
            visited: 12,
        }
    );
    // An exhausted source has exact zero bounds.
    assert_eq!(
        runner.run(a, 0, 3).await?,
        Results {
            lower_bound: 0,
            upper_bound: 0,
            next_value: NONE,
            visited: 123,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
