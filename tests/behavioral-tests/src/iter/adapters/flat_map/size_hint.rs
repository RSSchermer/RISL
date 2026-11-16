#![cfg(feature = "test_iter_adapters_flat_map_size_hint")]

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
        B: [u32] as Storage<[u32]>,
        FRONT: u32 as Uniform<u32>,
        BACK: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut calls = 0;
        let mut iter = A.iter().flat_map(|len| {
            calls += 1;

            unsafe { B.get_unchecked(..(*len as usize)) }.iter()
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
    let a: &[u32] = &[0, 2, 0, 3, 0];
    let b: &[u32] = &[1, 2, 3];
    let single: &[u32] = &[2];
    let empty: &[u32] = &[0, 0, 0];

    // Variable inner lengths give no initial upper bound and stay unmapped.
    assert_eq!(
        runner.run(a, b, 0, 0).await?,
        Results {
            lower_bound: 0,
            upper_bound: NONE,
            mapper_calls: 0,
        }
    );
    // Advancing to the front inner discovers one guaranteed item.
    assert_eq!(
        runner.run(a, b, 1, 0).await?,
        Results {
            lower_bound: 1,
            upper_bound: NONE,
            mapper_calls: 2,
        }
    );
    // Back-advancing to the back inner discovers two guaranteed items.
    assert_eq!(
        runner.run(a, b, 0, 1).await?,
        Results {
            lower_bound: 2,
            upper_bound: NONE,
            mapper_calls: 2,
        }
    );
    // Advancing to both the front and back inner sums the guaranteed items.
    assert_eq!(
        runner.run(a, b, 1, 1).await?,
        Results {
            lower_bound: 3,
            upper_bound: NONE,
            mapper_calls: 4,
        }
    );
    // An exhausted outer source makes the pending inner's bounds exact.
    assert_eq!(
        runner.run(single, b, 1, 0).await?,
        Results {
            lower_bound: 1,
            upper_bound: 1,
            mapper_calls: 1,
        }
    );
    // Iterating past the last item results in an exact zero bound.
    assert_eq!(
        runner.run(a, b, 6, 0).await?,
        Results {
            lower_bound: 0,
            upper_bound: 0,
            mapper_calls: 5,
        }
    );
    // With all empty-inner iterators, one unsuccessful `next` discovers exhaustion and returns
    // exact zero bounds.
    assert_eq!(
        runner.run(empty, b, 1, 0).await?,
        Results {
            lower_bound: 0,
            upper_bound: 0,
            mapper_calls: 3,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
