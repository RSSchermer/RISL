#![cfg(feature = "test_iter_adapters_inspect_size_hint")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

const NONE: u32 = u32::MAX;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    initial_lower: u32,
    initial_upper: u32,
    remaining_lower: u32,
    remaining_upper: u32,
    next_value: u32,
    inspected: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        FRONT: u32 as Uniform<u32>,
        BACK: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut inspected = 0;
        let mut iter = A.iter().copied().inspect(|value| {
            inspected = inspected * 10 + *value;
        });

        let (initial_lower, initial_upper) = iter.size_hint();
        let initial_upper = match initial_upper {
            Some(value) => value as u32,
            None => NONE,
        };
        let mut i = 0;
        while i < *FRONT {
            iter.next();
            i += 1;
        }

        i = 0;
        while i < *BACK {
            iter.next_back();
            i += 1;
        }

        let (remaining_lower, remaining_upper) = iter.size_hint();
        let remaining_upper = match remaining_upper {
            Some(value) => value as u32,
            None => NONE,
        };
        let next_value = match iter.next() {
            Some(value) => value,
            None => NONE,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                initial_lower: initial_lower as u32,
                initial_upper,
                remaining_lower: remaining_lower as u32,
                remaining_upper,
                next_value,
                inspected,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3, 4];

    // Both hints are exact, and querying them does not inspect the iterator.
    assert_eq!(
        runner.run(a, 0, 0).await?,
        Results {
            initial_lower: 4,
            initial_upper: 4,
            remaining_lower: 4,
            remaining_upper: 4,
            next_value: 1,
            inspected: 1,
        }
    );
    // Front consumption updates the bounds while the hint calls add no inspect calls.
    assert_eq!(
        runner.run(a, 1, 0).await?,
        Results {
            initial_lower: 4,
            initial_upper: 4,
            remaining_lower: 3,
            remaining_upper: 3,
            next_value: 2,
            inspected: 12,
        }
    );
    // Back consumption updates the bounds while the hint calls add no inspect calls.
    assert_eq!(
        runner.run(a, 0, 2).await?,
        Results {
            initial_lower: 4,
            initial_upper: 4,
            remaining_lower: 2,
            remaining_upper: 2,
            next_value: 1,
            inspected: 431,
        }
    );
    // Exact exhaustion reports zero bounds.
    assert_eq!(
        runner.run(a, 2, 2).await?,
        Results {
            initial_lower: 4,
            initial_upper: 4,
            remaining_lower: 0,
            remaining_upper: 0,
            next_value: NONE,
            inspected: 1243,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
