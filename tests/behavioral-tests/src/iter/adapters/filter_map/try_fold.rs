#![cfg(feature = "test_iter_adapters_filter_map_try_fold")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

const NONE: u32 = u32::MAX;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    is_ok: u32,
    value: u32,
    next_val: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        MIN: u32 as Uniform<u32>,
        BREAK_AT: u32 as Uniform<u32>,
        PRECONSUME: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut mapper_calls = 0;
        let mut iter = A.iter().filter_map(|value| {
            mapper_calls += 1;

            if *value >= *MIN && mapper_calls % 3 != 0 {
                Some(*value * 10 + mapper_calls)
            } else {
                None
            }
        });

        let mut i = 0;

        while i < *PRECONSUME {
            iter.next();
            i += 1;
        }

        let mut calls = 0;

        let result: Result<u32, u32> = iter.try_fold(9, |acc, value| {
            let value = acc * 100 + value;

            calls += 1;

            if calls == *BREAK_AT {
                Err(value)
            } else {
                Ok(value)
            }
        });

        let (is_ok, value) = match result {
            Ok(value) => (1, value),
            Err(value) => (0, value),
        };
        let next_val = match iter.next() {
            Some(value) => value,
            None => NONE,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results { is_ok, value, next_val };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3, 4, 5, 6];

    // Complete in source order using transformed values from the stateful mapper.
    assert_eq!(
        runner.run(a, 0, 0, 0).await?,
        Results {
            is_ok: 1,
            value: 911224455,
            next_val: NONE
        }
    );
    // Failing on the first mapped value consumes it but leaves the source resumable.
    assert_eq!(
        runner.run(a, 0, 1, 0).await?,
        Results {
            is_ok: 0,
            value: 911,
            next_val: 22
        }
    );
    // Resumption retains mapper state, skipping item 3 before producing 44.
    assert_eq!(
        runner.run(a, 0, 2, 0).await?,
        Results {
            is_ok: 0,
            value: 91122,
            next_val: 44
        }
    );
    // Rejected items do not advance fold-closure state; the final match can fail.
    assert_eq!(
        runner.run(a, 0, 4, 0).await?,
        Results {
            is_ok: 0,
            value: 911224455,
            next_val: NONE
        }
    );
    // Preconsumption and an early error both preserve the same mapping state.
    assert_eq!(
        runner.run(a, 0, 1, 1).await?,
        Results {
            is_ok: 0,
            value: 922,
            next_val: 44
        }
    );
    // A successful partial fold visits only the remaining mapped values.
    assert_eq!(
        runner.run(a, 0, 0, 1).await?,
        Results {
            is_ok: 1,
            value: 9224455,
            next_val: NONE
        }
    );
    // An all-rejected source cannot trigger an error in the fold closure.
    assert_eq!(
        runner.run(a, 7, 1, 0).await?,
        Results {
            is_ok: 1,
            value: 9,
            next_val: NONE
        }
    );
    // A rejected tail after the final mapped value leaves the seed intact.
    assert_eq!(
        runner.run(a, 0, 1, 4).await?,
        Results {
            is_ok: 1,
            value: 9,
            next_val: NONE
        }
    );
    // A fully exhausted source succeeds without calling either closure.
    assert_eq!(
        runner.run(a, 0, 1, 5).await?,
        Results {
            is_ok: 1,
            value: 9,
            next_val: NONE
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
