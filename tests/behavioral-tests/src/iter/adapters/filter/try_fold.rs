#![cfg(feature = "test_iter_adapters_filter_try_fold")]

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
        let mut predicate_calls = 0;
        let mut iter = A.iter().filter(|value| {
            predicate_calls += 1;

            **value >= *MIN && predicate_calls % 3 != 0
        });
        let mut i = 0;

        while i < *PRECONSUME {
            iter.next();
            i += 1;
        }

        let mut calls = 0;

        let result: Result<u32, u32> = iter.try_fold(9, |acc, value| {
            let value = acc * 10 + *value;

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
            Some(value) => *value,
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

    // Complete in source order, folding only items accepted by the stateful predicate.
    assert_eq!(
        runner.run(a, 0, 0, 0).await?,
        Results {
            is_ok: 1,
            value: 91245,
            next_val: NONE
        }
    );
    // An error on the first accepted item consumes it without exhausting the source.
    assert_eq!(
        runner.run(a, 0, 1, 0).await?,
        Results {
            is_ok: 0,
            value: 91,
            next_val: 2
        }
    );
    // Resumption retains predicate state and skips the rejected third source item.
    assert_eq!(
        runner.run(a, 0, 2, 0).await?,
        Results {
            is_ok: 0,
            value: 912,
            next_val: 4
        }
    );
    // Rejected items do not increment fold-closure state; the final match can fail.
    assert_eq!(
        runner.run(a, 0, 4, 0).await?,
        Results {
            is_ok: 0,
            value: 91245,
            next_val: NONE
        }
    );
    // Preconsumption and an early error both preserve the same predicate state.
    assert_eq!(
        runner.run(a, 0, 1, 1).await?,
        Results {
            is_ok: 0,
            value: 92,
            next_val: 4
        }
    );
    // Completing a partial fold visits just the remaining accepted items.
    assert_eq!(
        runner.run(a, 0, 0, 1).await?,
        Results {
            is_ok: 1,
            value: 9245,
            next_val: NONE
        }
    );
    // Rejecting every item succeeds without invoking an immediately failing closure.
    assert_eq!(
        runner.run(a, 7, 1, 0).await?,
        Results {
            is_ok: 1,
            value: 9,
            next_val: NONE
        }
    );
    // After consuming all matches, the rejected tail likewise leaves the seed intact.
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
