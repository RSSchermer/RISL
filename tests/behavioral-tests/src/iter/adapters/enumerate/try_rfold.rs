#![cfg(feature = "test_iter_adapters_enumerate_try_rfold")]

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
    next_index: u32,
    next_value: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        BREAK_AT: u32 as Uniform<u32>,
        FRONT: u32 as Uniform<u32>,
        BACK: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut iter = A.iter().enumerate();
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

        let mut calls = 0;

        let result: Result<u32, u32> = iter.try_rfold(9, |acc, (index, value)| {
            let value = acc * 100 + index as u32 * 10 + *value;

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
        let (next_index, next_value) = match iter.next_back() {
            Some((index, value)) => (index as u32, *value),
            None => (NONE, NONE),
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                is_ok,
                value,
                next_index,
                next_value,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3, 4];

    // Complete in reverse order while retaining each item's original forward index.
    assert_eq!(
        runner.run(a, 0, 0, 0).await?,
        Results {
            is_ok: 1,
            value: 934231201,
            next_index: NONE,
            next_value: NONE
        }
    );
    // A failure on the last source item consumes it and resumes at the previous index.
    assert_eq!(
        runner.run(a, 1, 0, 0).await?,
        Results {
            is_ok: 0,
            value: 934,
            next_index: 2,
            next_value: 3
        }
    );
    // Closure state advances in reverse visitation order without revisiting the error item.
    assert_eq!(
        runner.run(a, 2, 0, 0).await?,
        Results {
            is_ok: 0,
            value: 93423,
            next_index: 1,
            next_value: 2
        }
    );
    // Failure on the final visited item remains an error after exhaustion.
    assert_eq!(
        runner.run(a, 4, 0, 0).await?,
        Results {
            is_ok: 0,
            value: 934231201,
            next_index: NONE,
            next_value: NONE
        }
    );
    // Mixed preconsumption requires both front count and remaining length for resumption.
    assert_eq!(
        runner.run(a, 1, 1, 1).await?,
        Results {
            is_ok: 0,
            value: 923,
            next_index: 1,
            next_value: 2
        }
    );
    // A successful partial fold visits just the remaining middle pairs in reverse order.
    assert_eq!(
        runner.run(a, 0, 1, 1).await?,
        Results {
            is_ok: 1,
            value: 92312,
            next_index: NONE,
            next_value: NONE
        }
    );
    // Exhaustion succeeds without invoking even an immediately failing closure.
    assert_eq!(
        runner.run(a, 1, 2, 2).await?,
        Results {
            is_ok: 1,
            value: 9,
            next_index: NONE,
            next_value: NONE
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
