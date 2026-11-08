#![cfg(feature = "test_iter_adapters_cloned_try_rfold")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

use super::{InputValue, Value};

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
        A: [InputValue] as Storage<[Value]>,
        BREAK_AT: u32 as Uniform<u32>,
        PRECONSUME: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut iter = A.iter().cloned();
        let mut i = 0;

        while i < *PRECONSUME {
            iter.next_back();
            i += 1;
        }

        let mut calls = 0;
        let result: Result<u32, u32> = iter.try_rfold(9, |acc, value| {
            let value = acc * 10 + value.value;
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
        let next_val = match iter.next_back() {
            Some(value) => value.value,
            None => NONE,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                is_ok,
                value,
                next_val,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[InputValue] = &[
        InputValue { value: 0 },
        InputValue { value: 1 },
        InputValue { value: 2 },
        InputValue { value: 3 },
    ];

    // Complete in reverse order, cloning each item exactly once.
    assert_eq!(
        runner.run(a, 0, 0).await?,
        Results {
            is_ok: 1,
            value: 94321,
            next_val: NONE
        }
    );
    // Short-circuit on the last source item and resume backward.
    assert_eq!(
        runner.run(a, 1, 0).await?,
        Results {
            is_ok: 0,
            value: 94,
            next_val: 3
        }
    );
    // Stateful closure calls stop in the middle without revisiting the failing item.
    assert_eq!(
        runner.run(a, 2, 0).await?,
        Results {
            is_ok: 0,
            value: 943,
            next_val: 2
        }
    );
    // An error on the final visited item is preserved after exhaustion.
    assert_eq!(
        runner.run(a, 4, 0).await?,
        Results {
            is_ok: 0,
            value: 94321,
            next_val: NONE
        }
    );
    // Backward preconsumption leaves closure state and the initial accumulator untouched.
    assert_eq!(
        runner.run(a, 1, 1).await?,
        Results {
            is_ok: 0,
            value: 93,
            next_val: 2
        }
    );
    // Exhaustion succeeds without invoking the closure.
    assert_eq!(
        runner.run(a, 1, 4).await?,
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
