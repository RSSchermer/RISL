#![cfg(feature = "test_iter_adapters_copied_try_fold")]

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
        let mut iter = A.iter().copied();
        let mut i = 0;

        while i < *PRECONSUME {
            iter.next();
            i += 1;
        }

        let mut calls = 0;

        let result: Result<u32, u32> = iter.try_fold(9, |acc, value| {
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
        let next_val = match iter.next() {
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
        InputValue { value: 1 },
        InputValue { value: 2 },
        InputValue { value: 3 },
        InputValue { value: 4 },
    ];

    // Complete in order with source values unchanged by the custom Clone.
    assert_eq!(
        runner.run(a, 0, 0).await?,
        Results {
            is_ok: 1,
            value: 91234,
            next_val: NONE
        }
    );
    // Short-circuit on the first item, consuming it but leaving the next copy available.
    assert_eq!(
        runner.run(a, 1, 0).await?,
        Results {
            is_ok: 0,
            value: 91,
            next_val: 2
        }
    );
    // Stateful closure calls stop in the middle and resume at the following item.
    assert_eq!(
        runner.run(a, 2, 0).await?,
        Results {
            is_ok: 0,
            value: 912,
            next_val: 3
        }
    );
    // Failing on the final item is an error even though no items remain.
    assert_eq!(
        runner.run(a, 4, 0).await?,
        Results {
            is_ok: 0,
            value: 91234,
            next_val: NONE
        }
    );
    // Preconsumption does not affect the closure's initial call count or accumulator.
    assert_eq!(
        runner.run(a, 1, 1).await?,
        Results {
            is_ok: 0,
            value: 92,
            next_val: 3
        }
    );
    // An exhausted iterator succeeds without invoking even an immediately failing closure.
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
