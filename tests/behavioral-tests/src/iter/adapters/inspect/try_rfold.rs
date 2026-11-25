#![cfg(feature = "test_iter_adapters_inspect_try_rfold")]

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
    next_value: u32,
    inspected: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        BREAK_AT: u32 as Uniform<u32>,
        PRECONSUME: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut inspected = 0;
        let mut iter = A.iter().copied().inspect(|value| {
            inspected = inspected * 10 + *value;
        });

        let mut i = 0;

        while i < *PRECONSUME {
            iter.next_back();
            i += 1;
        }

        let mut calls = 0;

        let result: Result<u32, u32> = iter.try_rfold(9, |acc, value| {
            let value = acc * 10 + value;

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

        let next_value = match iter.next_back() {
            Some(value) => value,
            None => NONE,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                is_ok,
                value,
                next_value,
                inspected,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3, 4];

    // A successful reverse fold inspects every item once in backward order.
    assert_eq!(
        runner.run(a, 0, 0).await?,
        Results {
            is_ok: 1,
            value: 94321,
            next_value: NONE,
            inspected: 4321,
        }
    );
    // An error on the first item from the back consumes it, inspects it, and resumes at the second
    // item from the back.
    assert_eq!(
        runner.run(a, 1, 0).await?,
        Results {
            is_ok: 0,
            value: 94,
            next_value: 3,
            inspected: 43,
        }
    );
    // A fold on an exhausted adapter succeeds without invoking either closure.
    assert_eq!(
        runner.run(a, 1, 4).await?,
        Results {
            is_ok: 1,
            value: 9,
            next_value: NONE,
            inspected: 4321,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
