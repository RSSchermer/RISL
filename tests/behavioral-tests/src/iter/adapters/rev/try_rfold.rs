#![cfg(feature = "test_iter_adapters_rev_try_rfold")]

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
    resumed_value: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        PRECONSUME: u32 as Uniform<u32>,
        BREAK_AT: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut iter = A.iter().copied().rev();

        let mut i = 0;

        while i < *PRECONSUME {
            iter.next_back();
            i += 1;
        }

        let mut calls = 0;

        let result: Result<u32, u32> = iter.try_rfold(8, |acc, value| {
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

        let resumed_value = match iter.next_back() {
            Some(value) => value,
            None => NONE,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                is_ok,
                value,
                resumed_value,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3];

    // A successful reverse-fold visits all source items in forward order and exhausts the iterator.
    assert_eq!(
        runner.run(a, 0, 0).await?,
        Results {
            is_ok: 1,
            value: 8123,
            resumed_value: NONE,
        }
    );
    // An error on the source's first item stops folding; iteration resumes at its second item.
    assert_eq!(
        runner.run(a, 0, 1).await?,
        Results {
            is_ok: 0,
            value: 81,
            resumed_value: 2,
        }
    );
    // An error on the middle item preserves the accumulated value and leaves the final item.
    assert_eq!(
        runner.run(a, 0, 2).await?,
        Results {
            is_ok: 0,
            value: 812,
            resumed_value: 3,
        }
    );
    // An exhausted iterator succeeds with the initial accumulator without calling the closure.
    assert_eq!(
        runner.run(a, 3, 1).await?,
        Results {
            is_ok: 1,
            value: 8,
            resumed_value: NONE,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
