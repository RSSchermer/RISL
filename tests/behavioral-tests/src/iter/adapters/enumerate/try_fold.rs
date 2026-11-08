#![cfg(feature = "test_iter_adapters_enumerate_try_fold")]

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

        let result: Result<u32, u32> = iter.try_fold(9, |acc, (index, value)| {
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
        let (next_index, next_value) = match iter.next() {
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

    // Complete with index/value order encoded in the accumulator and no remaining item.
    assert_eq!(
        runner.run(a, 0, 0, 0).await?,
        Results {
            is_ok: 1,
            value: 901122334,
            next_index: NONE,
            next_value: NONE
        }
    );
    // The front count advances even when the closure fails on the first item.
    assert_eq!(
        runner.run(a, 1, 0, 0).await?,
        Results {
            is_ok: 0,
            value: 901,
            next_index: 1,
            next_value: 2
        }
    );
    // Stateful closure calls stop in the middle and resume after the failing pair.
    assert_eq!(
        runner.run(a, 2, 0, 0).await?,
        Results {
            is_ok: 0,
            value: 90112,
            next_index: 2,
            next_value: 3
        }
    );
    // An error on the final item is not mistaken for successful exhaustion.
    assert_eq!(
        runner.run(a, 4, 0, 0).await?,
        Results {
            is_ok: 0,
            value: 901122334,
            next_index: NONE,
            next_value: NONE
        }
    );
    // Mixed preconsumption retains the front count through short-circuit and resumption.
    assert_eq!(
        runner.run(a, 1, 1, 1).await?,
        Results {
            is_ok: 0,
            value: 912,
            next_index: 2,
            next_value: 3
        }
    );
    // A successful partial fold keeps original indices rather than renumbering its input.
    assert_eq!(
        runner.run(a, 0, 1, 1).await?,
        Results {
            is_ok: 1,
            value: 91223,
            next_index: NONE,
            next_value: NONE
        }
    );
    // An exhausted source succeeds without invoking even an immediately failing closure.
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
