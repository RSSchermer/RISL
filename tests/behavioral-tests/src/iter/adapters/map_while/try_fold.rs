#![cfg(feature = "test_iter_adapters_map_while_try_fold")]

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
    visited: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        STOP: u32 as Uniform<u32>,
        BREAK_AT: u32 as Uniform<u32>,
        PRECONSUME: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut visited = 0;
        let mut iter = A.iter().copied().map_while(|value| {
            visited = visited * 10 + value;

            if value == *STOP {
                None
            } else {
                Some(value * 2)
            }
        });

        let mut i = 0;

        while i < *PRECONSUME {
            iter.next();
            i += 1;
        }

        let mut calls = 0;

        let result: Result<u32, u32> = iter.try_fold(8, |acc, value| {
            let value = acc * 10 + value;

            calls += 1;

            if calls == *BREAK_AT {
                Err(value)
            } else {
                Ok(value)
            }
        });

        // Resume only after a fold error, not after the predicate may have returned None.
        let (is_ok, value, resumed_value) = match result {
            Ok(value) => (1, value, NONE),
            Err(value) => {
                let resumed_value = match iter.next() {
                    Some(value) => value,
                    None => NONE,
                };

                (0, value, resumed_value)
            }
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                is_ok,
                value,
                resumed_value,
                visited,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3];

    // A successful fold maps all items and preserves their forward order.
    assert_eq!(
        runner.run(a, 0, 0, 0).await?,
        Results {
            is_ok: 1,
            value: 8246,
            resumed_value: NONE,
            visited: 123,
        }
    );
    // Predicate termination on the first item succeeds with the initial accumulator.
    assert_eq!(
        runner.run(a, 1, 1, 0).await?,
        Results {
            is_ok: 1,
            value: 8,
            resumed_value: NONE,
            visited: 1,
        }
    );
    // A rejected middle item never reaches the fold closure's second call.
    assert_eq!(
        runner.run(a, 2, 2, 0).await?,
        Results {
            is_ok: 1,
            value: 82,
            resumed_value: NONE,
            visited: 12,
        }
    );
    // A first-item fold error leaves the adapter resumable at the second mapped item.
    assert_eq!(
        runner.run(a, 0, 1, 0).await?,
        Results {
            is_ok: 0,
            value: 82,
            resumed_value: 4,
            visited: 12,
        }
    );
    // A final-item fold error remains an error even though the source is now exhausted.
    assert_eq!(
        runner.run(a, 0, 3, 0).await?,
        Results {
            is_ok: 0,
            value: 8246,
            resumed_value: NONE,
            visited: 123,
        }
    );
    // Folding a partially consumed adapter preserves the predicate's existing state.
    assert_eq!(
        runner.run(a, 0, 0, 1).await?,
        Results {
            is_ok: 1,
            value: 846,
            resumed_value: NONE,
            visited: 123,
        }
    );
    // An exhausted source succeeds without invoking either closure again.
    assert_eq!(
        runner.run(a, 0, 1, 3).await?,
        Results {
            is_ok: 1,
            value: 8,
            resumed_value: NONE,
            visited: 123,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
