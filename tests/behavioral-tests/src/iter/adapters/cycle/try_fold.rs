#![cfg(feature = "test_iter_adapters_cycle_try_fold")]

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
        LEN: u32 as Uniform<u32>,
        BREAK_AT: u32 as Uniform<u32>,
        PRECONSUME: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let source = unsafe { A.get_unchecked(..(*LEN as usize)) };
        let mut iter = source.iter().cycle();
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
    let a: &[u32] = &[1, 2, 3];

    // Stop on the first item, consuming it but leaving the following item available.
    assert_eq!(
        runner.run(a, 3, 1, 0).await?,
        Results {
            is_ok: 0,
            value: 91,
            next_val: 2,
        }
    );
    // A failure at the current cycle's end still permits resumption in a new cycle.
    assert_eq!(
        runner.run(a, 3, 3, 0).await?,
        Results {
            is_ok: 0,
            value: 9123,
            next_val: 1,
        }
    );
    // Carry both closure state and accumulator into the first cloned iteration.
    assert_eq!(
        runner.run(a, 3, 4, 0).await?,
        Results {
            is_ok: 0,
            value: 91231,
            next_val: 2,
        }
    );
    // Preserve an error on the last item of that cloned iteration.
    assert_eq!(
        runner.run(a, 3, 6, 0).await?,
        Results {
            is_ok: 0,
            value: 9123123,
            next_val: 1,
        }
    );
    // Continue into later resets without losing call count or visitation order.
    assert_eq!(
        runner.run(a, 3, 7, 0).await?,
        Results {
            is_ok: 0,
            value: 91231231,
            next_val: 2,
        }
    );
    // Finish a partial current iteration before restarting the full original source.
    assert_eq!(
        runner.run(a, 3, 4, 1).await?,
        Results {
            is_ok: 0,
            value: 92312,
            next_val: 3,
        }
    );
    // An exhausted current iteration does not imply the original source is empty.
    assert_eq!(
        runner.run(a, 3, 1, 3).await?,
        Results {
            is_ok: 0,
            value: 91,
            next_val: 2,
        }
    );
    // An empty cycle succeeds without invoking even an immediately failing closure.
    assert_eq!(
        runner.run(a, 0, 1, 0).await?,
        Results {
            is_ok: 1,
            value: 9,
            next_val: NONE,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
