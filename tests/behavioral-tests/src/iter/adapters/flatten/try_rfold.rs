#![cfg(feature = "test_iter_adapters_flatten_try_rfold")]

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
        B: [u32] as Storage<[u32]>,
        BREAK_AT: u32 as Uniform<u32>,
        FRONT: u32 as Uniform<u32>,
        BACK: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut mapper_calls = 0;
        let mut iter = A.iter().map(|len| {
            mapper_calls += 1;

            let offset = mapper_calls;

            unsafe { B.get_unchecked(..(*len as usize)) }
                .iter()
                .map(move |value| *value + offset)
        }).flatten();

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

        let next_val = match iter.next_back() {
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
    let a: &[u32] = &[0, 2, 0, 2, 0];
    let b: &[u32] = &[1, 2];
    let empty: &[u32] = &[0, 0, 0];

    // Visit every inner item in reverse order, retaining mapper state across empty inner iterators.
    assert_eq!(
        runner.run(a, b, 0, 0, 0).await?,
        Results {
            is_ok: 1,
            value: 94365,
            next_val: NONE,
        }
    );
    // A first-item error resumes backward within the same inner iterator.
    assert_eq!(
        runner.run(a, b, 1, 0, 0).await?,
        Results {
            is_ok: 0,
            value: 94,
            next_val: 3,
        }
    );
    // An inner-boundary error resumes across an empty inner iterator with preserved mapper state.
    assert_eq!(
        runner.run(a, b, 2, 0, 0).await?,
        Results {
            is_ok: 0,
            value: 943,
            next_val: 6,
        }
    );
    // Failing on the final item must not turn into success, even with an empty outer tail.
    assert_eq!(
        runner.run(a, b, 4, 0, 0).await?,
        Results {
            is_ok: 0,
            value: 94365,
            next_val: NONE,
        }
    );
    // Pre-consuming an item from the back starts the fold one item later.
    assert_eq!(
        runner.run(a, b, 2, 0, 1).await?,
        Results {
            is_ok: 0,
            value: 936,
            next_val: 5,
        }
    );
    // Pre-consuming an item from the front advances the mapper state and ends the fold one item
    // earlier.
    assert_eq!(
        runner.run(a, b, 0, 1, 0).await?,
        Results {
            is_ok: 1,
            value: 9654,
            next_val: NONE,
        }
    );
    // An error in that pending front iterator must propagate.
    assert_eq!(
        runner.run(a, b, 3, 1, 0).await?,
        Results {
            is_ok: 0,
            value: 9654,
            next_val: NONE,
        }
    );
    // An error in a pending back iterator ends with the correct value.
    assert_eq!(
        runner.run(a, b, 1, 1, 1).await?,
        Results {
            is_ok: 0,
            value: 95,
            next_val: 4,
        }
    );
    // An exhausted adapter succeeds without invoking the fold closure.
    assert_eq!(
        runner.run(a, b, 1, 0, 5).await?,
        Results {
            is_ok: 1,
            value: 9,
            next_val: NONE,
        }
    );
    // All-empty inner iterators do not invoke the fold closure.
    assert_eq!(
        runner.run(empty, b, 1, 0, 0).await?,
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
