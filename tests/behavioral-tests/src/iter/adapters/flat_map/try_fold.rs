#![cfg(feature = "test_iter_adapters_flat_map_try_fold")]

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
        let mut iter = A.iter().flat_map(|len| {
            mapper_calls += 1;

            let offset = mapper_calls;

            unsafe { B.get_unchecked(..(*len as usize)) }
                .iter()
                .map(move |value| *value + offset)
        });

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

        let result: Result<u32, u32> = iter.try_fold(9, |acc, value| {
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

        let next_val = match iter.next() {
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

    // Visit every inner item in order, retaining mapper state across empty inner iterators.
    assert_eq!(
        runner.run(a, b, 0, 0, 0).await?,
        Results {
            is_ok: 1,
            value: 93456,
            next_val: NONE,
        }
    );
    // A first-item error consumes that item and resumes within the same inner iterator.
    assert_eq!(
        runner.run(a, b, 1, 0, 0).await?,
        Results {
            is_ok: 0,
            value: 93,
            next_val: 4,
        }
    );
    // An inner-boundary error resumes across an empty inner iterator with preserved mapper state.
    assert_eq!(
        runner.run(a, b, 2, 0, 0).await?,
        Results {
            is_ok: 0,
            value: 934,
            next_val: 5,
        }
    );
    // Failing on the final item must not turn into success, even with an empty outer tail.
    assert_eq!(
        runner.run(a, b, 4, 0, 0).await?,
        Results {
            is_ok: 0,
            value: 93456,
            next_val: NONE,
        }
    );
    // Pre-consuming an item from the front starts the fold one item later.
    assert_eq!(
        runner.run(a, b, 2, 1, 0).await?,
        Results {
            is_ok: 0,
            value: 945,
            next_val: 6,
        }
    );
    // Pre-consuming an item from the back advances the mapper state and ends the fold one item
    // earlier.
    assert_eq!(
        runner.run(a, b, 0, 0, 1).await?,
        Results {
            is_ok: 1,
            value: 9563,
            next_val: NONE,
        }
    );
    // An error in a pending back iterator ends with the correct value.
    assert_eq!(
        runner.run(a, b, 3, 0, 1).await?,
        Results {
            is_ok: 0,
            value: 9563,
            next_val: NONE,
        }
    );
    // An error in a pending front iterator ends with the correct value.
    assert_eq!(
        runner.run(a, b, 1, 1, 1).await?,
        Results {
            is_ok: 0,
            value: 94,
            next_val: 5,
        }
    );
    // An exhausted adapter succeeds without invoking the fold closure.
    assert_eq!(
        runner.run(a, b, 1, 5, 0).await?,
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
