#![cfg(feature = "test_iter_adapters_copied_nth")]

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
    value: u32,
    next_val: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        A: [InputValue] as Storage<[Value]>,
        N: u32 as Uniform<u32>,
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

        let value = match iter.nth(*N as usize) {
            Some(value) => value.value,
            None => NONE,
        };
        let next_val = match iter.next() {
            Some(value) => value.value,
            None => NONE,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
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

    // Index zero copies the first item without invoking the custom Clone.
    assert_eq!(
        runner.run(a, 0, 0).await?,
        Results {
            value: 1,
            next_val: 2,
        }
    );
    // An interior index skips items and resumes at the following item.
    assert_eq!(
        runner.run(a, 1, 0).await?,
        Results {
            value: 2,
            next_val: 3,
        }
    );
    // Selecting the final item exhausts the adapter.
    assert_eq!(
        runner.run(a, 3, 0).await?,
        Results {
            value: 4,
            next_val: NONE,
        }
    );
    // An out-of-bounds index returns None and leaves no remaining item.
    assert_eq!(
        runner.run(a, 4, 0).await?,
        Results {
            value: NONE,
            next_val: NONE,
        }
    );
    // Indexing is relative to the remaining items after preconsumption.
    assert_eq!(
        runner.run(a, 1, 1).await?,
        Results {
            value: 3,
            next_val: 4,
        }
    );
    // Index zero on an exhausted adapter still returns None.
    assert_eq!(
        runner.run(a, 0, 4).await?,
        Results {
            value: NONE,
            next_val: NONE,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
