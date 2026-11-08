#![cfg(feature = "test_iter_adapters_copied_last")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

use super::{InputValue, Value};

const NONE: u32 = u32::MAX;

test_runner! {
    name: Runner,
    inputs: {
        A: [InputValue] as Storage<[Value]>,
        PRECONSUME: u32 as Uniform<u32>,
    },
    result: u32,
    shader: {
        let mut iter = A.iter().copied();
        let mut i = 0;

        while i < *PRECONSUME {
            iter.next();
            i += 1;
        }

        let result = match iter.last() {
            Some(value) => value.value,
            None => NONE,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = result;
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

    // The final item is copied without invoking the custom Clone.
    assert_eq!(runner.run(a, 0).await?, 4);
    // Forward preconsumption leaves the same final item.
    assert_eq!(runner.run(a, 2).await?, 4);
    // An exhausted adapter has no final item.
    assert_eq!(runner.run(a, 4).await?, NONE);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
