#![cfg(feature = "test_iter_adapters_copied_fold")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

use super::{InputValue, Value};

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

        let value = iter.fold(9, |acc, value| acc * 10 + value.value);

        unsafe {
            *RESULT.as_mut_unchecked() = value;
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

    // The accumulator encodes forward visitation order without invoking Clone.
    assert_eq!(runner.run(a, 0).await?, 91234);
    // Only the remaining copied items participate after partial consumption.
    assert_eq!(runner.run(a, 2).await?, 934);
    // No closure call changes the initial value once exhausted.
    assert_eq!(runner.run(a, 4).await?, 9);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
