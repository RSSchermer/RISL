#![cfg(feature = "test_iter_adapters_copied_next")]

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
    result: [u32; 6],
    shader: {
        let mut iter = A.iter().copied();
        let mut i = 0;

        while i < *PRECONSUME {
            iter.next();
            i += 1;
        }

        unsafe {
            let res = RESULT.as_mut_unchecked();
            let mut i = 0;

            while i < 6 {
                *res.get_unchecked_mut(i) = match iter.next() {
                    Some(value) => value.value,
                    None => NONE,
                };

                i += 1;
            }
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

    // Copy unchanged values in order without calling Clone, then report repeated None.
    assert_eq!(runner.run(a, 0).await?, [1, 2, 3, 4, NONE, NONE]);
    // Resume after forward preconsumption without revisiting removed items.
    assert_eq!(runner.run(a, 2).await?, [3, 4, NONE, NONE, NONE, NONE]);
    // An exhausted adapter produces no copied values.
    assert_eq!(runner.run(a, 4).await?, [NONE; 6]);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
