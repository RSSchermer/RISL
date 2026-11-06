#![cfg(feature = "test_iter_adapters_chain_find")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

const NONE: u32 = u32::MAX;

#[derive(Copy, Clone, abi::Sized)]
#[gpu]
struct Value {
    value: u32,
    key: u32,
}

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    value: u32,
    next_val: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        A: [Value] as Storage<[Value]>,
        B: [Value] as Storage<[Value]>,
        TARGET: u32 as Uniform<u32>,
        PRECONSUME: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut iter = A.iter().chain(B.iter());

        for _ in 0..*PRECONSUME {
            iter.next();
        }

        let value = match iter.find(|value| value.key == *TARGET) {
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
    let a: &[Value] = &[
        Value { value: 1, key: 40 },
        Value { value: 2, key: 20 },
        Value { value: 3, key: 80 },
        Value { value: 4, key: 10 },
    ];
    let b: &[Value] = &[
        Value { value: 5, key: 60 },
        Value { value: 6, key: 30 },
        Value { value: 7, key: 20 },
        Value { value: 8, key: 70 },
    ];

    // Return the first match in A, not the item in B with the same key.
    assert_eq!(
        runner.run(a, b, 20, 0).await?,
        Results {
            value: 2,
            next_val: 3,
        }
    );
    // Consuming the match in A allows the same key to match in B.
    assert_eq!(
        runner.run(a, b, 20, 2).await?,
        Results {
            value: 7,
            next_val: 8,
        }
    );
    assert_eq!(
        runner.run(a, b, 10, 0).await?,
        Results {
            value: 4,
            next_val: 5,
        }
    );
    assert_eq!(
        runner.run(a, b, 30, 0).await?,
        Results {
            value: 6,
            next_val: 7,
        }
    );
    assert_eq!(
        runner.run(a, b, 90, 0).await?,
        Results {
            value: NONE,
            next_val: NONE,
        }
    );
    // An item that was already consumed is not found.
    assert_eq!(
        runner.run(a, b, 40, 1).await?,
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
