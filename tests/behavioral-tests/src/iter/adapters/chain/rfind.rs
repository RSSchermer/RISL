#![cfg(feature = "test_iter_adapters_chain_rfind")]

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
        let mut i = 0;

        while i < *PRECONSUME {
            iter.next_back();
            i += 1;
        }

        let value = match iter.rfind(|value| value.key == *TARGET) {
            Some(value) => value.value,
            None => NONE,
        };

        let next_val = match iter.next_back() {
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

    // Return the first match from the back in B, not the item in A with the same key.
    assert_eq!(
        runner.run(a, b, 20, 0).await?,
        Results {
            value: 7,
            next_val: 6,
        }
    );
    // Consuming the match in B allows the same key to match in A.
    assert_eq!(
        runner.run(a, b, 20, 2).await?,
        Results {
            value: 2,
            next_val: 1,
        }
    );
    assert_eq!(
        runner.run(a, b, 60, 0).await?,
        Results {
            value: 5,
            next_val: 4,
        }
    );
    assert_eq!(
        runner.run(a, b, 80, 0).await?,
        Results {
            value: 3,
            next_val: 2,
        }
    );
    assert_eq!(
        runner.run(a, b, 90, 0).await?,
        Results {
            value: NONE,
            next_val: NONE,
        }
    );
    // An item that was already consumed from the back is not found.
    assert_eq!(
        runner.run(a, b, 70, 1).await?,
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
