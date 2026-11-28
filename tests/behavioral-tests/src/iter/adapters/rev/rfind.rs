#![cfg(feature = "test_iter_adapters_rev_rfind")]

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
    resumed_value: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        A: [Value] as Storage<[Value]>,
        TARGET: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut iter = A.iter().rev();

        let value = match iter.rfind(|value| value.key == *TARGET) {
            Some(value) => value.value,
            None => NONE,
        };

        let resumed_value = match iter.next_back() {
            Some(value) => value.value,
            None => NONE,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                value,
                resumed_value,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[Value] = &[
        Value { value: 1, key: 60 },
        Value { value: 2, key: 20 },
        Value { value: 3, key: 40 },
        Value { value: 4, key: 20 },
    ];

    // Searching from the front skips a nonmatch and selects the first source item with this key.
    assert_eq!(
        runner.run(a, 20).await?,
        Results {
            value: 2,
            resumed_value: 3,
        }
    );
    // A missing key exhausts the iterator.
    assert_eq!(
        runner.run(a, 90).await?,
        Results {
            value: NONE,
            resumed_value: NONE,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
