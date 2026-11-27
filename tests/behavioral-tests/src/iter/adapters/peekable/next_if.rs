#![cfg(feature = "test_iter_adapters_peekable_next_if")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

const NONE: u32 = u32::MAX;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    matched: u32,
    resumed_value: u32,
    predicate_calls: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        EXPECTED: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut iter = A.iter().copied().peekable();

        let mut predicate_calls = 0;

        let matched = match iter.next_if(|value| {
            predicate_calls += 1;

            *value == *EXPECTED
        }) {
            Some(value) => value,
            None => NONE,
        };

        let resumed_value = match iter.next() {
            Some(value) => value,
            None => NONE,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                matched,
                resumed_value,
                predicate_calls,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3];

    // Consumes the first value if the predicate matches, and resumes at the second value.
    assert_eq!(
        runner.run(a, 1).await?,
        Results {
            matched: 1,
            resumed_value: 2,
            predicate_calls: 1,
        }
    );
    // Does not consume the first value if the predicate does not match, and resumes at the first
    // value.
    assert_eq!(
        runner.run(a, 2).await?,
        Results {
            matched: NONE,
            resumed_value: 1,
            predicate_calls: 1,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
