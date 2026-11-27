#![cfg(feature = "test_iter_adapters_peekable_next_if_eq")]

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

        let matched = match iter.next_if_eq(&*EXPECTED) {
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
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3];

    // Consumes the first value if it equals the expected value, and resumes at the second value.
    assert_eq!(
        runner.run(a, 1).await?,
        Results {
            matched: 1,
            resumed_value: 2,
        }
    );
    // Does not consume the first value if it does not equal the expected value, and resumes at the
    // first value.
    assert_eq!(
        runner.run(a, 2).await?,
        Results {
            matched: NONE,
            resumed_value: 1,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
