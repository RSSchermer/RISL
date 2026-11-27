#![cfg(feature = "test_iter_adapters_peekable_peek_mut")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

const NONE: u32 = u32::MAX;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    mutated_value: u32,
    repeated_peek: u32,
    next_value: u32,
    following_value: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        DELTA: u32 as Uniform<u32>,
        PRECONSUME: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut iter = A.iter().copied().peekable();
        let mut i = 0;

        while i < *PRECONSUME {
            iter.next();
            i += 1;
        }

        let mutated_value = match iter.peek_mut() {
            Some(value) => {
                *value += *DELTA;

                *value
            }
            None => NONE,
        };

        let repeated_peek = match iter.peek_mut() {
            Some(value) => *value,
            None => NONE,
        };

        let next_value = match iter.next() {
            Some(value) => value,
            None => NONE,
        };

        let following_value = match iter.next() {
            Some(value) => value,
            None => NONE,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                mutated_value,
                repeated_peek,
                next_value,
                following_value,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3];

    // Mutation persists through another peek_mut and next, leaving the following item unchanged.
    assert_eq!(
        runner.run(a, 10, 0).await?,
        Results {
            mutated_value: 11,
            repeated_peek: 11,
            next_value: 11,
            following_value: 2,
        }
    );
    // After partial consumption, only the new head item is mutated.
    assert_eq!(
        runner.run(a, 10, 1).await?,
        Results {
            mutated_value: 12,
            repeated_peek: 12,
            next_value: 12,
            following_value: 3,
        }
    );
    // Peeking an exhausted iterator returns None.
    assert_eq!(
        runner.run(a, 10, 3).await?,
        Results {
            mutated_value: NONE,
            repeated_peek: NONE,
            next_value: NONE,
            following_value: NONE,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
