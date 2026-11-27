#![cfg(feature = "test_iter_adapters_peekable_peek")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

const NONE: u32 = u32::MAX;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    first_peek: u32,
    repeated_peek: u32,
    next_value: u32,
    following_peek: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
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

        let first_peek = match iter.peek() {
            Some(value) => *value,
            None => NONE,
        };

        let repeated_peek = match iter.peek() {
            Some(value) => *value,
            None => NONE,
        };

        let next_value = match iter.next() {
            Some(value) => value,
            None => NONE,
        };

        let following_peek = match iter.peek() {
            Some(value) => *value,
            None => NONE,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                first_peek,
                repeated_peek,
                next_value,
                following_peek,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3];

    // Repeated peeks retain the first item until next consumes it.
    assert_eq!(
        runner.run(a, 0).await?,
        Results {
            first_peek: 1,
            repeated_peek: 1,
            next_value: 1,
            following_peek: 2,
        }
    );
    // A partially consumed adapter peeks at its current position.
    assert_eq!(
        runner.run(a, 1).await?,
        Results {
            first_peek: 2,
            repeated_peek: 2,
            next_value: 2,
            following_peek: 3,
        }
    );
    // A singleton remainder is cached until consumption; a follow-up peek then returns None.
    assert_eq!(
        runner.run(a, 2).await?,
        Results {
            first_peek: 3,
            repeated_peek: 3,
            next_value: 3,
            following_peek: NONE,
        }
    );
    // Peeking an exhausted iterator returns None.
    assert_eq!(
        runner.run(a, 3).await?,
        Results {
            first_peek: NONE,
            repeated_peek: NONE,
            next_value: NONE,
            following_peek: NONE,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
