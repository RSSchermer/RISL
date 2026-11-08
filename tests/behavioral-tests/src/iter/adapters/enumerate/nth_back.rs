#![cfg(feature = "test_iter_adapters_enumerate_nth_back")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

const NONE: u32 = u32::MAX;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    index: u32,
    value: u32,
    next_index: u32,
    next_value: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        N: u32 as Uniform<u32>,
        FRONT: u32 as Uniform<u32>,
        BACK: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut iter = A.iter().enumerate();
        let mut i = 0;

        while i < *FRONT {
            iter.next();
            i += 1;
        }

        let mut i = 0;

        while i < *BACK {
            iter.next_back();
            i += 1;
        }

        let (index, value) = match iter.nth_back(*N as usize) {
            Some((index, value)) => (index as u32, *value),
            None => (NONE, NONE),
        };
        let (next_index, next_value) = match iter.next_back() {
            Some((index, value)) => (index as u32, *value),
            None => (NONE, NONE),
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                index,
                value,
                next_index,
                next_value,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3, 4];

    // A zero backward skip returns the final forward index, then resumes backward.
    assert_eq!(
        runner.run(a, 0, 0, 0).await?,
        Results {
            index: 3,
            value: 4,
            next_index: 2,
            next_value: 3
        }
    );
    // Interior skips retain the original forward positions.
    assert_eq!(
        runner.run(a, 2, 0, 0).await?,
        Results {
            index: 1,
            value: 2,
            next_index: 0,
            next_value: 1
        }
    );
    // Selecting the first source item exhausts the adapter from the back.
    assert_eq!(
        runner.run(a, 3, 0, 0).await?,
        Results {
            index: 0,
            value: 1,
            next_index: NONE,
            next_value: NONE
        }
    );
    // An oversized request exhausts the source without producing a pair.
    assert_eq!(
        runner.run(a, 5, 0, 0).await?,
        Results {
            index: NONE,
            value: NONE,
            next_index: NONE,
            next_value: NONE
        }
    );
    // Backward preconsumption and skipping both reduce the returned index.
    assert_eq!(
        runner.run(a, 1, 0, 1).await?,
        Results {
            index: 1,
            value: 2,
            next_index: 0,
            next_value: 1
        }
    );
    // Mixed consumption requires adding the front count to the remaining length.
    assert_eq!(
        runner.run(a, 0, 1, 1).await?,
        Results {
            index: 2,
            value: 3,
            next_index: 1,
            next_value: 2
        }
    );
    // A source exhausted from both ends stays exhausted on a zero skip.
    assert_eq!(
        runner.run(a, 0, 2, 2).await?,
        Results {
            index: NONE,
            value: NONE,
            next_index: NONE,
            next_value: NONE
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
