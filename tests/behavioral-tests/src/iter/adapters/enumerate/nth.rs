#![cfg(feature = "test_iter_adapters_enumerate_nth")]

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
        PRECONSUME: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut iter = A.iter().enumerate();
        let mut i = 0;

        while i < *PRECONSUME {
            iter.next();
            i += 1;
        }

        let (index, value) = match iter.nth(*N as usize) {
            Some((index, value)) => (index as u32, *value),
            None => (NONE, NONE),
        };
        let (next_index, next_value) = match iter.next() {
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

    // A zero skip returns the first pair and advances the next index once.
    assert_eq!(
        runner.run(a, 0, 0).await?,
        Results {
            index: 0,
            value: 1,
            next_index: 1,
            next_value: 2
        }
    );
    // Skipped items contribute to both the returned and subsequent indices.
    assert_eq!(
        runner.run(a, 2, 0).await?,
        Results {
            index: 2,
            value: 3,
            next_index: 3,
            next_value: 4
        }
    );
    // Selecting the final item exhausts the adapter.
    assert_eq!(
        runner.run(a, 3, 0).await?,
        Results {
            index: 3,
            value: 4,
            next_index: NONE,
            next_value: NONE
        }
    );
    // An oversized skip returns None and consumes the remaining source.
    assert_eq!(
        runner.run(a, 5, 0).await?,
        Results {
            index: NONE,
            value: NONE,
            next_index: NONE,
            next_value: NONE
        }
    );
    // A skip after preconsumption adds to the existing front count.
    assert_eq!(
        runner.run(a, 1, 1).await?,
        Results {
            index: 2,
            value: 3,
            next_index: 3,
            next_value: 4
        }
    );
    // Even a zero skip returns None on an already exhausted source.
    assert_eq!(
        runner.run(a, 0, 4).await?,
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
