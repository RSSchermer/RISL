#![cfg(feature = "test_iter_adapters_flatten_advance_back_by")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

const NONE: u32 = u32::MAX;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    rem: u32,
    next_val: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        B: [u32] as Storage<[u32]>,
        N: u32 as Uniform<u32>,
        PRECONSUME: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut iter = A.iter().map(|len| {
            unsafe { B.get_unchecked(..(*len as usize)) }
        }).flatten();

        let mut i = 0;

        while i < *PRECONSUME {
            iter.next_back();
            i += 1;
        }

        let res = iter.advance_back_by(*N as usize);

        let rem = match res {
            Ok(()) => 0,
            Err(n) => n.get() as u32,
        };

        let next_val = match iter.next_back() {
            Some(value) => *value,
            None => NONE,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results { rem, next_val };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[0, 2, 0, 3, 0];
    let b: &[u32] = &[1, 2, 3];
    let empty: &[u32] = &[0, 0, 0];

    // Zero steps leave the final item available despite an empty inner tail.
    assert_eq!(
        runner.run(a, b, 0, 0).await?,
        Results {
            rem: 0,
            next_val: 3,
        }
    );
    // Stop within the last nonempty inner iterator.
    assert_eq!(
        runner.run(a, b, 1, 0).await?,
        Results {
            rem: 0,
            next_val: 2,
        }
    );
    // An exact inner boundary resumes backward across an empty inner iterator.
    assert_eq!(
        runner.run(a, b, 3, 0).await?,
        Results {
            rem: 0,
            next_val: 2,
        }
    );
    // Cross that boundary and stop within the first nonempty inner iterator.
    assert_eq!(
        runner.run(a, b, 4, 0).await?,
        Results {
            rem: 0,
            next_val: 1,
        }
    );
    // Advancing by the exact flattened length succeeds, including an empty inner prefix.
    assert_eq!(
        runner.run(a, b, 5, 0).await?,
        Results {
            rem: 0,
            next_val: NONE,
        }
    );
    // Oversized advancement reports unfulfilled steps.
    assert_eq!(
        runner.run(a, b, 7, 0).await?,
        Results {
            rem: 2,
            next_val: NONE,
        }
    );
    // Continue from a partially consumed back iterator into the preceding one.
    assert_eq!(
        runner.run(a, b, 3, 1).await?,
        Results {
            rem: 0,
            next_val: 1,
        }
    );
    // Positive advancement on an exhausted adapter is entirely unfulfilled.
    assert_eq!(
        runner.run(a, b, 2, 6).await?,
        Results {
            rem: 2,
            next_val: NONE,
        }
    );
    // Zero advancement still succeeds after exhaustion.
    assert_eq!(
        runner.run(a, b, 0, 6).await?,
        Results {
            rem: 0,
            next_val: NONE,
        }
    );
    // Empty inner iterators cannot satisfy even one backward step.
    assert_eq!(
        runner.run(empty, b, 1, 0).await?,
        Results {
            rem: 1,
            next_val: NONE,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
