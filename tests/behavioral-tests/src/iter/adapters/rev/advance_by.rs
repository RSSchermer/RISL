#![cfg(feature = "test_iter_adapters_rev_advance_by")]

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
        N: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut iter = A.iter().copied().rev();

        let rem = match iter.advance_by(*N as usize) {
            Ok(()) => 0,
            Err(n) => n.get() as u32,
        };

        let next_val = match iter.next() {
            Some(value) => value,
            None => NONE,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                rem,
                next_val,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3];

    // Advancing by zero leaves the source's final item available.
    assert_eq!(
        runner.run(a, 0).await?,
        Results {
            rem: 0,
            next_val: 3,
        }
    );
    // Advancing skips items from the back of the source.
    assert_eq!(
        runner.run(a, 2).await?,
        Results {
            rem: 0,
            next_val: 1,
        }
    );
    // Advancing by the exact length succeeds and exhausts the iterator.
    assert_eq!(
        runner.run(a, 3).await?,
        Results {
            rem: 0,
            next_val: NONE,
        }
    );
    // An oversized advance reports the number of items that could not be skipped.
    assert_eq!(
        runner.run(a, 5).await?,
        Results {
            rem: 2,
            next_val: NONE,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
