#![cfg(feature = "test_iter_adapters_cycle_advance_by")]

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
        LEN: u32 as Uniform<u32>,
        N: u32 as Uniform<u32>,
        PRECONSUME: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let source = unsafe { A.get_unchecked(..(*LEN as usize)) };
        let mut iter = source.iter().cycle();
        let mut i = 0;

        while i < *PRECONSUME {
            iter.next();
            i += 1;
        }

        let res = iter.advance_by(*N as usize);

        let rem = match res {
            Ok(()) => 0,
            Err(n) => n.get() as u32,
        };

        let next_val = match iter.next() {
            Some(value) => *value,
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
    let a: &[u32] = &[1, 2, 3, 4];

    // Advancing by zero leaves the first item untouched.
    assert_eq!(
        runner.run(a, 4, 0, 0).await?,
        Results {
            rem: 0,
            next_val: 1,
        }
    );
    // Advance within the current iteration.
    assert_eq!(
        runner.run(a, 4, 2, 0).await?,
        Results {
            rem: 0,
            next_val: 3,
        }
    );
    // An exact cycle succeeds, and the next call starts a new cycle.
    assert_eq!(
        runner.run(a, 4, 4, 0).await?,
        Results {
            rem: 0,
            next_val: 1,
        }
    );
    // Requests spanning multiple cycles succeed without an unfulfilled remainder.
    assert_eq!(
        runner.run(a, 4, 10, 0).await?,
        Results {
            rem: 0,
            next_val: 3,
        }
    );
    // Use the current remainder before resetting to the full original source.
    assert_eq!(
        runner.run(a, 4, 2, 3).await?,
        Results {
            rem: 0,
            next_val: 2,
        }
    );
    // A request starting at an exhausted current iteration resets before advancing.
    assert_eq!(
        runner.run(a, 4, 1, 4).await?,
        Results {
            rem: 0,
            next_val: 2,
        }
    );
    // Empty sources return the entire positive request instead of resetting forever.
    assert_eq!(
        runner.run(a, 0, 3, 0).await?,
        Results {
            rem: 3,
            next_val: NONE,
        }
    );
    // Advancing an empty cycle by zero still succeeds.
    assert_eq!(
        runner.run(a, 0, 0, 0).await?,
        Results {
            rem: 0,
            next_val: NONE,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
