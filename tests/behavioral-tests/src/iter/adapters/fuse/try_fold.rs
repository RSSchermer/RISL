#![cfg(feature = "test_iter_adapters_fuse_try_fold")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

use super::{NONE, NonFusedIter};

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    is_ok: u32,
    value: u32,
    next_val: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        A: [u32; 5] as Storage<[u32; 5]>,
        BREAK_AT: u32 as Uniform<u32>,
        PRECONSUME: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut iter = NonFusedIter::new(A.as_ref()).fuse();
        let mut i = 0;

        while i < *PRECONSUME {
            iter.next();
            i += 1;
        }

        let mut calls = 0;

        let result: Result<u32, u32> = iter.try_fold(9, |acc, value| {
            let value = acc * 10 + value;

            calls += 1;

            if calls == *BREAK_AT {
                Err(value)
            } else {
                Ok(value)
            }
        });

        let (is_ok, value) = match result {
            Ok(value) => (1, value),
            Err(value) => (0, value),
        };

        let next_val = match iter.next() {
            Some(value) => value,
            None => NONE,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results { is_ok, value, next_val };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a = &[1, 2, NONE, 3, 4];
    let full = &[1, 2, 3, 4, 5];
    let first_none = &[NONE, 1, 2, 3, 4];

    // Reaching the first gap skips the remaining items.
    assert_eq!(
        runner.run(a, 0, 0).await?,
        Results {
            is_ok: 1,
            value: 912,
            next_val: NONE
        }
    );
    // A closure error must not fuse; resume immediately after the failing item.
    assert_eq!(
        runner.run(a, 1, 0).await?,
        Results {
            is_ok: 0,
            value: 91,
            next_val: 2
        }
    );
    // An error just before a gap is still an error.
    assert_eq!(
        runner.run(a, 2, 0).await?,
        Results {
            is_ok: 0,
            value: 912,
            next_val: NONE
        }
    );
    // An already-fused adapter returns the initial accumulator without invoking the closure.
    assert_eq!(
        runner.run(a, 1, 3).await?,
        Results {
            is_ok: 1,
            value: 9,
            next_val: NONE
        }
    );
    // Gap-free exhaustion includes every item in order.
    assert_eq!(
        runner.run(full, 0, 0).await?,
        Results {
            is_ok: 1,
            value: 912345,
            next_val: NONE
        }
    );
    // Failing on the physical last item must not turn the error into success.
    assert_eq!(
        runner.run(full, 5, 0).await?,
        Results {
            is_ok: 0,
            value: 912345,
            next_val: NONE
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
