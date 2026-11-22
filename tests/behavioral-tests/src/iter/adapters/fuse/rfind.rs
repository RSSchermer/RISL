#![cfg(feature = "test_iter_adapters_fuse_rfind")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

use super::{NONE, NonFusedIter};

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    value: u32,
    next_val: u32,
    predicate_calls: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        A: [u32; 5] as Storage<[u32; 5]>,
        TARGET_REMAINDER: u32 as Uniform<u32>,
        PRECONSUME: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut iter = NonFusedIter::new(A.as_ref()).fuse();
        let mut i = 0;

        while i < *PRECONSUME {
            iter.next_back();
            i += 1;
        }

        let mut predicate_calls = 0;

        let value = match iter.rfind(|value| {
            predicate_calls += 1;

            *value % 2 == *TARGET_REMAINDER
        }) {
            Some(value) => value,
            None => NONE,
        };

        let next_val = match iter.next_back() {
            Some(value) => value,
            None => NONE,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results { value, next_val, predicate_calls };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a = &[1, 2, NONE, 3, 4];
    let full = &[1, 2, 3, 4, 5];
    let last_none = &[1, 2, 3, 4, NONE];

    // Return the last item whose remainder modulo 2 matches the target, not an earlier match.
    assert_eq!(
        runner.run(full, 1, 0).await?,
        Results {
            value: 5,
            next_val: 4,
            predicate_calls: 1
        }
    );
    // Skip a backward item with a different remainder without calling the predicate beyond the
    // match.
    assert_eq!(
        runner.run(full, 0, 0).await?,
        Results {
            value: 4,
            next_val: 3,
            predicate_calls: 2
        }
    );
    // An item with the target remainder beyond a backward gap is not a match; next_back stays None.
    assert_eq!(
        runner.run(a, 0, 1).await?,
        Results {
            value: NONE,
            next_val: NONE,
            predicate_calls: 1
        }
    );
    // A match immediately before the backward gap remains successful.
    assert_eq!(
        runner.run(a, 1, 0).await?,
        Results {
            value: 3,
            next_val: NONE,
            predicate_calls: 2
        }
    );
    // A target remainder of 2 cannot match; reaching the gap prevents resuming beyond it.
    assert_eq!(
        runner.run(a, 2, 0).await?,
        Results {
            value: NONE,
            next_val: NONE,
            predicate_calls: 2
        }
    );
    // Searching a previously fused adapter never invokes the predicate.
    assert_eq!(
        runner.run(a, 1, 3).await?,
        Results {
            value: NONE,
            next_val: NONE,
            predicate_calls: 0
        }
    );
    // An initial backward None also skips all predicate calls.
    assert_eq!(
        runner.run(last_none, 0, 0).await?,
        Results {
            value: NONE,
            next_val: NONE,
            predicate_calls: 0
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
