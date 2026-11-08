#![cfg(feature = "test_iter_adapters_enumerate_advance_back_by")]

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
    index: u32,
    value: u32,
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
            iter.next_back();
            i += 1;
        }

        let res = iter.advance_back_by(*N as usize);

        let rem = match res {
            Ok(()) => 0,
            Err(n) => n.get() as u32,
        };

        let (index, value) = match iter.next_back() {
            Some((index, value)) => (index as u32, *value),
            None => (NONE, NONE),
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                rem,
                index,
                value,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3, 4];

    // Advancing by zero preserves the final index and value.
    assert_eq!(
        runner.run(a, 0, 0).await?,
        Results {
            rem: 0,
            index: 3,
            value: 4,
        }
    );
    // Backward advance must not shift the front count or the original index.
    assert_eq!(
        runner.run(a, 2, 0).await?,
        Results {
            rem: 0,
            index: 1,
            value: 2,
        }
    );
    // Advancing by the exact length succeeds and exhausts the adapter.
    assert_eq!(
        runner.run(a, 4, 0).await?,
        Results {
            rem: 0,
            index: NONE,
            value: NONE,
        }
    );
    // An oversized request reports only the unfulfilled steps.
    assert_eq!(
        runner.run(a, 6, 0).await?,
        Results {
            rem: 2,
            index: NONE,
            value: NONE,
        }
    );
    // Back preconsumption and advance preserve the first item's index and value.
    assert_eq!(
        runner.run(a, 2, 1).await?,
        Results {
            rem: 0,
            index: 0,
            value: 1,
        }
    );
    // A positive request on an exhausted adapter is entirely unfulfilled.
    assert_eq!(
        runner.run(a, 2, 4).await?,
        Results {
            rem: 2,
            index: NONE,
            value: NONE,
        }
    );
    // Advancing by zero still succeeds after exhaustion.
    assert_eq!(
        runner.run(a, 0, 4).await?,
        Results {
            rem: 0,
            index: NONE,
            value: NONE,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
