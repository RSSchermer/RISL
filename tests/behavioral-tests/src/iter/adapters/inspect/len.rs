#![cfg(feature = "test_iter_adapters_inspect_len")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

const NONE: u32 = u32::MAX;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    initial_len: u32,
    remaining_len: u32,
    next_value: u32,
    inspected: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        FRONT: u32 as Uniform<u32>,
        BACK: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut inspected = 0;
        let mut iter = A.iter().copied().inspect(|value| {
            inspected = inspected * 10 + *value;
        });
        let initial_len = iter.len() as u32;

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

        let remaining_len = iter.len() as u32;
        let next_value = match iter.next() {
            Some(value) => value,
            None => NONE,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                initial_len,
                remaining_len,
                next_value,
                inspected,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3, 4];

    // Length queries preserve all four items and do not call the inspector.
    assert_eq!(
        runner.run(a, 0, 0).await?,
        Results {
            initial_len: 4,
            remaining_len: 4,
            next_value: 1,
            inspected: 1,
        }
    );
    // Front consumption updates length.
    assert_eq!(
        runner.run(a, 1, 0).await?,
        Results {
            initial_len: 4,
            remaining_len: 3,
            next_value: 2,
            inspected: 12,
        }
    );
    // Back consumption updates length.
    assert_eq!(
        runner.run(a, 0, 2).await?,
        Results {
            initial_len: 4,
            remaining_len: 2,
            next_value: 1,
            inspected: 431,
        }
    );
    // Exact exhaustion reports zero.
    assert_eq!(
        runner.run(a, 2, 2).await?,
        Results {
            initial_len: 4,
            remaining_len: 0,
            next_value: NONE,
            inspected: 1243,
        }
    );
    // Over-consumption reports zero.
    assert_eq!(
        runner.run(a, 5, 5).await?,
        Results {
            initial_len: 4,
            remaining_len: 0,
            next_value: NONE,
            inspected: 1234,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
