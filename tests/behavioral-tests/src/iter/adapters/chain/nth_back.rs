#![cfg(feature = "test_iter_adapters_chain_nth_back")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

const NONE: u32 = u32::MAX;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    value: u32,
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
        let mut iter = A.iter().chain(B.iter());
        let mut i = 0;

        while i < *PRECONSUME {
            iter.next_back();
            i += 1;
        }

        let value = match iter.nth_back(*N as usize) {
            Some(value) => *value,
            None => NONE,
        };
        let next_val = match iter.next_back() {
            Some(value) => *value,
            None => NONE,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                value,
                next_val,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3, 4];
    let b: &[u32] = &[5, 6, 7, 8];

    assert_eq!(
        runner.run(a, b, 0, 0).await?,
        Results {
            value: 8,
            next_val: 7,
        }
    );
    assert_eq!(
        runner.run(a, b, 2, 0).await?,
        Results {
            value: 6,
            next_val: 5,
        }
    );
    assert_eq!(
        runner.run(a, b, 3, 0).await?,
        Results {
            value: 5,
            next_val: 4,
        }
    );
    assert_eq!(
        runner.run(a, b, 4, 0).await?,
        Results {
            value: 4,
            next_val: 3,
        }
    );
    assert_eq!(
        runner.run(a, b, 7, 0).await?,
        Results {
            value: 1,
            next_val: NONE,
        }
    );
    assert_eq!(
        runner.run(a, b, 8, 0).await?,
        Results {
            value: NONE,
            next_val: NONE,
        }
    );
    assert_eq!(
        runner.run(a, b, 4, 1).await?,
        Results {
            value: 3,
            next_val: 2,
        }
    );
    assert_eq!(
        runner.run(a, b, 0, 5).await?,
        Results {
            value: 3,
            next_val: 2,
        }
    );
    assert_eq!(
        runner.run(a, b, 0, 9).await?,
        Results {
            value: NONE,
            next_val: NONE,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
