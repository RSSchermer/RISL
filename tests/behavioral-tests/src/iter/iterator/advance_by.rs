#![cfg(feature = "test_iter_iterator_advance_by")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

use crate::iter::iterator::SimpleIter;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    rem: u32,
    next_val: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        START: u32 as Uniform<u32>,
        N: u32 as Uniform<u32>,
        VALUES: [u32; 4] as Storage<[u32; 4]>,
    },
    result: Results,
    shader: {
        let mut iter = SimpleIter {
            values: VALUES.clone(),
            index: *START as usize,
        };

        let res = iter.advance_by(*N as usize);

        let rem = match res {
            Ok(()) => 0u32,
            Err(n) => n.get() as u32,
        };

        let next_val = iter.next().unwrap_or(u32::MAX);

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

    let values = [10, 20, 30, 40];

    // Advance by 0: rem = 0, next_val = 10
    assert_eq!(
        runner.run(0, 0, values).await?,
        Results {
            rem: 0,
            next_val: 10
        }
    );

    // Advance by 2: rem = 0, next_val = 30
    assert_eq!(
        runner.run(0, 2, values).await?,
        Results {
            rem: 0,
            next_val: 30
        }
    );

    // Advance by 4: rem = 0, next_val = None (u32::MAX)
    assert_eq!(
        runner.run(0, 4, values).await?,
        Results {
            rem: 0,
            next_val: u32::MAX
        }
    );

    // Advance by 10 from start: rem = 6 (advanced 4), next_val = None (u32::MAX)
    assert_eq!(
        runner.run(0, 10, values).await?,
        Results {
            rem: 6,
            next_val: u32::MAX
        }
    );

    // Advance by 2 from index 1: rem = 0, next_val = 40
    assert_eq!(
        runner.run(1, 2, values).await?,
        Results {
            rem: 0,
            next_val: 40
        }
    );

    // Advance by 5 from index 2: rem = 3 (advanced 2), next_val = None (u32::MAX)
    assert_eq!(
        runner.run(2, 5, values).await?,
        Results {
            rem: 3,
            next_val: u32::MAX
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
