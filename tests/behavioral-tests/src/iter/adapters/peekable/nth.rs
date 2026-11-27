#![cfg(feature = "test_iter_adapters_peekable_nth")]

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
    resumed_value: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        PEEK: u32 as Uniform<u32>,
        N: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut iter = A.iter().copied().peekable();

        if *PEEK != 0 {
            iter.peek();
        }

        let value = iter.nth(*N as usize).unwrap_or(NONE);
        let resumed_value = iter.next().unwrap_or(NONE);

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                value,
                resumed_value,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3];

    // On an unpeeked iterator, nth skips to the correct item in the inner iterator.
    assert_eq!(
        runner.run(a, 0, 1).await?,
        Results {
            value: 2,
            resumed_value: 3
        }
    );
    // On a peeked iterator, nth skips to the correct item in the inner iterator.
    assert_eq!(
        runner.run(a, 1, 1).await?,
        Results {
            value: 2,
            resumed_value: 3
        }
    );
    // Skipping beyond the end returns None.
    assert_eq!(
        runner.run(a, 1, 4).await?,
        Results {
            value: NONE,
            resumed_value: NONE
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
