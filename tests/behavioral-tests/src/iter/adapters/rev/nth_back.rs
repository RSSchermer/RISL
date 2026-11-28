#![cfg(feature = "test_iter_adapters_rev_nth_back")]

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
        N: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut iter = A.iter().copied().rev();

        let value = match iter.nth_back(*N as usize) {
            Some(value) => value,
            None => NONE,
        };

        let resumed_value = match iter.next_back() {
            Some(value) => value,
            None => NONE,
        };

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

    // nth_back(0) consumes the source's first item and resumes towards its back.
    assert_eq!(
        runner.run(a, 0).await?,
        Results {
            value: 1,
            resumed_value: 2,
        }
    );
    // Skipping one item selects the second item from the front.
    assert_eq!(
        runner.run(a, 1).await?,
        Results {
            value: 2,
            resumed_value: 3,
        }
    );
    // An out-of-range index returns None and exhausts the iterator.
    assert_eq!(
        runner.run(a, 4).await?,
        Results {
            value: NONE,
            resumed_value: NONE,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
