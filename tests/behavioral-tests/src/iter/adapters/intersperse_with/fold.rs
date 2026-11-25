#![cfg(feature = "test_iter_adapters_intersperse_with_fold")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    value: u32,
    separator_calls: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        LEN: u32 as Uniform<u32>,
        PRECONSUME: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut separator_calls = 0;
        let mut iter = unsafe { A.get_unchecked(..(*LEN as usize)) }
            .iter()
            .copied()
            .intersperse_with(|| {
                separator_calls += 1;

                6 + separator_calls
            });

        let mut i = 0;

        while i < *PRECONSUME {
            iter.next();
            i += 1;
        }

        let value = iter.fold(8, |acc, value| acc * 10 + value);

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                value,
                separator_calls,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3];

    // A fresh adapter folds items and separators in forward order.
    assert_eq!(
        runner.run(a, 3, 0).await?,
        Results {
            value: 817283,
            separator_calls: 2,
        }
    );
    // After consuming the first item, fold visits a separator first.
    assert_eq!(
        runner.run(a, 3, 1).await?,
        Results {
            value: 87283,
            separator_calls: 2,
        }
    );
    // After consuming the separator item, fold visits the second inner item first.
    assert_eq!(
        runner.run(a, 3, 2).await?,
        Results {
            value: 8283,
            separator_calls: 2,
        }
    );
    // An exhausted adapter preserves the initial accumulator.
    assert_eq!(
        runner.run(a, 3, 5).await?,
        Results {
            value: 8,
            separator_calls: 2,
        }
    );
    // A singleton produces no separator.
    assert_eq!(
        runner.run(a, 1, 0).await?,
        Results {
            value: 81,
            separator_calls: 0,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
