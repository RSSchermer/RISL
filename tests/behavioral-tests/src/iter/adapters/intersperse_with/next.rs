#![cfg(feature = "test_iter_adapters_intersperse_with_next")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

const NONE: u32 = u32::MAX;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    values: [u32; 7],
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
        let source = unsafe { A.get_unchecked(..(*LEN as usize)) };
        let mut separator_calls = 0;
        let mut iter = source.iter().copied().intersperse_with(|| {
            separator_calls += 1;

            6 + separator_calls
        });

        let mut i = 0;

        while i < *PRECONSUME {
            iter.next();
            i += 1;
        }

        unsafe {
            let result = RESULT.as_mut_unchecked();

            let mut i = 0;

            while i < 7 {
                *result.values.get_unchecked_mut(i) = match iter.next() {
                    Some(value) => value,
                    None => NONE,
                };

                i += 1;
            }

            result.separator_calls = separator_calls;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3];

    // Each separator is generated in order, with no extra calls after exhaustion.
    assert_eq!(
        runner.run(a, 3, 0).await?,
        Results {
            values: [1, 7, 2, 8, 3, NONE, NONE],
            separator_calls: 2,
        }
    );
    // Consuming the first item leaves the first generated separator as the next output.
    assert_eq!(
        runner.run(a, 3, 1).await?,
        Results {
            values: [7, 2, 8, 3, NONE, NONE, NONE],
            separator_calls: 2,
        }
    );
    // Consuming the first separator leaves the second inner item as the next output.
    assert_eq!(
        runner.run(a, 3, 2).await?,
        Results {
            values: [2, 8, 3, NONE, NONE, NONE, NONE],
            separator_calls: 2,
        }
    );
    // An exhausted adapter returns None without invoking the separator closure again.
    assert_eq!(
        runner.run(a, 3, 5).await?,
        Results {
            values: [NONE, NONE, NONE, NONE, NONE, NONE, NONE],
            separator_calls: 2,
        }
    );
    // A singleton produces no separator and never calls the closure.
    assert_eq!(
        runner.run(a, 1, 0).await?,
        Results {
            values: [1, NONE, NONE, NONE, NONE, NONE, NONE],
            separator_calls: 0,
        }
    );
    // An empty inner iterator produces no items and never calls the closure.
    assert_eq!(
        runner.run(a, 0, 0).await?,
        Results {
            values: [NONE, NONE, NONE, NONE, NONE, NONE, NONE],
            separator_calls: 0,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
