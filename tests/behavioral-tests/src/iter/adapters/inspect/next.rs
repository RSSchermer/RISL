#![cfg(feature = "test_iter_adapters_inspect_next")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

const NONE: u32 = u32::MAX;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    values: [u32; 6],
    inspected: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        PRECONSUME: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut inspected = 0;
        let mut iter = A.iter().copied().inspect(|value| {
            inspected = inspected * 10 + *value;
        });

        let mut i = 0;

        while i < *PRECONSUME {
            iter.next();
            i += 1;
        }

        let value_0 = match iter.next() {
            Some(value) => value,
            None => NONE,
        };
        let value_1 = match iter.next() {
            Some(value) => value,
            None => NONE,
        };
        let value_2 = match iter.next() {
            Some(value) => value,
            None => NONE,
        };
        let value_3 = match iter.next() {
            Some(value) => value,
            None => NONE,
        };
        let value_4 = match iter.next() {
            Some(value) => value,
            None => NONE,
        };
        let value_5 = match iter.next() {
            Some(value) => value,
            None => NONE,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                values: [value_0, value_1, value_2, value_3, value_4, value_5],
                inspected,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3, 4];

    // Forward reads inspect each item in order.
    assert_eq!(
        runner.run(a, 0).await?,
        Results {
            values: [1, 2, 3, 4, NONE, NONE],
            inspected: 1234,
        }
    );
    // Front preconsumption is not reread when iterating over the remaining items.
    assert_eq!(
        runner.run(a, 2).await?,
        Results {
            values: [3, 4, NONE, NONE, NONE, NONE],
            inspected: 1234,
        }
    );
    // Repeated iteration after exhaustion returns None without invoking the inspector again.
    assert_eq!(
        runner.run(a, 4).await?,
        Results {
            values: [NONE, NONE, NONE, NONE, NONE, NONE],
            inspected: 1234,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
