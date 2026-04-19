#![cfg(feature = "test_slice_split_first_mut")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Clone, Copy, abi::Sized, Debug, PartialEq, Eq, Default)]
#[gpu]
struct TestResult {
    first: u32,
    rest_sum: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        START: u32 as Uniform<u32>,
        VALUES: [u32] as StorageMut<[u32]>,
    },
    result: TestResult,
    shader: {
        let mut values = unsafe { VALUES.as_mut_unchecked() };
        let start = *START as usize;

        let slice = if let Some(slice) = values.get_mut(start..) {
            slice
        } else {
            &mut []
        };

        let result = if let Some((first, rest)) = slice.split_first_mut() {
            *first += 100;

            let mut sum = 0;

            for v in rest {
                *v += 1;
                sum += *v;
            }

            TestResult {
                first: *first,
                rest_sum: sum,
            }
        } else {
            TestResult {
                first: 99,
                rest_sum: 99,
            }
        };

        unsafe {
            *RESULT.as_mut_unchecked() = result;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let values = vec![10u32, 20u32, 30u32, 40u32];

    // Case 1: Sub-slice [10, 20, 30, 40]
    // mutate: first (10) -> 110, rest ([20, 30, 40]) -> [21, 31, 41] (sum: 93)
    assert_eq!(
        runner.run(0, values.clone()).await?,
        TestResult {
            first: 110,
            rest_sum: 93
        }
    );

    // Case 2: Sub-slice [20, 30, 40]
    // mutate: first (20) -> 120, rest ([30, 40]) -> [31, 41] (sum: 72)
    assert_eq!(
        runner.run(1, values.clone()).await?,
        TestResult {
            first: 120,
            rest_sum: 72
        }
    );

    // Case 3: Sub-slice [40]
    // mutate: first (40) -> 140, rest ([]) -> [] (sum: 0)
    assert_eq!(
        runner.run(3, values.clone()).await?,
        TestResult {
            first: 140,
            rest_sum: 0
        }
    );

    // Case 4: Sub-slice []
    // -> first: 99, rest_sum: 99
    assert_eq!(
        runner.run(4, values.clone()).await?,
        TestResult {
            first: 99,
            rest_sum: 99
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
