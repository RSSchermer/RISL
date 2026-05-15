#![cfg(feature = "test_slice_split_last")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Clone, Copy, abi::Sized, Debug, PartialEq, Eq, Default)]
#[gpu]
struct TestResult {
    last: u32,
    rest_sum: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        END: u32 as Uniform<u32>,
        VALUES: [u32] as Storage<[u32]>,
    },
    result: TestResult,
    shader: {
        let values = VALUES.as_ref();
        let end = *END as usize;

        let slice = if let Some(slice) = values.get(..end) {
            slice
        } else {
            &[]
        };

        let result = if let Some((last, rest)) = slice.split_last() {
            let mut sum = 0;

            for v in rest {
                sum += *v;
            }

            TestResult {
                last: *last,
                rest_sum: sum,
            }
        } else {
            TestResult {
                last: 99,
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
    let values = vec![10, 20, 30, 40];

    // Case 1: Sub-slice [] -> last: 99, rest_sum: 99
    assert_eq!(
        runner.run(0, values.clone()).await?,
        TestResult {
            last: 99,
            rest_sum: 99
        }
    );

    // Case 2: Sub-slice [10] -> last: 10, rest: [] (sum: 0)
    assert_eq!(
        runner.run(1, values.clone()).await?,
        TestResult {
            last: 10,
            rest_sum: 0
        }
    );

    // Case 3: Sub-slice [10, 20, 30] -> last: 30, rest: [10, 20] (sum: 30)
    assert_eq!(
        runner.run(3, values.clone()).await?,
        TestResult {
            last: 30,
            rest_sum: 30
        }
    );

    // Case 4: Sub-slice [10, 20, 30, 40] -> last: 40, rest: [10, 20, 30] (sum: 60)
    assert_eq!(
        runner.run(4, values.clone()).await?,
        TestResult {
            last: 40,
            rest_sum: 60
        }
    );

    // Case 5: Out of bounds -> last: 99, rest_sum: 99
    assert_eq!(
        runner.run(5, values.clone()).await?,
        TestResult {
            last: 99,
            rest_sum: 99
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
