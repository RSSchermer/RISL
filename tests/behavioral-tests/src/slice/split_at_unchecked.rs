#![cfg(feature = "test_slice_split_at_unchecked")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Clone, Copy, abi::Sized, Debug, PartialEq, Eq, Default)]
#[gpu]
struct TestResult {
    len_left: u32,
    len_right: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        MID: u32 as Uniform<u32>,
        VALUES: [u32] as Storage<[u32]>,
    },
    result: TestResult,
    shader: {
        let values = VALUES.as_ref();
        let mid = *MID as usize;

        let (left, right) = unsafe { values.split_at_unchecked(mid) };
        let result = TestResult {
            len_left: left.len() as u32,
            len_right: right.len() as u32,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = result;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let values = vec![10u32, 20u32, 30u32, 40u32];

    // Case 1: split_at_unchecked(0) -> [10, 20, 30, 40] -> ([], [10, 20, 30, 40])
    assert_eq!(
        runner.run(0, values.clone()).await?,
        TestResult {
            len_left: 0,
            len_right: 4,
        }
    );

    // Case 2: split_at_unchecked(2) -> [10, 20, 30, 40] -> ([10, 20], [30, 40])
    assert_eq!(
        runner.run(2, values.clone()).await?,
        TestResult {
            len_left: 2,
            len_right: 2,
        }
    );

    // Case 3: split_at_unchecked(4) -> [10, 20, 30, 40] -> ([10, 20, 30, 40], [])
    assert_eq!(
        runner.run(4, values.clone()).await?,
        TestResult {
            len_left: 4,
            len_right: 0,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
