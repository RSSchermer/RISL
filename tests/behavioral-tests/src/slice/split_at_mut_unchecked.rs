#![cfg(feature = "test_slice_split_at_mut_unchecked")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Clone, Copy, abi::Sized, Debug, PartialEq, Eq, Default)]
#[gpu]
struct TestResult {
    left: u32,
    right: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        MID: u32 as Uniform<u32>,
        VALUES: [u32] as StorageMut<[u32]>,
    },
    result: TestResult,
    shader: {
        let values = unsafe { VALUES.as_mut_unchecked() };
        let mid = *MID as usize;

        let (left, right) = unsafe { values.split_at_mut_unchecked(mid) };

        // Modify left and right to verify they are mutable
        let first_left = if let Some(first_left) = left.get_mut(0) {
            *first_left += 1;
            *first_left
        } else {
            99
        };

        let first_right = if let Some(first_right) = right.get_mut(0) {
            *first_right += 2;
            *first_right
        } else {
            99
        };

        unsafe {
            *RESULT.as_mut_unchecked() = TestResult {
                left: first_left,
                right: first_right,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    // Case 1: split_at_mut_unchecked(2) -> [10, 20, 30, 40] -> ([11, 20], [32, 40])
    let values1 = vec![10u32, 20u32, 30u32, 40u32];
    let res1 = runner.run(2, values1).await?;
    assert_eq!(res1.left, 11);
    assert_eq!(res1.right, 32);

    // Case 2: split_at_mut_unchecked(0) -> [10, 20, 30, 40] -> ([], [12, 20, 30, 40])
    let values2 = vec![10u32, 20u32, 30u32, 40u32];
    let res2 = runner.run(0, values2).await?;
    assert_eq!(res2.left, 99);
    assert_eq!(res2.right, 12);

    // Case 3: split_at_mut_unchecked(4) -> [10, 20, 30, 40] -> ([11, 20, 30, 40], [])
    let values3 = vec![10u32, 20u32, 30u32, 40u32];
    let res3 = runner.run(4, values3).await?;
    assert_eq!(res3.left, 11);
    assert_eq!(res3.right, 99);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
