#![cfg(feature = "test_slice_split_at_mut_checked")]

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
    found: u32,
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

        let result = if let Some((left, right)) = values.split_at_mut_checked(mid) {
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

            TestResult {
                left: first_left,
                right: first_right,
                found: 1,
            }
        } else {
            TestResult {
                left: 0,
                right: 0,
                found: 0,
            }
        };

        unsafe {
            *RESULT.as_mut_unchecked() = result;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    // Case 1: split_at_mut_checked(2) -> [10, 20, 30, 40] -> ([11, 20], [32, 40])
    let values1 = vec![10, 20, 30, 40];
    let res1 = runner.run(2, values1).await?;
    assert_eq!(res1.left, 11);
    assert_eq!(res1.right, 32);
    assert_eq!(res1.found, 1);

    // Case 2: split_at_mut_checked(0) -> [10, 20, 30, 40] -> ([], [12, 20, 30, 40])
    let values2 = vec![10, 20, 30, 40];
    let res2 = runner.run(0, values2).await?;
    assert_eq!(res2.left, 99);
    assert_eq!(res2.right, 12);
    assert_eq!(res2.found, 1);

    // Case 3: split_at_mut_checked(4) -> [10, 20, 30, 40] -> ([11, 20, 30, 40], [])
    let values3 = vec![10, 20, 30, 40];
    let res3 = runner.run(4, values3).await?;
    assert_eq!(res3.left, 11);
    assert_eq!(res3.right, 99);
    assert_eq!(res3.found, 1);

    // Case 4: split_at_mut_checked(5) -> [10, 20, 30, 40] -> None
    let values4 = vec![10, 20, 30, 40];
    let res4 = runner.run(5, values4).await?;
    assert_eq!(res4.left, 0);
    assert_eq!(res4.right, 0);
    assert_eq!(res4.found, 0);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
