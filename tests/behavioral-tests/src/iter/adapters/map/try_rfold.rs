#![cfg(feature = "test_iter_adapters_map_try_rfold")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    is_ok: u32,
    fold_value: u32,
    next_back_item: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        VALUES: [u32; 4] as Storage<[u32; 4]>,
    },
    result: Results,
    shader: {
        let mut iter = VALUES.iter().map(|x| *x * 2);

        let fold_result: Result<u32, u32> = iter.try_rfold(0, |acc, x| {
            if x < 10 {
                Err(x)
            } else {
                Ok(acc + x)
            }
        });

        let (is_ok, fold_value) = match fold_result {
            Ok(fold_value) => (1, fold_value),
            Err(err) => (0, err),
        };

        let next_back_item = if let Some(next_back_item) = iter.next_back() {
            next_back_item
        } else {
            u32::MAX
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                is_ok,
                fold_value,
                next_back_item,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    // values: [2, 4, 6, 8] -> mapped: [4, 8, 12, 16]
    // try_rfold(0, ...)
    // 1. x=16, acc=0 -> Ok(16)
    // 2. x=12, acc=16 -> Ok(28)
    // 3. x=8, acc=28 -> x < 10 -> Err(8)
    // next_back_item should be 4
    assert_eq!(
        runner.run([2, 4, 6, 8]).await?,
        Results {
            is_ok: 0,
            fold_value: 8,
            next_back_item: 4,
        }
    );

    // values: [6, 7, 8, 9] -> mapped: [12, 14, 16, 18]
    // try_rfold(0, ...)
    // 1. x=18 -> Ok(18)
    // 2. x=16 -> Ok(34)
    // 3. x=14 -> Ok(48)
    // 4. x=12 -> Ok(60)
    // next_back_item should be MAX
    assert_eq!(
        runner.run([6, 7, 8, 9]).await?,
        Results {
            is_ok: 1,
            fold_value: 60,
            next_back_item: u32::MAX,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
