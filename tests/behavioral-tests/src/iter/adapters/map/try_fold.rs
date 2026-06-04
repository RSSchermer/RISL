#![cfg(feature = "test_iter_adapters_map_try_fold")]

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
    next_item: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        VALUES: [u32; 4] as Storage<[u32; 4]>,
    },
    result: Results,
    shader: {
        let mut iter = VALUES.iter().map(|x| *x * 2);

        let fold_result: Result<u32, u32> = iter.try_fold(0, |acc, x| {
            if x > 10 {
                Err(x)
            } else {
                Ok(acc + x)
            }
        });

        let (is_ok, fold_value) = match fold_result {
            Ok(fold_value) => (1, fold_value),
            Err(err) => (0, err),
        };

        let next_item = if let Some(next_item) = iter.next() {
            next_item
        } else {
            u32::MAX
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                is_ok,
                fold_value,
                next_item,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    // values: [2, 4, 6, 8] -> mapped: [4, 8, 12, 16]
    // try_fold(0, ...)
    // 1. x=4, acc=0 -> Ok(4)
    // 2. x=8, acc=4 -> Ok(12)
    // 3. x=12, acc=12 -> x > 10 -> Err(12)
    // next_item should be 16
    assert_eq!(
        runner.run([2, 4, 6, 8]).await?,
        Results {
            is_ok: 0,
            fold_value: 12,
            next_item: 16,
        }
    );

    // values: [1, 2, 3, 4] -> mapped: [2, 4, 6, 8]
    // try_fold(0, ...)
    // 1. x=2 -> Ok(2)
    // 2. x=4 -> Ok(6)
    // 3. x=6 -> Ok(12)
    // 4. x=8 -> Ok(20)
    // next_item should be MAX
    assert_eq!(
        runner.run([1, 2, 3, 4]).await?,
        Results {
            is_ok: 1,
            fold_value: 20,
            next_item: u32::MAX,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
