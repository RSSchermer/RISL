#![cfg(feature = "test_iter_iterator_try_fold")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;

use crate::iter::iterator::SimpleIter;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
struct Results {
    is_ok: u32,
    fold_value: u32,
    next_item: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        START: u32 as Uniform<u32>,
        VALUES: [u32; 4] as Storage<[u32; 4]>,
    },
    result: Results,
    shader: {
        let mut iter = SimpleIter {
            values: VALUES.clone(),
            index: *START as usize,
        };

        let fold_result: Result<u32, u32> = iter.try_fold(0u32, |acc, x| {
            let next = acc + x;

            if next > 50 {
                Err(x)
            } else {
                Ok(next)
            }
        });

        let (is_ok, fold_value) = if let Ok(fold_value) = fold_result {
            (1u32, fold_value)
        } else {
            (0u32, u32::MAX)
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

    let values = [40u32, 30u32, 20u32, 10u32];

    assert_eq!(
        runner.run(0u32, values).await?,
        Results {
            is_ok: 0u32,
            fold_value: u32::MAX,
            next_item: 20,
        }
    );

    assert_eq!(
        runner.run(1u32, values).await?,
        Results {
            is_ok: 0u32,
            fold_value: u32::MAX,
            next_item: u32::MAX,
        }
    );

    assert_eq!(
        runner.run(2u32, values).await?,
        Results {
            is_ok: 1u32,
            fold_value: 30,
            next_item: u32::MAX,
        }
    );

    assert_eq!(
        runner.run(3u32, values).await?,
        Results {
            is_ok: 1u32,
            fold_value: 10,
            next_item: u32::MAX,
        }
    );

    assert_eq!(
        runner.run(4u32, values).await?,
        Results {
            is_ok: 1u32,
            fold_value: 0,
            next_item: u32::MAX,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
