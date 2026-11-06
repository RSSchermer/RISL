#![cfg(feature = "test_iter_adapters_chain_try_fold")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

const NONE: u32 = u32::MAX;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    is_ok: u32,
    value: u32,
    next_val: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        B: [u32] as Storage<[u32]>,
        BREAK_AT: u32 as Uniform<u32>,
        PRECONSUME: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut iter = A.iter().chain(B.iter());

        if *PRECONSUME != 0 {
            iter.next();
        }

        let mut calls = 0;

        let result: Result<u32, u32> = iter.try_fold(9, |acc, value| {
            let value = acc * 10 + *value;

            calls += 1;

            if calls == *BREAK_AT {
                Err(value)
            } else {
                Ok(value)
            }
        });

        let (is_ok, value) = match result {
            Ok(value) => (1, value),
            Err(value) => (0, value),
        };

        let next_val = match iter.next() {
            Some(value) => *value,
            None => NONE,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                is_ok,
                value,
                next_val,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3, 4];
    let b: &[u32] = &[5, 6, 7, 8];

    assert_eq!(
        runner.run(a, b, 0, 0).await?,
        Results {
            is_ok: 1,
            value: 912345678,
            next_val: NONE,
        }
    );
    assert_eq!(
        runner.run(a, b, 2, 0).await?,
        Results {
            is_ok: 0,
            value: 912,
            next_val: 3,
        }
    );
    assert_eq!(
        runner.run(a, b, 4, 0).await?,
        Results {
            is_ok: 0,
            value: 91234,
            next_val: 5,
        }
    );
    assert_eq!(
        runner.run(a, b, 6, 0).await?,
        Results {
            is_ok: 0,
            value: 9123456,
            next_val: 7,
        }
    );
    assert_eq!(
        runner.run(a, b, 8, 0).await?,
        Results {
            is_ok: 0,
            value: 912345678,
            next_val: NONE,
        }
    );
    assert_eq!(
        runner.run(a, b, 4, 1).await?,
        Results {
            is_ok: 0,
            value: 92345,
            next_val: 6,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
