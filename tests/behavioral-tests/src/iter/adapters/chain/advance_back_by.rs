#![cfg(feature = "test_iter_adapters_chain_advance_back_by")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

const NONE: u32 = u32::MAX;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    rem: u32,
    next_val: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        B: [u32] as Storage<[u32]>,
        N: u32 as Uniform<u32>,
        PRECONSUME: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut iter = A.iter().chain(B.iter());

        if *PRECONSUME != 0 {
            iter.next_back();
        }

        let res = iter.advance_back_by(*N as usize);

        let rem = match res {
            Ok(()) => 0,
            Err(n) => n.get() as u32,
        };

        let next_val = match iter.next_back() {
            Some(value) => *value,
            None => NONE,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                rem,
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
            rem: 0,
            next_val: 8,
        }
    );
    assert_eq!(
        runner.run(a, b, 2, 0).await?,
        Results {
            rem: 0,
            next_val: 6,
        }
    );
    assert_eq!(
        runner.run(a, b, 4, 0).await?,
        Results {
            rem: 0,
            next_val: 4,
        }
    );
    assert_eq!(
        runner.run(a, b, 6, 0).await?,
        Results {
            rem: 0,
            next_val: 2,
        }
    );
    assert_eq!(
        runner.run(a, b, 8, 0).await?,
        Results {
            rem: 0,
            next_val: NONE,
        }
    );
    assert_eq!(
        runner.run(a, b, 10, 0).await?,
        Results {
            rem: 2,
            next_val: NONE,
        }
    );
    assert_eq!(
        runner.run(a, b, 4, 1).await?,
        Results {
            rem: 0,
            next_val: 3,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
