#![cfg(feature = "test_iter_adapters_intersperse_size_hint")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

const NONE: u32 = u32::MAX;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    lower_bound: u32,
    upper_bound: u32,
    next_value: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        LEN: u32 as Uniform<u32>,
        PRECONSUME: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let source = unsafe { A.get_unchecked(..(*LEN as usize)) };
        let mut iter = source.iter().copied().intersperse(9);

        let mut i = 0;

        while i < *PRECONSUME {
            iter.next();
            i += 1;
        }

        let (lower_bound, upper_bound) = iter.size_hint();

        let upper_bound = match upper_bound {
            Some(value) => value as u32,
            None => NONE,
        };

        let next_value = match iter.next() {
            Some(value) => value,
            None => NONE,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                lower_bound: lower_bound as u32,
                upper_bound,
                next_value,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3];

    // Before starting, three items require exactly two separators.
    assert_eq!(
        runner.run(a, 3, 0).await?,
        Results {
            lower_bound: 5,
            upper_bound: 5,
            next_value: 1,
        }
    );
    // After consuming an item, 2 items and 2 separators remain.
    assert_eq!(
        runner.run(a, 3, 1).await?,
        Results {
            lower_bound: 4,
            upper_bound: 4,
            next_value: 9,
        }
    );
    // Exhaustion leaves exact zero bounds.
    assert_eq!(
        runner.run(a, 3, 5).await?,
        Results {
            lower_bound: 0,
            upper_bound: 0,
            next_value: NONE,
        }
    );
    // A singleton contributes no separator to the initial bounds.
    assert_eq!(
        runner.run(a, 1, 0).await?,
        Results {
            lower_bound: 1,
            upper_bound: 1,
            next_value: 1,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
