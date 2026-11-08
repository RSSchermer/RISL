#![cfg(feature = "test_iter_adapters_cloned_size_hint")]

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
}

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        FRONT: u32 as Uniform<u32>,
        BACK: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut iter = A.iter().cloned();
        let mut i = 0;

        while i < *FRONT {
            iter.next();
            i += 1;
        }

        i = 0;

        while i < *BACK {
            iter.next_back();
            i += 1;
        }

        let (lower_bound, upper_bound) = iter.size_hint();
        let upper_bound = match upper_bound {
            Some(value) => value as u32,
            None => NONE,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                lower_bound: lower_bound as u32,
                upper_bound,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3, 4];

    // An untouched iterator forwards both exact bounds.
    assert_eq!(
        runner.run(a, 0, 0).await?,
        Results {
            lower_bound: 4,
            upper_bound: 4,
        }
    );
    // Front consumption reduces both bounds.
    assert_eq!(
        runner.run(a, 1, 0).await?,
        Results {
            lower_bound: 3,
            upper_bound: 3,
        }
    );
    // Back consumption also reduces both bounds.
    assert_eq!(
        runner.run(a, 0, 2).await?,
        Results {
            lower_bound: 2,
            upper_bound: 2,
        }
    );
    // Mixed consumption leaves one item in both bounds.
    assert_eq!(
        runner.run(a, 1, 2).await?,
        Results {
            lower_bound: 1,
            upper_bound: 1,
        }
    );
    // Exact exhaustion forwards zero bounds.
    assert_eq!(
        runner.run(a, 2, 2).await?,
        Results {
            lower_bound: 0,
            upper_bound: 0,
        }
    );
    // Further requests from both ends keep the bounds at zero.
    assert_eq!(
        runner.run(a, 5, 5).await?,
        Results {
            lower_bound: 0,
            upper_bound: 0,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
