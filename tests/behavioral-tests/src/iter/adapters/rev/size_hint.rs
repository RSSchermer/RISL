#![cfg(feature = "test_iter_adapters_rev_size_hint")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

const NONE: u32 = u32::MAX;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    lower: u32,
    upper: u32,
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
        let mut iter = A.iter().copied().rev();

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

        let (lower, upper) = iter.size_hint();
        let upper = match upper {
            Some(value) => value as u32,
            None => NONE,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                lower: lower as u32,
                upper,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3];

    // Reversing preserves the source's exact bounds.
    assert_eq!(runner.run(a, 0, 0).await?, Results { lower: 3, upper: 3 });
    // The bounds account for consumption from both ends.
    assert_eq!(runner.run(a, 1, 1).await?, Results { lower: 1, upper: 1 });
    // Exhaustion gives an exact zero upper bound, not an unknown bound.
    assert_eq!(runner.run(a, 2, 1).await?, Results { lower: 0, upper: 0 });

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
