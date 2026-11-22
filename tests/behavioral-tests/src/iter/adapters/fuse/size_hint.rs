#![cfg(feature = "test_iter_adapters_fuse_size_hint")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

use super::{NONE, NonFusedIter};

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    lower_bound: u32,
    upper_bound: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        A: [u32; 5] as Storage<[u32; 5]>,
        FRONT: u32 as Uniform<u32>,
        BACK: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut iter = NonFusedIter::new(A.as_ref()).fuse();
        let mut i = 0;

        while i < *FRONT {
            iter.next();
            i += 1;
        }

        let mut i = 0;

        while i < *BACK {
            iter.next_back();
            i += 1;
        }

        let (lower, upper) = iter.size_hint();
        let upper_bound = match upper {
            Some(value) => value as u32,
            None => NONE,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results { lower_bound: lower as u32, upper_bound };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a = &[1, 2, NONE, 3, 4];

    // Forwards the inner iterator's upper-bound, not the number of items before its first None.
    assert_eq!(
        runner.run(a, 0, 0).await?,
        Results {
            lower_bound: 0,
            upper_bound: 5
        }
    );
    // Looking at the bounds must not eagerly inspect the upcoming None.
    assert_eq!(
        runner.run(a, 2, 0).await?,
        Results {
            lower_bound: 0,
            upper_bound: 3
        }
    );
    // If forward iteration discovers the gap, then the upper-bound becomes zero, despite the inner
    // iterator containing more items after the gap.
    assert_eq!(
        runner.run(a, 3, 0).await?,
        Results {
            lower_bound: 0,
            upper_bound: 0
        }
    );
    // If backward iteration discovers the gap, then the upper-bound becomes zero, despite the inner
    // iterator containing more items before the gap.
    assert_eq!(
        runner.run(a, 0, 3).await?,
        Results {
            lower_bound: 0,
            upper_bound: 0
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
