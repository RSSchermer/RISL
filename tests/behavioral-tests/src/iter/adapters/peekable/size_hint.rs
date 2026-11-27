#![cfg(feature = "test_iter_adapters_peekable_size_hint")]

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
        FRONT: u32 as Uniform<u32>,
        BACK: u32 as Uniform<u32>,
        PEEK: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut iter = A.iter().copied().peekable();
        let mut i = 0;

        while i < *FRONT {
            iter.next();
            i += 1;
        }

        if *PEEK != 0 {
            iter.peek();
        }

        let mut i = 0;

        while i < *BACK {
            iter.next_back();
            i += 1;
        }

        let (lower_bound, upper_bound) = iter.size_hint();
        let next_value = iter.next().unwrap_or(NONE);

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                lower_bound: lower_bound as u32,
                upper_bound: upper_bound.unwrap_or(usize::MAX) as u32,
                next_value,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3];

    // An unpeeked iterator forwards the inner iterator's bounds.
    assert_eq!(
        runner.run(a, 0, 0, 0).await?,
        Results {
            lower_bound: 3,
            upper_bound: 3,
            next_value: 1
        }
    );
    // A peeked iterator forwards the inner iterator's bounds without consuming an item.
    assert_eq!(
        runner.run(a, 0, 0, 1).await?,
        Results {
            lower_bound: 3,
            upper_bound: 3,
            next_value: 1
        }
    );
    // After peeking, then draining all except the peeked item from the back, the cached peeked item
    // still contributes one.
    assert_eq!(
        runner.run(a, 0, 2, 1).await?,
        Results {
            lower_bound: 1,
            upper_bound: 1,
            next_value: 1
        }
    );
    // Peeking at an exhausted iterator, then querying the bounds, returns zero bounds.
    assert_eq!(
        runner.run(a, 3, 0, 1).await?,
        Results {
            lower_bound: 0,
            upper_bound: 0,
            next_value: NONE
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
