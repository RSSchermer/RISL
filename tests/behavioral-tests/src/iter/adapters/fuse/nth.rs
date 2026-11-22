#![cfg(feature = "test_iter_adapters_fuse_nth")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

use super::{NONE, NonFusedIter};

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    value: u32,
    next_val: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        A: [u32; 5] as Storage<[u32; 5]>,
        N: u32 as Uniform<u32>,
        PRECONSUME: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut iter = NonFusedIter::new(A.as_ref()).fuse();
        let mut i = 0;

        while i < *PRECONSUME {
            iter.next();
            i += 1;
        }

        let value = match iter.nth(*N as usize) {
            Some(value) => value,
            None => NONE,
        };

        let next_val = match iter.next() {
            Some(value) => value,
            None => NONE,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results { value, next_val };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a = &[1, 2, NONE, 3, 4];
    let full = &[1, 2, 3, 4, 5];

    // nth(0) returns the next item.
    assert_eq!(
        runner.run(a, 0, 0).await?,
        Results {
            value: 1,
            next_val: 2
        }
    );
    // Skipping to the last item before the gap finds that the next item is None.
    assert_eq!(
        runner.run(a, 1, 0).await?,
        Results {
            value: 2,
            next_val: NONE
        }
    );
    // Skipping to the gap returns None and find that the next item remains None.
    assert_eq!(
        runner.run(a, 2, 0).await?,
        Results {
            value: NONE,
            next_val: NONE
        }
    );
    // Skipping past the gap returns None and find that the next item remains None.
    assert_eq!(
        runner.run(a, 4, 0).await?,
        Results {
            value: NONE,
            next_val: NONE
        }
    );
    // Preconsuming one item shifts the nth item by one.
    assert_eq!(
        runner.run(a, 0, 1).await?,
        Results {
            value: 2,
            next_val: NONE
        }
    );
    // A previously fused adapter stays exhausted even for nth(0).
    assert_eq!(
        runner.run(a, 0, 3).await?,
        Results {
            value: NONE,
            next_val: NONE
        }
    );
    // A gap-free iterator can reach its final item.
    assert_eq!(
        runner.run(full, 4, 0).await?,
        Results {
            value: 5,
            next_val: NONE
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
