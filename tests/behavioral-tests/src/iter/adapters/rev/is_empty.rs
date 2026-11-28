#![cfg(feature = "test_iter_adapters_rev_is_empty")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        FRONT: u32 as Uniform<u32>,
        BACK: u32 as Uniform<u32>,
    },
    result: u32,
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

        unsafe {
            *RESULT.as_mut_unchecked() = iter.is_empty() as u32;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3];

    // A fresh reversed iterator is not empty.
    assert_eq!(runner.run(a, 0, 0).await?, 0);
    // Consuming from both ends still leaves one item.
    assert_eq!(runner.run(a, 1, 1).await?, 0);
    // Consuming all items makes the reversed iterator empty.
    assert_eq!(runner.run(a, 2, 1).await?, 1);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
