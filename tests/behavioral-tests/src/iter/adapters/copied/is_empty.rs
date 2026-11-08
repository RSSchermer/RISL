#![cfg(feature = "test_iter_adapters_copied_is_empty")]

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
        let mut iter = A.iter().copied();
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
            *RESULT.as_mut_unchecked() = if iter.is_empty() { 1 } else { 0 };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3, 4];

    // An untouched iterator is not empty.
    assert_eq!(runner.run(a, 0, 0).await?, 0);
    // Front consumption leaves the iterator nonempty.
    assert_eq!(runner.run(a, 1, 0).await?, 0);
    // Back consumption also leaves the iterator nonempty.
    assert_eq!(runner.run(a, 0, 2).await?, 0);
    // One remaining item after mixed consumption is still nonempty.
    assert_eq!(runner.run(a, 1, 2).await?, 0);
    // Exact exhaustion makes the iterator empty.
    assert_eq!(runner.run(a, 2, 2).await?, 1);
    // Further requests from both ends preserve emptiness.
    assert_eq!(runner.run(a, 5, 5).await?, 1);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
