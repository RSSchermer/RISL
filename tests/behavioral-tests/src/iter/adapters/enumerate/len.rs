#![cfg(feature = "test_iter_adapters_enumerate_len")]

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
        let mut iter = A.iter().enumerate();
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
            *RESULT.as_mut_unchecked() = iter.len() as u32;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3, 4];

    // A fresh enumerate preserves the full length.
    assert_eq!(runner.run(a, 0, 0).await?, 4);
    // Front consumption reduces length rather than counting assigned indices.
    assert_eq!(runner.run(a, 1, 0).await?, 3);
    // Back consumption also reduces the remaining length.
    assert_eq!(runner.run(a, 0, 2).await?, 2);
    // Mixed consumption leaves one item.
    assert_eq!(runner.run(a, 1, 2).await?, 1);
    // Exact exhaustion reports a zero length.
    assert_eq!(runner.run(a, 2, 2).await?, 0);
    // Further requests from both ends keep the length at zero.
    assert_eq!(runner.run(a, 5, 5).await?, 0);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
