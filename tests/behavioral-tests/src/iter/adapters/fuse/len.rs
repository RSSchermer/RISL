#![cfg(feature = "test_iter_adapters_fuse_len")]

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
        let mut iter = A.iter().fuse();
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

    assert_eq!(runner.run(a, 0, 0).await?, 4);
    assert_eq!(runner.run(a, 1, 0).await?, 3);
    assert_eq!(runner.run(a, 0, 1).await?, 3);
    assert_eq!(runner.run(a, 1, 1).await?, 2);
    // Exact exhaustion reports zero.
    assert_eq!(runner.run(a, 2, 2).await?, 0);
    // Requests past exhaustion from both ends keep the length at zero.
    assert_eq!(runner.run(a, 5, 5).await?, 0);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
