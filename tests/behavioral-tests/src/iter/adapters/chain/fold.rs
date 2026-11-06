#![cfg(feature = "test_iter_adapters_chain_fold")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        B: [u32] as Storage<[u32]>,
        PRECONSUME: u32 as Uniform<u32>,
    },
    result: u32,
    shader: {
        let mut iter = A.iter().chain(B.iter());
        let mut i = 0;

        while i < *PRECONSUME {
            iter.next();
            i += 1;
        }

        let value = iter.fold(9, |acc, value| acc * 10 + *value);

        unsafe {
            *RESULT.as_mut_unchecked() = value;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3, 4];
    let b: &[u32] = &[5, 6, 7, 8];

    assert_eq!(runner.run(a, b, 0).await?, 912345678);
    assert_eq!(runner.run(a, b, 1).await?, 92345678);
    assert_eq!(runner.run(a, b, 5).await?, 9678);
    assert_eq!(runner.run(a, b, 8).await?, 9);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
