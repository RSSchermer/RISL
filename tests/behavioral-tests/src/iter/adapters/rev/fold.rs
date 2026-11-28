#![cfg(feature = "test_iter_adapters_rev_fold")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        PRECONSUME: u32 as Uniform<u32>,
    },
    result: u32,
    shader: {
        let mut iter = A.iter().copied().rev();

        let mut i = 0;

        while i < *PRECONSUME {
            iter.next();
            i += 1;
        }

        let value = iter.fold(8, |acc, value| acc * 10 + value);

        unsafe {
            *RESULT.as_mut_unchecked() = value;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3];

    // Folding a fresh reversed iterator visits the source items in reverse order.
    assert_eq!(runner.run(a, 0).await?, 8321);
    // Pre-consuming one item starts folding at the next item in reversed order.
    assert_eq!(runner.run(a, 1).await?, 821);
    // Folding an exhausted reversed iterator preserves the initial accumulator.
    assert_eq!(runner.run(a, 3).await?, 8);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
