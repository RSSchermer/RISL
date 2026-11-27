#![cfg(feature = "test_iter_adapters_peekable_fold")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        PRECONSUME: u32 as Uniform<u32>,
        PEEK: u32 as Uniform<u32>,
    },
    result: u32,
    shader: {
        let mut iter = A.iter().copied().peekable();

        let mut i = 0;

        while i < *PRECONSUME {
            iter.next();
            i += 1;
        }

        if *PEEK != 0 {
            iter.peek();
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

    // Folding an unpeeked iterator accumulates the inner iterator's items in order.
    assert_eq!(runner.run(a, 0, 0).await?, 8123);
    // Folding a peeked iterator accumulates the inner iterator's items in order.
    assert_eq!(runner.run(a, 0, 1).await?, 8123);
    // An iterator that was peeked after pre-consuming one item starts folding at the second item.
    assert_eq!(runner.run(a, 1, 1).await?, 823);
    // A peeked iterator that has exhausted its inner iterator folds its cached item.
    assert_eq!(runner.run(a, 2, 1).await?, 83);
    // Folding an unpeeked exhausted iterator preserves the initial accumulator.
    assert_eq!(runner.run(a, 3, 0).await?, 8);
    // Folding a peeked exhausted iterator preserves the initial accumulator.
    assert_eq!(runner.run(a, 3, 1).await?, 8);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
