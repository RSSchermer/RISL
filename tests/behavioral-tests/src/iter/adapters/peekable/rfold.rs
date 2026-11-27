#![cfg(feature = "test_iter_adapters_peekable_rfold")]

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
            iter.next_back();
            i += 1;
        }

        if *PEEK != 0 {
            iter.peek();
        }

        let value = iter.rfold(8, |acc, value| acc * 10 + value);

        unsafe {
            *RESULT.as_mut_unchecked() = value;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3];

    // Reverse-folding an unpeeked iterator accumulates the inner iterator's items in reverse order.
    assert_eq!(runner.run(a, 0, 0).await?, 8321);
    // Reverse-folding a peeked iterator accumulates the inner iterator's items in reverse order.
    assert_eq!(runner.run(a, 0, 1).await?, 8321);
    // An iterator peeked after pre-consuming one item from the back starts reverse-folding at the
    // second item from the back.
    assert_eq!(runner.run(a, 1, 1).await?, 821);
    // A peeked iterator that has exhausted its inner iterator folds its cached item.
    assert_eq!(runner.run(a, 2, 1).await?, 81);
    // Reverse-folding an unpeeked exhausted iterator preserves the initial accumulator.
    assert_eq!(runner.run(a, 3, 0).await?, 8);
    // Reverse-folding a peeked exhausted iterator preserves the initial accumulator.
    assert_eq!(runner.run(a, 3, 1).await?, 8);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
