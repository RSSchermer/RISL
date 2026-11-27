#![cfg(feature = "test_iter_adapters_peekable_count")]

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

        unsafe {
            *RESULT.as_mut_unchecked() = iter.count() as u32;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3];

    // An unpeeked adapter reproduces the inner iterator's count.
    assert_eq!(runner.run(a, 0, 0).await?, 3);
    // A peeked adapter reproduces the inner iterator's count.
    assert_eq!(runner.run(a, 0, 1).await?, 3);
    // Pre-consumed items are subtracted from the count.
    assert_eq!(runner.run(a, 1, 1).await?, 2);
    // An unpeeked exhausted adapter reports a count of zero.
    assert_eq!(runner.run(a, 3, 0).await?, 0);
    // A peeked exhausted adapter reports a count of zero.
    assert_eq!(runner.run(a, 3, 1).await?, 0);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
