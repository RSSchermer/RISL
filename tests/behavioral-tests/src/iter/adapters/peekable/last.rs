#![cfg(feature = "test_iter_adapters_peekable_last")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

const NONE: u32 = u32::MAX;

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
            *RESULT.as_mut_unchecked() = iter.last().unwrap_or(NONE);
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3];

    // An unpeeked iterator returns the inner iterator's last item.
    assert_eq!(runner.run(a, 0, 0).await?, 3);
    // A peeked iterator returns the inner iterator's last item.
    assert_eq!(runner.run(a, 0, 1).await?, 3);
    // If peeking exhausts the inner iterator, last falls back to the cached item.
    assert_eq!(runner.run(a, 2, 1).await?, 3);
    // An exhausted unpeeked iterator has no last item.
    assert_eq!(runner.run(a, 3, 0).await?, NONE);
    // An exhausted peeked iterator has no last item.
    assert_eq!(runner.run(a, 3, 1).await?, NONE);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
