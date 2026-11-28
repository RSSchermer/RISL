#![cfg(feature = "test_iter_adapters_rev_rfold")]

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
            iter.next_back();
            i += 1;
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

    // Reverse-folding a fresh reversed iterator visits the source items in forward order.
    assert_eq!(runner.run(a, 0).await?, 8123);
    // Pre-consuming one item from the back starts reverse-folding at the next source item.
    assert_eq!(runner.run(a, 1).await?, 823);
    // Reverse-folding an exhausted reversed iterator preserves the initial accumulator.
    assert_eq!(runner.run(a, 3).await?, 8);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
