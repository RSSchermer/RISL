#![cfg(feature = "test_iter_adapters_intersperse_fold")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        LEN: u32 as Uniform<u32>,
        PRECONSUME: u32 as Uniform<u32>,
    },
    result: u32,
    shader: {
        let mut iter = unsafe { A.get_unchecked(..(*LEN as usize)) }
            .iter()
            .copied()
            .intersperse(9);

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

    // A fresh adapter folds items and separators in forward order.
    assert_eq!(runner.run(a, 3, 0).await?, 819293);
    // After consuming the first item, fold visits a separator first.
    assert_eq!(runner.run(a, 3, 1).await?, 89293);
    // After consuming the separator item, fold visits the second inner item first.
    assert_eq!(runner.run(a, 3, 2).await?, 8293);
    // An exhausted adapter preserves the initial accumulator.
    assert_eq!(runner.run(a, 3, 5).await?, 8);
    // A singleton produces no separator.
    assert_eq!(runner.run(a, 1, 0).await?, 81);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
