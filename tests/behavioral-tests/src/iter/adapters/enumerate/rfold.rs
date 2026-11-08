#![cfg(feature = "test_iter_adapters_enumerate_rfold")]

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

        let mut i = 0;

        while i < *BACK {
            iter.next_back();
            i += 1;
        }

        let value = iter.rfold(9, |acc, (index, value)| {
            acc * 100 + index as u32 * 10 + *value
        });

        unsafe {
            *RESULT.as_mut_unchecked() = value;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3, 4];

    // Append index/value pairs 34, 23, 12, 01 in reverse order to a nonzero seed.
    assert_eq!(runner.run(a, 0, 0).await?, 934231201);
    // Backward preconsumption lowers the starting index without renumbering items.
    assert_eq!(runner.run(a, 0, 1).await?, 9231201);
    // The front count offsets the remaining length to recover original indices.
    assert_eq!(runner.run(a, 1, 0).await?, 9342312);
    // Mixed consumption folds just the middle items in reverse order.
    assert_eq!(runner.run(a, 1, 1).await?, 92312);
    // An exhausted source leaves the seed untouched without calling the closure.
    assert_eq!(runner.run(a, 2, 2).await?, 9);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
