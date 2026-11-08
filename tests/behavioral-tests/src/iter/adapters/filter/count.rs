#![cfg(feature = "test_iter_adapters_filter_count")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        MIN: u32 as Uniform<u32>,
        PRECONSUME: u32 as Uniform<u32>,
    },
    result: u32,
    shader: {
        let mut iter = A.iter().filter(|value| **value % 2 == 0 && **value >= *MIN);
        let mut i = 0;

        while i < *PRECONSUME {
            iter.next();
            i += 1;
        }

        let result = iter.count() as u32;

        unsafe {
            *RESULT.as_mut_unchecked() = result;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3, 4, 5, 6];

    // Count only the three even matches, not all six source items.
    assert_eq!(runner.run(a, 0, 0).await?, 3);
    // The threshold also excludes otherwise matching evens.
    assert_eq!(runner.run(a, 5, 0).await?, 1);
    // Rejecting every item produces a zero count.
    assert_eq!(runner.run(a, 7, 0).await?, 0);
    // A prior next removes one match from the remaining count.
    assert_eq!(runner.run(a, 0, 1).await?, 2);
    // Consuming all matches leaves no items for count.
    assert_eq!(runner.run(a, 0, 3).await?, 0);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
