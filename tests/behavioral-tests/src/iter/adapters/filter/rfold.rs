#![cfg(feature = "test_iter_adapters_filter_rfold")]

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
        let mut predicate_calls = 0;
        let mut iter = A.iter().filter(|value| {
            predicate_calls += 1;

            **value >= *MIN && predicate_calls % 3 != 0
        });
        let mut i = 0;

        while i < *PRECONSUME {
            iter.next_back();
            i += 1;
        }

        let value = iter.rfold(9, |acc, value| acc * 10 + *value);

        unsafe {
            *RESULT.as_mut_unchecked() = value;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3, 4, 5, 6];

    // Predicate state follows reverse visitation, rejecting 4 and 1 rather than 3 and 6.
    assert_eq!(runner.run(a, 0, 0).await?, 96532);
    // A rejected suffix does not invoke the reverse-fold closure.
    assert_eq!(runner.run(a, 4, 0).await?, 965);
    // An all-rejected source leaves the nonzero seed untouched.
    assert_eq!(runner.run(a, 7, 0).await?, 9);
    // Backward preconsumption must preserve the predicate state moved into rfold.
    assert_eq!(runner.run(a, 0, 1).await?, 9532);
    // Only a rejected item remains after the final backward match.
    assert_eq!(runner.run(a, 0, 4).await?, 9);
    // An exhausted source does not invoke either closure.
    assert_eq!(runner.run(a, 0, 5).await?, 9);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
