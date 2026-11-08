#![cfg(feature = "test_iter_adapters_filter_map_fold")]

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
        let mut mapper_calls = 0;
        let mut iter = A.iter().filter_map(|value| {
            mapper_calls += 1;

            if *value >= *MIN && mapper_calls % 3 != 0 {
                Some(*value * 10 + mapper_calls)
            } else {
                None
            }
        });

        let mut i = 0;

        while i < *PRECONSUME {
            iter.next();
            i += 1;
        }

        let value = iter.fold(9, |acc, value| acc * 100 + value);

        unsafe {
            *RESULT.as_mut_unchecked() = value;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3, 4, 5, 6];

    // Fold mapped values 11, 22, 44, 55 in order, rejecting every third source item.
    assert_eq!(runner.run(a, 0, 0).await?, 911224455);
    // Rejected prefix items still advance the mapper's state before producing 44 and 55.
    assert_eq!(runner.run(a, 4, 0).await?, 94455);
    // An all-rejected source never calls the fold closure, preserving its seed.
    assert_eq!(runner.run(a, 7, 0).await?, 9);
    // Moving the mapper into fold must preserve state established by next.
    assert_eq!(runner.run(a, 0, 1).await?, 9224455);
    // Only a rejected tail remains after consuming the final mapped value.
    assert_eq!(runner.run(a, 0, 4).await?, 9);
    // A fully exhausted source does not invoke either closure.
    assert_eq!(runner.run(a, 0, 5).await?, 9);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
