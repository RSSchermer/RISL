#![cfg(feature = "test_iter_adapters_flatten_fold")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        B: [u32] as Storage<[u32]>,
        FRONT: u32 as Uniform<u32>,
        BACK: u32 as Uniform<u32>,
    },
    result: u32,
    shader: {
        let mut mapper_calls = 0;
        let mut iter = A.iter().map(|len| {
            mapper_calls += 1;

            let offset = mapper_calls;

            unsafe { B.get_unchecked(..(*len as usize)) }
                .iter()
                .map(move |value| *value + offset)
        }).flatten();

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

        let value = iter.fold(9, |acc, value| acc * 10 + value);

        unsafe {
            *RESULT.as_mut_unchecked() = value;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[0, 2, 0, 2, 0];
    let b: &[u32] = &[1, 2];
    let empty: &[u32] = &[0, 0, 0];

    // Empty inner iterators still advance outer mapping state; visit 3, 4, 5, 6 in order.
    assert_eq!(runner.run(a, b, 0, 0).await?, 93456);
    // Resume inside the pending front iterator.
    assert_eq!(runner.run(a, b, 1, 0).await?, 9456);
    // Backward mapping happens first; fold finishes with the pending back iterator.
    assert_eq!(runner.run(a, b, 0, 1).await?, 9563);
    // Preserve both pending inner iterators, with an empty outer entry between them.
    assert_eq!(runner.run(a, b, 1, 1).await?, 945);
    // Only the final item of the second nonempty inner iterator remains.
    assert_eq!(runner.run(a, b, 3, 0).await?, 96);
    // Exhausting the iterator before folding never calls the fold closure.
    assert_eq!(runner.run(a, b, 5, 0).await?, 9);
    // An all-empty mapping never calls the fold closure.
    assert_eq!(runner.run(empty, b, 0, 0).await?, 9);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
