#![cfg(feature = "test_iter_adapters_flat_map_count")]

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
        let mut iter = A
            .iter()
            .flat_map(|len| unsafe { B.get_unchecked(..(*len as usize)) }.iter());
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

        let result = iter.count() as u32;

        unsafe {
            *RESULT.as_mut_unchecked() = result;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[0, 2, 0, 3, 0];
    let b: &[u32] = &[1, 2, 3];
    let empty: &[u32] = &[0, 0, 0];

    // Count inner items rather than outer lengths, ignoring empty inners.
    assert_eq!(runner.run(a, b, 0, 0).await?, 5);
    // Include a pending front inner's remainder.
    assert_eq!(runner.run(a, b, 1, 0).await?, 4);
    // Include a pending back inner's remainder.
    assert_eq!(runner.run(a, b, 0, 1).await?, 4);
    // Combine both pending inners without counting consumed items.
    assert_eq!(runner.run(a, b, 1, 1).await?, 3);
    // Crossing the two ends leaves no items to count.
    assert_eq!(runner.run(a, b, 3, 3).await?, 0);
    // Only empty inner iterators.
    assert_eq!(runner.run(empty, b, 0, 0).await?, 0);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
