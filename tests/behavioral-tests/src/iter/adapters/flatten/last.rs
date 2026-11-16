#![cfg(feature = "test_iter_adapters_flatten_last")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

const NONE: u32 = u32::MAX;

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
            .map(|len| unsafe { B.get_unchecked(..(*len as usize)) })
            .flatten();
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

        let result = match iter.last() {
            Some(value) => *value,
            None => NONE,
        };

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

    assert_eq!(runner.run(a, b, 0, 0).await?, 3);
    assert_eq!(runner.run(a, b, 1, 0).await?, 3);
    assert_eq!(runner.run(a, b, 0, 2).await?, 1);
    assert_eq!(runner.run(a, b, 0, 3).await?, 2);
    // Crossing the two ends leaves no last item.
    assert_eq!(runner.run(a, b, 3, 3).await?, NONE);
    // All-empty inners do not have a last item.
    assert_eq!(runner.run(empty, b, 0, 0).await?, NONE);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
