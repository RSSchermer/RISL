#![cfg(feature = "test_iter_adapters_rev_next")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

const NONE: u32 = u32::MAX;

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
    },
    result: [u32; 5],
    shader: {
        let mut iter = A.iter().copied().rev();

        unsafe {
            let result = RESULT.as_mut_unchecked();

            let mut i = 0;

            while i < 5 {
                *result.get_unchecked_mut(i) = match iter.next() {
                    Some(value) => value,
                    None => NONE,
                };

                i += 1;
            }
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3];

    // next consumes the source from the back, then repeatedly returns None.
    assert_eq!(runner.run(a).await?, [3, 2, 1, NONE, NONE]);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
