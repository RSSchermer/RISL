#![cfg(feature = "test_iter_adapters_chain_next_back")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

const NONE: u32 = u32::MAX;

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        B: [u32] as Storage<[u32]>,
    },
    result: [u32; 9],
    shader: {
        let mut iter = A.iter().chain(B.iter());

        unsafe {
            let res = RESULT.as_mut_unchecked();
            let mut i = 0;

            while i < 9 {
                *res.get_unchecked_mut(i) = match iter.next_back() {
                    Some(value) => *value,
                    None => NONE,
                };

                i += 1;
            }
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3, 4];
    let b: &[u32] = &[5, 6, 7, 8];

    // Visit the back source before the front source, then report exhaustion.
    assert_eq!(runner.run(a, b).await?, [8, 7, 6, 5, 4, 3, 2, 1, NONE]);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
