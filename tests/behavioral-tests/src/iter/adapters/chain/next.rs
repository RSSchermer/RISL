#![cfg(feature = "test_iter_adapters_chain_next")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

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

            while let Some(v) = iter.next() {
                *res.get_unchecked_mut(i) = *v;
                i += 1;
            }
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    let a = vec![1, 2, 3, 4];
    let b = vec![5, 6, 7, 8];

    assert_eq!(runner.run(a, b).await?, [1, 2, 3, 4, 5, 6, 7, 8, 0]);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
