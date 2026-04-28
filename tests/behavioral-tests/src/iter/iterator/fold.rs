#![cfg(feature = "test_iter_iterator_fold")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;

use crate::iter::iterator::SimpleIter;

test_runner! {
    name: Runner,
    inputs: {
        VALUES: [u32; 4] as Storage<[u32; 4]>,
    },
    result: u32,
    shader: {
        let mut iter = SimpleIter {
            values: VALUES.clone(),
            index: 0,
        };

        let result = iter.fold(0, |acc, x| acc + x);

        unsafe {
            *RESULT.as_mut_unchecked() = result;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run([10u32, 20u32, 30u32, 40u32]).await?, 100);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
