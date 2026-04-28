#![cfg(feature = "test_iter_iterator_last")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;

use crate::iter::iterator::SimpleIter;

test_runner! {
    name: Runner,
    inputs: {
        START: u32 as Uniform<u32>,
        VALUES: [u32; 4] as Storage<[u32; 4]>,
    },
    result: u32,
    shader: {
        let iter = SimpleIter {
            values: VALUES.clone(),
            index: *START as usize,
        };

        let result = if let Some(value) = iter.last() {
            value
        } else {
            u32::MAX
        };

        unsafe {
            *RESULT.as_mut_unchecked() = result;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    let values = [10u32, 20u32, 30u32, 40u32];

    assert_eq!(runner.run(0u32, values).await?, 40u32);
    assert_eq!(runner.run(3u32, values).await?, 40u32);
    assert_eq!(runner.run(4u32, values).await?, u32::MAX);
    assert_eq!(runner.run(10u32, values).await?, u32::MAX);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
