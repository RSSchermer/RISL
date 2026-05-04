#![cfg(feature = "test_iter_iterator_size_hint")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

use crate::iter::iterator::SimpleIter;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    low: u32,
    high: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        VALUES: [u32; 4] as Storage<[u32; 4]>,
    },
    result: Results,
    shader: {
        let iter = SimpleIter {
            values: VALUES.clone(),
            index: 0,
        };

        let (low, high) = iter.size_hint();
        let low = low as u32;
        let high = if let Some(high) = high { high as u32 } else { u32::MAX };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                low,
                high,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(
        runner.run([10u32, 20u32, 30u32, 40u32]).await?,
        Results {
            low: 0,
            high: u32::MAX,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
