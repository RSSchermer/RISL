#![cfg(feature = "test_slice_iter_nth_back")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    nth_back_0: u32,
    nth_back_1: u32,
    nth_back_10: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        VALUES: [u32] as Storage<[u32]>,
    },
    result: Results,
    shader: {
        let mut iter = VALUES.as_ref().iter();

        // Take last element (40)
        let nth_back_0 = match iter.nth_back(0) {
            Some(v) => *v,
            None => u32::MAX,
        };

        // Skip one (30) and take next (20)
        let nth_back_1 = match iter.nth_back(1) {
            Some(v) => *v,
            None => u32::MAX,
        };

        // Exhaust
        let nth_back_10 = match iter.nth_back(10) {
            Some(v) => *v,
            None => u32::MAX,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                nth_back_0,
                nth_back_1,
                nth_back_10,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    let values = vec![10, 20, 30, 40];
    let results = runner.run(values).await?;

    assert_eq!(
        results,
        Results {
            nth_back_0: 40,
            nth_back_1: 20,
            nth_back_10: u32::MAX,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
