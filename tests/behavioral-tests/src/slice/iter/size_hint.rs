#![cfg(feature = "test_slice_iter_size_hint")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct SizeHint {
    low: u32,
    high: u32,
}

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    initial: SizeHint,
    after_one: SizeHint,
    after_all: SizeHint,
}

test_runner! {
    name: Runner,
    inputs: {
        VALUES: [u32] as Storage<[u32]>,
    },
    result: Results,
    shader: {
        let mut iter = VALUES.as_ref().iter();

        let (low, high) = iter.size_hint();
        let initial = SizeHint {
            low: low as u32,
            high: if let Some(high) = high { high as u32 } else { u32::MAX },
        };

        iter.next();

        let (low, high) = iter.size_hint();
        let after_one = SizeHint {
            low: low as u32,
            high: if let Some(high) = high { high as u32 } else { u32::MAX },
        };

        while let Some(_) = iter.next() {}

        let (low, high) = iter.size_hint();
        let after_all = SizeHint {
            low: low as u32,
            high: if let Some(high) = high { high as u32 } else { u32::MAX },
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                initial,
                after_one,
                after_all,
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
            initial: SizeHint { low: 4, high: 4 },
            after_one: SizeHint { low: 3, high: 3 },
            after_all: SizeHint { low: 0, high: 0 },
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
