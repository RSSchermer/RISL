#![cfg(feature = "test_slice_iter_last")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    full_last: u32,
    exhausted_last: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        VALUES: [u32] as Storage<[u32]>,
    },
    result: Results,
    shader: {
        let full_last = match VALUES.as_ref().iter().last() {
            Some(v) => *v,
            None => u32::MAX,
        };

        let mut iter = VALUES.as_ref().iter();
        while let Some(_) = iter.next() {}
        let exhausted_last = match iter.last() {
            Some(v) => *v,
            None => u32::MAX,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                full_last,
                exhausted_last,
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
            full_last: 40,
            exhausted_last: u32::MAX,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
