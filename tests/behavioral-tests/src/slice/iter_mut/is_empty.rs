#![cfg(feature = "test_slice_iter_mut_is_empty")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    is_empty_before: u32,
    is_empty_after: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        VALUES: [u32; 1] as Storage<[u32; 1]>,
    },
    result: Results,
    shader: {
        let mut values = *VALUES;
        let mut iter = values.iter_mut();

        let is_empty_before = if iter.is_empty() { 1 } else { 0 };
        iter.next();
        let is_empty_after = if iter.is_empty() { 1 } else { 0 };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                is_empty_before,
                is_empty_after,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(
        runner.run([1]).await?,
        Results {
            is_empty_before: 0,
            is_empty_after: 1,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
