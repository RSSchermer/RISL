#![cfg(feature = "test_slice_iter_clone")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    initial_val: u32,
    cloned_val: u32,
    after_advance_orig_val: u32,
    after_advance_cloned_val: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        VALUES: [u32] as Storage<[u32]>,
    },
    result: Results,
    shader: {
        let mut iter = VALUES.as_ref().iter();

        let mut cloned_iter = iter.clone();

        let initial_val = *iter.next().unwrap_or(&u32::MAX);
        let cloned_val = *cloned_iter.next().unwrap_or(&u32::MAX);

        iter.next(); // original is now at index 2, clone is at index 1

        let after_advance_orig_val = *iter.next().unwrap_or(&u32::MAX);
        let after_advance_cloned_val = *cloned_iter.next().unwrap_or(&u32::MAX);

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                initial_val,
                cloned_val,
                after_advance_orig_val,
                after_advance_cloned_val,
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
            initial_val: 10,
            cloned_val: 10,
            after_advance_orig_val: 30,
            after_advance_cloned_val: 20,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
