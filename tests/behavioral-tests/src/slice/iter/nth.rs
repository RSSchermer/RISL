#![cfg(feature = "test_slice_iter_nth")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    first: u32,
    second: u32,
    after_nth: u32,
    none_val: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        VALUES: [u32] as Storage<[u32]>,
    },
    result: Results,
    shader: {
        let mut iter = VALUES.as_ref().iter();
        let first = match iter.nth(0) {
            Some(v) => *v,
            None => u32::MAX,
        };
        let second = match iter.next() {
            Some(v) => *v,
            None => u32::MAX,
        };

        let mut iter = VALUES.as_ref().iter();
        let after_nth = match iter.nth(2) {
            Some(v) => *v,
            None => u32::MAX,
        };
        let none_val = match iter.nth(10) {
            Some(v) => *v,
            None => u32::MAX,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                first,
                second,
                after_nth,
                none_val,
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
            first: 10,
            second: 20,
            after_nth: 30,
            none_val: u32::MAX,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
