#![cfg(feature = "test_slice_iter_next_back")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    last: u32,
    second_to_last: u32,
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

        let last = match iter.next_back() {
            Some(v) => *v,
            None => u32::MAX,
        };

        let second_to_last = match iter.next_back() {
            Some(v) => *v,
            None => u32::MAX,
        };

        // Advance from front
        iter.next();
        iter.next();

        let none_val = match iter.next_back() {
            Some(v) => *v,
            None => u32::MAX,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                last,
                second_to_last,
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
            last: 40,
            second_to_last: 30,
            none_val: u32::MAX, // After taking 40, 30 from back and 10, 20 from front, it should be empty
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
