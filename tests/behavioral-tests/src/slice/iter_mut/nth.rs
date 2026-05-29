#![cfg(feature = "test_slice_iter_mut_nth")]

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
}

test_runner! {
    name: Runner,
    inputs: {
        VALUES: [u32] as StorageMut<[u32]>,
    },
    result: Results,
    shader: {
        let mut iter = unsafe { VALUES.as_mut_unchecked().iter_mut() };

        let first = match iter.nth(0) {
            Some(first_ref) => {
                *first_ref += 1;
                *first_ref
            }
            None => u32::MAX,
        };

        let second = match iter.next() {
            Some(second_ref) => {
                *second_ref += 2;
                *second_ref
            }
            None => u32::MAX,
        };

        let mut iter = unsafe { VALUES.as_mut_unchecked().iter_mut() };
        let after_nth = match iter.nth(2) {
            Some(after_nth_ref) => {
                *after_nth_ref += 3;
                *after_nth_ref
            }
            None => u32::MAX,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                first,
                second,
                after_nth,
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
            first: 11,
            second: 22,
            after_nth: 33,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
