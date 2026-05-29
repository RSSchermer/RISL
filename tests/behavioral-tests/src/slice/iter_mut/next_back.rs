#![cfg(feature = "test_slice_iter_mut_next_back")]

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
}

test_runner! {
    name: Runner,
    inputs: {
        VALUES: [u32] as StorageMut<[u32]>,
    },
    result: Results,
    shader: {
        let mut iter = unsafe { VALUES.as_mut_unchecked().iter_mut() };

        let last = match iter.next_back() {
            Some(v) => {
                *v += 1;
                *v
            }
            None => u32::MAX,
        };

        let second_to_last = match iter.next_back() {
            Some(v) => {
                *v += 2;
                *v
            }
            None => u32::MAX,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                last,
                second_to_last,
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
            last: 41,
            second_to_last: 32,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
