#![cfg(feature = "test_slice_cmp_ref_mut_partial_eq_array")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;

#[derive(Copy, Clone, abi::Sized)]
struct Values {
    v0: u32,
    v1: u32,
    v2: u32,
    v3: u32,
}

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
struct Results {
    eq: u32,
    ne: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        SLICE: [u32] as StorageMut<[u32]>,
        VALUES: Values as Uniform<Values>,
    },
    result: Results,
    shader: {
        unsafe {
            let slice = SLICE.as_mut_unchecked();
            let array = [VALUES.v0, VALUES.v1, VALUES.v2, VALUES.v3];

            let eq = slice == array;
            let ne = slice != array;

            *RESULT.as_mut_unchecked() = Results {
                eq: eq as u32,
                ne: ne as u32
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    // Case 1: Equal
    assert_eq!(
        runner
            .run(
                vec![10, 20, 30, 40],
                Values {
                    v0: 10,
                    v1: 20,
                    v2: 30,
                    v3: 40,
                },
            )
            .await?,
        Results { eq: 1, ne: 0 }
    );

    // Case 1: Not equal (same length)
    assert_eq!(
        runner
            .run(
                vec![10, 20, 30, 50],
                Values {
                    v0: 10,
                    v1: 20,
                    v2: 30,
                    v3: 40,
                },
            )
            .await?,
        Results { eq: 0, ne: 1 }
    );

    // Case 2: Not equal (different length)
    assert_eq!(
        runner
            .run(
                vec![10, 20, 30],
                Values {
                    v0: 10,
                    v1: 20,
                    v2: 30,
                    v3: 40,
                },
            )
            .await?,
        Results { eq: 0, ne: 1 }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
