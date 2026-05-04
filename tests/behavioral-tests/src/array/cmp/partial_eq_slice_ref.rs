#![cfg(feature = "test_array_cmp_partial_eq_slice_ref")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Copy, Clone, abi::Sized)]
#[gpu]
struct Values {
    v0: u32,
    v1: u32,
    v2: u32,
    v3: u32,
}

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    eq: u32,
    ne: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        VALUES: Values as Uniform<Values>,
        SLICE: [u32] as Storage<[u32]>,
    },
    result: Results,
    shader: {
        let array = [VALUES.v0, VALUES.v1, VALUES.v2, VALUES.v3];
        let slice = SLICE.as_ref();

        let eq = array == slice;
        let ne = array != slice;

        unsafe {
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
                Values {
                    v0: 10,
                    v1: 20,
                    v2: 30,
                    v3: 40,
                },
                vec![10, 20, 30, 40]
            )
            .await?,
        Results { eq: 1, ne: 0 }
    );

    // Case 1: Not equal (same length)
    assert_eq!(
        runner
            .run(
                Values {
                    v0: 10,
                    v1: 20,
                    v2: 30,
                    v3: 40,
                },
                vec![10, 20, 30, 50]
            )
            .await?,
        Results { eq: 0, ne: 1 }
    );

    // Case 2: Not equal (different length)
    assert_eq!(
        runner
            .run(
                Values {
                    v0: 10,
                    v1: 20,
                    v2: 30,
                    v3: 40,
                },
                vec![10, 20, 30]
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
