#![cfg(feature = "test_array_cmp_partial_eq_array")]

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
        A_VALUES: Values as Uniform<Values>,
        B_VALUES: Values as Uniform<Values>,
    },
    result: Results,
    shader: {
        let a = [A_VALUES.v0, A_VALUES.v1, A_VALUES.v2, A_VALUES.v3];
        let b = [B_VALUES.v0, B_VALUES.v1, B_VALUES.v2, B_VALUES.v3];

        let eq = a == b;
        let ne = a != b;

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
                Values {
                    v0: 10,
                    v1: 20,
                    v2: 30,
                    v3: 40,
                }
            )
            .await?,
        Results { eq: 1, ne: 0 }
    );

    // Case 1: Not equal
    assert_eq!(
        runner
            .run(
                Values {
                    v0: 10,
                    v1: 20,
                    v2: 30,
                    v3: 40,
                },
                Values {
                    v0: 10,
                    v1: 20,
                    v2: 30,
                    v3: 50,
                }
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
