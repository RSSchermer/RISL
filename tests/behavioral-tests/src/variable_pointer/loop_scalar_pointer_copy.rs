#![cfg(feature = "test_variable_pointer_loop_scalar_pointer_copy")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Clone, Copy, abi::Sized)]
#[gpu]
struct Values {
    a: u32,
    b: u32,
    c: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        ITERATIONS: u32 as Uniform<u32>,
        VALUES: Values as Storage<Values>,
    },
    result: u32,
    shader: {
        let values = VALUES.as_ref();
        let replacement = &values.c;
        let mut saved = &values.a;
        let mut current = &values.b;
        let mut i = 0;

        while i < *ITERATIONS {
            saved = current;
            current = replacement;

            i += 1;
        }

        unsafe {
            *RESULT.as_mut_unchecked() = *saved;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    // With no iterations, saved retains its initial target.
    assert_eq!(
        runner
            .run(
                0,
                Values {
                    a: 10,
                    b: 20,
                    c: 30
                }
            )
            .await?,
        10
    );
    // Load emulation must follow the normalized pointer, not the invariant initial target.
    assert_eq!(
        runner
            .run(
                1,
                Values {
                    a: 10,
                    b: 20,
                    c: 30
                }
            )
            .await?,
        20
    );
    // The next iteration copies current's replacement, not its initial target.
    assert_eq!(
        runner
            .run(
                2,
                Values {
                    a: 10,
                    b: 20,
                    c: 30
                }
            )
            .await?,
        30
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
