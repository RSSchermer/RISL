#![cfg(feature = "test_variable_pointer_loop_alternating_no_cycle")]

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
}

test_runner! {
    name: Runner,
    inputs: {
        ITERATIONS: u32 as Uniform<u32>,
        VALUES: Values as Storage<Values>,
    },
    result: [u32; 6],
    shader: {
        let values = VALUES.as_ref();
        let result = unsafe {
            RESULT.as_mut_unchecked()
        };

        let a = &values.a;
        let b = &values.b;

        let mut active = a;
        let mut i = 0;

        while i < *ITERATIONS {
            unsafe {
                *result.get_unchecked_mut(i as usize) = *active;
            }

            if i % 2 == 0 {
                active = b;
            } else {
                active = a;
            }

            i += 1;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(
        runner.run(5, Values { a: 1, b: 2 }).await?,
        [1, 2, 1, 2, 1, 0]
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
