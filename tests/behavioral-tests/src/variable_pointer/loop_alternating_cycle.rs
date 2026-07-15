#![cfg(feature = "test_variable_pointer_loop_alternating_cycle")]

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

        let mut a = &values.a;
        let mut b = &values.b;

        let mut i = 0;

        while i < *ITERATIONS {
            unsafe {
                *result.get_unchecked_mut(i as usize) = *a;
            }

            let tmp = a;

            a = b;
            b = tmp;

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
