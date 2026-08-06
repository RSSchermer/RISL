#![cfg(feature = "test_variable_pointer_loop_interleaved_variable_pointer_promotion")]

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

struct State<'a> {
    iteration: u32,
    active: &'a u32,
}

test_runner! {
    name: Runner,
    inputs: {
        ITERATIONS: u32 as Uniform<u32>,
        VALUES: Values as Storage<Values>,
    },
    result: [u32; 6],
    shader: {
        let values = *VALUES;
        let result = unsafe {
            RESULT.as_mut_unchecked()
        };

        let mut a = &values.a;
        let b = &values.b;
        let c = &values.c;

        let mut state = State {
            iteration: 0,
            active: a,
        };

        while state.iteration < *ITERATIONS {
            unsafe {
                *result.get_unchecked_mut(state.iteration as usize) = *state.active;
            }

            if state.iteration % 2 == 0 {
                state.active = c;
            } else {
                state.active = a;
            }

            a = b;

            state.iteration += 1;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(
        runner.run(5, Values { a: 1, b: 2, c: 3 }).await?,
        [1, 3, 2, 3, 2, 0]
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
