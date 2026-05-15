#![cfg(feature = "test_closure_capture_move_single_scalar")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;
use risl::gpu;

#[gpu]
fn closure_caller<F>(f: F) -> u32
where
    F: Fn() -> u32,
{
    f()
}

test_runner! {
    name: Runner,
    inputs: {
        VALUE: u32 as Uniform<u32>,
    },
    result: u32,
    shader: {
        let val = *VALUE;
        let closure = move || val + 1;

        unsafe {
            *RESULT.as_mut_unchecked() = closure_caller(closure);
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run(41u32).await?, 42u32);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
