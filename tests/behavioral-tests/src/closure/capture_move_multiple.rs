#![cfg(feature = "test_closure_capture_move_multiple")]

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
        A: u32 as Uniform<u32>,
        B: u32 as Uniform<u32>,
        C: u32 as Uniform<u32>,
    },
    result: u32,
    shader: {
        // Capture at least three values to ensure closure value is assigned an "aggregate" ABI, not
        // a scalar or scalar-pair ABI.

        let a = *A;
        let b = *B;
        let c = *C;

        let closure = move || a + b + c;

        unsafe {
            *RESULT.as_mut_unchecked() = closure_caller(closure);
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run(10, 20, 30).await?, 60);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
