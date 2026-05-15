#![cfg(feature = "test_closure_capture_ref_multiple")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;
use risl::gpu;

#[gpu]
struct Wrapper {
    a: u32,
}

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
        // Wrap the values in a struct that does not implement Copy, to ensure the values are
        // captured by reference. Capture at least three values to ensure closure value is assigned
        // an "aggregate" ABI, not a scalar or scalar-pair ABI.
        let w1 = Wrapper { a: *A };
        let w2 = Wrapper { a: *B };
        let w3 = Wrapper { a: *C };

        let closure = || w1.a + w2.a + w3.a;

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
