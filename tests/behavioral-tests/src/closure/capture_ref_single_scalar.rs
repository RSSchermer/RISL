#![cfg(feature = "test_closure_capture_ref_single_scalar")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;
use risl::gpu;

#[gpu]
fn closure_caller<F>(mut f: F)
where
    F: FnMut(),
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
        // Test capturing a single scalar. While generally rustc assigns "aggregate" ABIs to closure
        // values, for a closure that captures a single scalar, it can assign a scalar-ABI; rislc
        // needs to handle this correctly.
        
        unsafe {
            let mut val = *VALUE;

            let closure = || {
                let val_ref = &mut val;

                *val_ref += 1;
            };

            // Don't call the closure directly, pass it to a helper function to ensure we also
            // exercise function-ABI related behavior.
            closure_caller(closure);

            *RESULT.as_mut_unchecked() = val;
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
