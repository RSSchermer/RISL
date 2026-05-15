#![cfg(feature = "test_closure_capture_ref_struct_scalar")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[gpu]
struct Wrapper {
    a: u32,
}

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
        unsafe {
            // Test passing a scalar wrapped in a struct. This should result in rustc assigned a
            // scalar-ABI for the struct, which rislc needs to handle correctly.
            let mut val = Wrapper { a: *VALUE };

            let closure = || {
                let val_ref = &mut val;

                val_ref.a += 1;
            };

            // Don't call the closure directly, pass it to a helper function to ensure we also
            // exercise function-ABI related behavior.
            closure_caller(closure);

            *RESULT.as_mut_unchecked() = val.a;
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
