#![cfg(feature = "test_closure_capture_ref_single_slice")]

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
        VALUES: [u32] as Storage<[u32]>,
    },
    result: u32,
    shader: {
        // Rustc may assign a scalar-pair ABI for closure values that capture a single value with
        // a scalar-pair ABI; rislc must handle these correctly.

        let values = VALUES.as_ref();

        let closure = || {
            let mut sum = 0;

            for v in values {
                sum += *v;
            }

            sum
        };

        unsafe {
            *RESULT.as_mut_unchecked() = closure_caller(closure);
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run(vec![10, 20, 30]).await?, 60);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
