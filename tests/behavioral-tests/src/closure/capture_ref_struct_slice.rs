#![cfg(feature = "test_closure_capture_ref_struct_slice")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;
use risl::gpu;

#[gpu]
struct Wrapper<'a> {
    slice: &'a [u32],
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
        VALUES: [u32] as Storage<[u32]>,
    },
    result: u32,
    shader: {
        // Test capturing a slice wrapped in a struct. This should result in rustc assigned a
        // scalar-pair ABI for the struct, which rislc needs to handle correctly.

        let values = VALUES.as_ref();
        let wrapper = Wrapper { slice: values };

        let closure = || {
            let mut sum = 0;

            for v in wrapper.slice {
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
