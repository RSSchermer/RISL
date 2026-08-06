#![cfg(feature = "test_variable_pointer_if_else_slice_of_refs")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;
use risl::gpu;

test_runner! {
    name: Runner,
    inputs: {
        SELECTOR: u32 as Uniform<u32>,
        VALUES: [u32; 5] as Storage<[u32; 5]>,
    },
    result: [u32; 4],
    shader: {
        let values = VALUES.as_ref();
        let result = unsafe {
            RESULT.as_mut_unchecked()
        };

        let short: [&u32; 2] = unsafe { [values.get_unchecked(0), values.get_unchecked(1)] };
        let long: [&u32; 3] = unsafe {
            [
                values.get_unchecked(2),
                values.get_unchecked(3),
                values.get_unchecked(4),
            ]
        };

        let selected = if *SELECTOR == 0 {
            short.as_slice()
        } else {
            long.as_slice()
        };

        let mut i = 0;

        while i < selected.len() {
            unsafe {
                *result.get_unchecked_mut(i) = **selected.get_unchecked(i);
            }

            i += 1;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    let values = [10, 20, 30, 40, 50];

    assert_eq!(runner.run(0, values).await?, [10, 20, 0, 0]);
    assert_eq!(runner.run(1, values).await?, [30, 40, 50, 0]);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
