#![cfg(feature = "test_condition_partial_initialization")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: PartialInitializationRunner,
    inputs: {
        OUTER: u32 as Uniform<u32>,
        INNER: u32 as Uniform<u32>,
    },
    result: u32,
    shader: {
        let value;

        if *OUTER == 0 {
            if *INNER == 0 {
                unsafe {
                    *RESULT.as_mut_unchecked() = 11;
                }

                return;
            }

            value = 22;
        } else {
            value = 33;
        }

        unsafe {
            *RESULT.as_mut_unchecked() = value;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = PartialInitializationRunner::init().await?;

    assert_eq!(runner.run(0, 0).await?, 11);
    assert_eq!(runner.run(0, 1).await?, 22);
    assert_eq!(runner.run(1, 0).await?, 33);
    assert_eq!(runner.run(1, 1).await?, 33);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
