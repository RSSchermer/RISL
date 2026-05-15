#![cfg(feature = "test_array_as_slice")]

use std::error::Error;
use std::ops::Deref;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: Runner,
    inputs: {
        VALUES: [u32; 4] as Storage<[u32; 4]>,
    },
    result: u32,
    shader: {
        let slice = VALUES.deref().as_slice();

        if let Some(val) = slice.get(0) {
            unsafe {
                *RESULT.as_mut_unchecked() = *val;
            }
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run([10, 20, 30, 40]).await?, 10);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
