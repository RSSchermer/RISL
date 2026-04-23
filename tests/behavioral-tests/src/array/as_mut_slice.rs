#![cfg(feature = "test_array_as_mut_slice")]

use std::error::Error;
use std::ops::Deref;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: Runner,
    inputs: {
        INDEX: u32 as Uniform<u32>,
    },
    result: [u32; 4],
    shader: {
        unsafe {
            let slice = RESULT.as_mut_unchecked().as_mut_slice();

            *slice.get_unchecked_mut(*INDEX as usize) = 1;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run(0).await?, [1, 0, 0, 0]);
    assert_eq!(runner.run(1).await?, [0, 1, 0, 0]);
    assert_eq!(runner.run(2).await?, [0, 0, 1, 0]);
    assert_eq!(runner.run(3).await?, [0, 0, 0, 1]);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
