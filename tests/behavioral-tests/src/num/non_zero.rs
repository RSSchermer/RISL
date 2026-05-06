#![cfg(feature = "test_num_non_zero")]

use std::error::Error;
use std::num::NonZero;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;

test_runner! {
    name: Runner,
    inputs: {
        N: u32 as Uniform<u32>,
    },
    result: u32,
    shader: {
        let res = match NonZero::new(*N) {
            Some(n) => n.get(),
            None => 0,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = res;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run(0).await?, 0);
    assert_eq!(runner.run(42).await?, 42);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
