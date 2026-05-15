#![cfg(feature = "test_while_loop")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: Runner,
    inputs: {
        COUNT: u32 as Uniform<u32>,
    },
    result: u32,
    shader: {
        let mut i = 0;
        let mut sum = 0;

        while i < *COUNT {
            sum += i;
            i += 1;
        }

        unsafe {
            *RESULT.as_mut_unchecked() = sum;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    // sum(0..0) = 0
    assert_eq!(runner.run(0).await?, 0);
    // sum(0..1) = 0
    assert_eq!(runner.run(1).await?, 0);
    // sum(0..2) = 0 + 1 = 1
    assert_eq!(runner.run(2).await?, 1);
    // sum(0..5) = 0 + 1 + 2 + 3 + 4 = 10
    assert_eq!(runner.run(5).await?, 10);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
