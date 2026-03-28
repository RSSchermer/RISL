#![cfg(feature = "test_while_loop")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: WhileLoopRunner,
    inputs: {
        COUNT: u32 as Uniform<u32>,
    },
    result: u32,
    shader: {
        let mut i = 0u32;
        let mut sum = 0u32;

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
    let runner = WhileLoopRunner::init().await?;

    // sum(0..0) = 0
    assert_eq!(runner.run(0u32).await?, 0u32);
    // sum(0..1) = 0
    assert_eq!(runner.run(1u32).await?, 0u32);
    // sum(0..2) = 0 + 1 = 1
    assert_eq!(runner.run(2u32).await?, 1u32);
    // sum(0..5) = 0 + 1 + 2 + 3 + 4 = 10
    assert_eq!(runner.run(5u32).await?, 10u32);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
