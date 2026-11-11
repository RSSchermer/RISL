#![cfg(feature = "test_num_u32_saturating_add")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: Runner,
    inputs: {
        A: u32 as Uniform<u32>,
        B: u32 as Uniform<u32>,
    },
    result: u32,
    shader: {
        let res = (*A).saturating_add(*B);

        unsafe {
            *RESULT.as_mut_unchecked() = res;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    // Ordinary addition and zero do not saturate.
    assert_eq!(runner.run(3, 4).await?, 7);
    assert_eq!(runner.run(0, 0).await?, 0);

    // Exact maximum sums fit the u32 range.
    assert_eq!(runner.run(u32::MAX, 0).await?, 4294967295);
    assert_eq!(runner.run(0, u32::MAX).await?, 4294967295);
    assert_eq!(runner.run(4294967294, 1).await?, 4294967295);

    // Overflow saturates at MAX instead of wrapping, for both small and extreme values.
    assert_eq!(runner.run(u32::MAX, 1).await?, 4294967295);
    assert_eq!(runner.run(u32::MAX, 2).await?, 4294967295);
    assert_eq!(runner.run(u32::MAX, u32::MAX).await?, 4294967295);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
