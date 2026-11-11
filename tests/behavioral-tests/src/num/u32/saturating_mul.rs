#![cfg(feature = "test_num_u32_saturating_mul")]

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
        let res = (*A).saturating_mul(*B);

        unsafe {
            *RESULT.as_mut_unchecked() = res;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    // Ordinary multiplication and zero in either position don't saturate.
    assert_eq!(runner.run(3, 4).await?, 12);
    assert_eq!(runner.run(0, 0).await?, 0);
    assert_eq!(runner.run(0, u32::MAX).await?, 0);
    assert_eq!(runner.run(u32::MAX, 0).await?, 0);

    // Overflow produces the MAX value rather than wrapping.
    assert_eq!(runner.run(u32::MAX, 2).await?, 4294967295);
    assert_eq!(runner.run(65536, 65536).await?, 4294967295);
    assert_eq!(runner.run(u32::MAX, u32::MAX).await?, 4294967295);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
