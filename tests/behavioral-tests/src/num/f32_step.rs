#![cfg(feature = "test_num_f32_step")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::num::F32Ext;

test_runner! {
    name: Runner,
    inputs: {
        A: f32 as Uniform<f32>,
    },
    result: f32,
    shader: {
        let res = (*A).step(0.5);

        unsafe {
            *RESULT.as_mut_unchecked() = res;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run(0.0).await?, 0.0);
    assert_eq!(runner.run(0.4).await?, 0.0);
    assert_eq!(runner.run(0.5).await?, 1.0);
    assert_eq!(runner.run(0.6).await?, 1.0);
    assert_eq!(runner.run(1.0).await?, 1.0);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
