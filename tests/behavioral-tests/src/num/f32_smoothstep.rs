#![cfg(feature = "test_num_f32_smoothstep")]

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
        let res = (*A).smoothstep(0.0, 1.0);

        unsafe {
            *RESULT.as_mut_unchecked() = res;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    let cases = [
        (0.0f32, 0.0f32),
        (0.5f32, 0.5f32),
        (1.0f32, 1.0f32),
        (-0.5f32, 0.0f32),
        (1.5f32, 1.0f32),
    ];

    for (input, expected) in cases {
        let actual = runner.run(input).await?;
        assert!(
            (actual - expected).abs() < 1e-6,
            "smoothstep(0.0, 1.0, {}) = {}, expected {}",
            input,
            actual,
            expected
        );
    }

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
