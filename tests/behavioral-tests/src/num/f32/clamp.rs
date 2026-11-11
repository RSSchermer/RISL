#![cfg(feature = "test_num_f32_clamp")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;

test_runner! {
    name: Runner,
    inputs: {
        A: f32 as Uniform<f32>,
        MIN: f32 as Uniform<f32>,
        MAX: f32 as Uniform<f32>,
    },
    result: f32,
    shader: {
        let res = (*A).clamp(*MIN, *MAX);

        unsafe {
            *RESULT.as_mut_unchecked() = res;
        }
    },
}

#[test]
fn test() -> Result<(), Box<dyn Error>> {
    pollster::block_on(async {
        let runner = Runner::init().await?;

        assert_eq!(runner.run(10.0, 5.0, 15.0).await?, 10.0);
        assert_eq!(runner.run(2.0, 5.0, 15.0).await?, 5.0);
        assert_eq!(runner.run(20.0, 5.0, 15.0).await?, 15.0);
        assert_eq!(runner.run(-10.0, -15.0, -5.0).await?, -10.0);
        assert_eq!(runner.run(-20.0, -15.0, -5.0).await?, -15.0);
        assert_eq!(runner.run(0.0, -15.0, -5.0).await?, -5.0);

        Ok(())
    })
}
