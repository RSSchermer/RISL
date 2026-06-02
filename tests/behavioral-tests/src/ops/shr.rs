#![cfg(feature = "test_ops_shr")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct ResultsU32 {
    val_val: u32,
    val_ref: u32,
    ref_val: u32,
}

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct ResultsI32 {
    val_val: i32,
    val_ref: i32,
    ref_val: i32,
}

mod test_u32_u32 {
    use super::*;

    test_runner! {
        name: Runner,
        inputs: {
            LHS: u32 as Uniform<u32>,
            RHS: u32 as Uniform<u32>,
        },
        result: ResultsU32,
        shader: {
            let lhs = *LHS;
            let rhs = *RHS;

            let res1 = lhs >> rhs;
            let res2 = lhs >> &rhs;
            let res3 = &lhs >> rhs;

            unsafe {
                *RESULT.as_mut_unchecked() = ResultsU32 {
                    val_val: res1,
                    val_ref: res2,
                    ref_val: res3,
                };
            }
        },
    }

    pub async fn run() -> Result<(), Box<dyn Error>> {
        let runner = Runner::init().await?;

        let results = runner.run(0b1000, 2).await?;
        assert_eq!(
            results,
            ResultsU32 {
                val_val: 0b0010,
                val_ref: 0b0010,
                ref_val: 0b0010,
            }
        );

        Ok(())
    }
}

mod test_u32_i32 {
    use super::*;

    test_runner! {
        name: Runner,
        inputs: {
            LHS: u32 as Uniform<u32>,
            RHS: i32 as Uniform<i32>,
        },
        result: ResultsU32,
        shader: {
            let lhs = *LHS;
            let rhs = *RHS;

            let res1 = lhs >> rhs;
            let res2 = lhs >> &rhs;
            let res3 = &lhs >> rhs;

            unsafe {
                *RESULT.as_mut_unchecked() = ResultsU32 {
                    val_val: res1,
                    val_ref: res2,
                    ref_val: res3,
                };
            }
        },
    }

    pub async fn run() -> Result<(), Box<dyn Error>> {
        let runner = Runner::init().await?;

        // Standard shift
        assert_eq!(
            runner.run(0b1000, 2).await?,
            ResultsU32 {
                val_val: 0b0010,
                val_ref: 0b0010,
                ref_val: 0b0010,
            }
        );

        // Negative shift (should mask to shift & 31)
        // -1 & 31 = 31
        assert_eq!(
            runner.run(1 << 31, -1).await?,
            ResultsU32 {
                val_val: 1,
                val_ref: 1,
                ref_val: 1,
            }
        );

        // Overflow shift (32 & 31 = 0)
        assert_eq!(
            runner.run(10, 32).await?,
            ResultsU32 {
                val_val: 10,
                val_ref: 10,
                ref_val: 10,
            }
        );

        Ok(())
    }
}

mod test_i32_i32 {
    use super::*;

    test_runner! {
        name: Runner,
        inputs: {
            LHS: i32 as Uniform<i32>,
            RHS: i32 as Uniform<i32>,
        },
        result: ResultsI32,
        shader: {
            let lhs = *LHS;
            let rhs = *RHS;

            let res1 = lhs >> rhs;
            let res2 = lhs >> &rhs;
            let res3 = &lhs >> rhs;

            unsafe {
                *RESULT.as_mut_unchecked() = ResultsI32 {
                    val_val: res1,
                    val_ref: res2,
                    ref_val: res3,
                };
            }
        },
    }

    pub async fn run() -> Result<(), Box<dyn Error>> {
        let runner = Runner::init().await?;

        // Standard shift
        assert_eq!(
            runner.run(0b1000, 2).await?,
            ResultsI32 {
                val_val: 0b0010,
                val_ref: 0b0010,
                ref_val: 0b0010,
            }
        );

        // Arithmetic shift (negative LHS)
        // -10 >> 2
        // -10 is ...11110110
        // -10 >> 2 is ...11111101 (-3)
        assert_eq!(
            runner.run(-10, 2).await?,
            ResultsI32 {
                val_val: -3,
                val_ref: -3,
                ref_val: -3,
            }
        );

        // Negative shift
        // -10 >> -1 -> -10 >> 31
        // Arithmetic shift of negative by 31 is -1.
        assert_eq!(
            runner.run(-10, -1).await?,
            ResultsI32 {
                val_val: -1,
                val_ref: -1,
                ref_val: -1,
            }
        );

        Ok(())
    }
}

async fn run() -> Result<(), Box<dyn Error>> {
    test_u32_u32::run().await?;
    test_u32_i32::run().await?;
    test_i32_i32::run().await?;

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
