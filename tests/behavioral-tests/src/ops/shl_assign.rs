#![cfg(feature = "test_ops_shl_assign")]

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

            let mut res1 = lhs;
            res1 <<= rhs;

            let mut res2 = lhs;
            res2 <<= &rhs;

            unsafe {
                *RESULT.as_mut_unchecked() = ResultsU32 {
                    val_val: res1,
                    val_ref: res2,
                };
            }
        },
    }

    pub async fn run() -> Result<(), Box<dyn Error>> {
        let runner = Runner::init().await?;

        let results = runner.run(0b0001, 2).await?;
        assert_eq!(
            results,
            ResultsU32 {
                val_val: 0b0100,
                val_ref: 0b0100,
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

            let mut res1 = lhs;
            res1 <<= rhs;

            let mut res2 = lhs;
            res2 <<= &rhs;

            unsafe {
                *RESULT.as_mut_unchecked() = ResultsU32 {
                    val_val: res1,
                    val_ref: res2,
                };
            }
        },
    }

    pub async fn run() -> Result<(), Box<dyn Error>> {
        let runner = Runner::init().await?;

        // Standard shift
        assert_eq!(
            runner.run(0b0001, 2).await?,
            ResultsU32 {
                val_val: 0b0100,
                val_ref: 0b0100,
            }
        );

        // Negative shift
        assert_eq!(
            runner.run(1, -1).await?,
            ResultsU32 {
                val_val: 1 << 31,
                val_ref: 1 << 31,
            }
        );

        // Overflow shift
        assert_eq!(
            runner.run(10, 32).await?,
            ResultsU32 {
                val_val: 10,
                val_ref: 10,
            }
        );

        Ok(())
    }
}

async fn run() -> Result<(), Box<dyn Error>> {
    test_u32_u32::run().await?;
    test_u32_i32::run().await?;

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
