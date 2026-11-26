#![cfg(feature = "test_ops_bit_not")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

mod unsigned {
    use super::*;

    #[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
    #[gpu]
    struct Results {
        val: u32,
        ref_val: u32,
    }

    test_runner! {
        name: Runner,
        inputs: {
            VAL: u32 as Uniform<u32>,
        },
        result: Results,
        shader: {
            let val = *VAL;

            unsafe {
                *RESULT.as_mut_unchecked() = Results {
                    val: !val,
                    ref_val: !&val,
                };
            }
        },
    }

    async fn run() -> Result<(), Box<dyn Error>> {
        let runner = Runner::init().await?;

        assert_eq!(
            runner.run(0b1010).await?,
            Results {
                val: !0b1010,
                ref_val: !0b1010,
            }
        );

        Ok(())
    }

    #[test]
    fn test() {
        pollster::block_on(run().map(|res| res.unwrap()));
    }
}

mod signed {
    use super::*;

    #[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
    #[gpu]
    struct Results {
        val: i32,
        ref_val: i32,
    }

    test_runner! {
        name: Runner,
        inputs: {
            VAL: i32 as Uniform<i32>,
        },
        result: Results,
        shader: {
            let val = *VAL;

            unsafe {
                *RESULT.as_mut_unchecked() = Results {
                    val: !val,
                    ref_val: !&val,
                };
            }
        },
    }

    async fn run() -> Result<(), Box<dyn Error>> {
        let runner = Runner::init().await?;

        assert_eq!(
            runner.run(10).await?,
            Results {
                val: !10,
                ref_val: !10,
            }
        );

        assert_eq!(
            runner.run(-42).await?,
            Results {
                val: !(-42),
                ref_val: !(-42),
            }
        );

        Ok(())
    }

    #[test]
    fn test() {
        pollster::block_on(run().map(|res| res.unwrap()));
    }
}
