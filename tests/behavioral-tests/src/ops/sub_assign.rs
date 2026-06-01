#![cfg(feature = "test_ops_sub_assign")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    val_val: u32,
    val_ref: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        LHS: u32 as Uniform<u32>,
        RHS: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let lhs = *LHS;
        let rhs = *RHS;

        let mut res1 = lhs;
        res1 -= rhs;

        let mut res2 = lhs;
        res2 -= &rhs;

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                val_val: res1,
                val_ref: res2,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    let results = runner.run(30, 10).await?;
    assert_eq!(
        results,
        Results {
            val_val: 20,
            val_ref: 20,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
