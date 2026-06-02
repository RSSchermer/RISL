#![cfg(feature = "test_ops_neg")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

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

        let res1 = -val;
        let res2 = -&val;

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                val: res1,
                ref_val: res2,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    let results = runner.run(10).await?;
    assert_eq!(
        results,
        Results {
            val: -10,
            ref_val: -10,
        }
    );

    let results = runner.run(-42).await?;
    assert_eq!(
        results,
        Results {
            val: 42,
            ref_val: 42,
        }
    );

    let results = runner.run(0).await?;
    assert_eq!(results, Results { val: 0, ref_val: 0 });

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
