#![cfg(feature = "test_ops_not")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

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
        let val = *VAL != 0;

        let res1 = !val;
        let res2 = !&val;

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                val: res1 as u32,
                ref_val: res2 as u32,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    let results = runner.run(1).await?;
    assert_eq!(results, Results { val: 0, ref_val: 0 });

    let results = runner.run(0).await?;
    assert_eq!(results, Results { val: 1, ref_val: 1 });

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
