#![cfg(feature = "test_ops_try_op")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Copy, Clone, abi::Sized)]
#[gpu]
struct Inputs {
    a: u32,
    b: u32,
}

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    ok: u32,
    err: u32,
}

#[gpu]
fn try_add_inner(a: u32, b: u32) -> Result<u32, u32> {
    if a > 10 { Err(1000) } else { Ok(a + b) }
}

#[gpu]
fn try_add_outer(a: u32, b: u32) -> Result<u32, u32> {
    let sum = try_add_inner(a, b)?;

    Ok(sum + 100)
}

test_runner! {
    name: Runner,
    inputs: {
        INPUTS: Inputs as Uniform<Inputs>,
    },
    result: Results,
    shader: {
        let res = try_add_outer(INPUTS.a, INPUTS.b);

        unsafe {
            *RESULT.as_mut_unchecked() = match res {
                Ok(v) => Results {
                    ok: v,
                    err: u32::MAX,
                },
                Err(v) => Results {
                    ok: u32::MAX,
                    err: v,
                },
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(
        runner.run(Inputs { a: 1, b: 2 }).await?,
        Results {
            ok: 103,
            err: u32::MAX
        }
    );
    assert_eq!(
        runner.run(Inputs { a: 11, b: 2 }).await?,
        Results {
            ok: u32::MAX,
            err: 1000
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
