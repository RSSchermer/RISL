#![cfg(feature = "test_unreachable_function_call")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;
use risl::gpu;

#[gpu]
unsafe fn impossible() -> u32 {
    unsafe { std::intrinsics::unreachable() }
}

// Keep both branches in generic MIR; the chosen const argument makes the call unreachable.
#[gpu]
unsafe fn with_dead_call<const BRANCH: u32>(value: u32) -> u32 {
    match BRANCH {
        0 => value,
        _ => unsafe { impossible() },
    }
}

test_runner! {
    name: Runner,
    inputs: {
        VALUE: u32 as Uniform<u32>,
    },
    result: u32,
    shader: {
        unsafe {
            // BRANCH is zero, so the call to impossible can never execute.
            *RESULT.as_mut_unchecked() = with_dead_call::<0>(*VALUE);
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run(7).await?, 7);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
