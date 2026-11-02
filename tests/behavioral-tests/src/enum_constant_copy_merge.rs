#![cfg(feature = "test_enum_constant_copy_merge")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;
use risl::gpu;

#[derive(Clone, Copy)]
#[gpu]
enum ThreeVariantEnum {
    First(u32),
    Second(u32),
    Third(u32),
}

const FIRST: ThreeVariantEnum = ThreeVariantEnum::First(11);
const SECOND: ThreeVariantEnum = ThreeVariantEnum::Second(22);

test_runner! {
    name: Runner,
    inputs: {
        SELECTOR: u32 as Uniform<u32>,
    },
    result: u32,
    shader: {
        let selected = if *SELECTOR == 0 { &FIRST } else { &SECOND };
        let copied = *selected;

        let value = match copied {
            ThreeVariantEnum::First(value) => value,
            ThreeVariantEnum::Second(value) => value,
            ThreeVariantEnum::Third(value) => value,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = value;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run(0).await?, 11);
    assert_eq!(runner.run(1).await?, 22);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
