#![cfg(feature = "test_adt_enum_single_inhabited_variant")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[gpu]
enum Never {}

#[gpu]
enum Signed {
    Value = -7,
}

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    first_variant: u32,
    second_variant: u32,
    signed_discriminant: u32,
}

// Keep the match generic so MIR still reads the discriminant before monomorphization.
#[gpu]
fn unwrap_or_zero<E>(value: Result<u32, E>) -> u32 {
    match value {
        Ok(value) => value,
        Err(_) => 0,
    }
}

#[gpu]
fn unwrap_second_or_zero<E>(value: Result<E, u32>) -> u32 {
    match value {
        Ok(_) => 0,
        Err(value) => value,
    }
}

#[gpu]
fn unwrap_signed_or_zero(value: u32, marker: Signed) -> u32 {
    match marker as i32 {
        -7 => value,
        _ => 0,
    }
}

test_runner! {
    name: Runner,
    inputs: {
        VALUE: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        // Result<u32, Never> stores only the u32 payload, with no discriminant tag.
        let first_variant = unwrap_or_zero::<Never>(Ok(*VALUE));
        // The surviving variant has index and discriminant 1, not 0.
        let second_variant = unwrap_second_or_zero::<Never>(Err(*VALUE));
        // The declared discriminant is -7, not the variant index 0.
        let signed_discriminant = unwrap_signed_or_zero(*VALUE, Signed::Value);

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                first_variant,
                second_variant,
                signed_discriminant,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(
        runner.run(7).await?,
        Results {
            first_variant: 7,
            second_variant: 7,
            signed_discriminant: 7,
        }
    );
    assert_eq!(
        runner.run(0).await?,
        Results {
            first_variant: 0,
            second_variant: 0,
            signed_discriminant: 0,
        }
    );
    assert_eq!(
        runner.run(u32::MAX).await?,
        Results {
            first_variant: u32::MAX,
            second_variant: u32::MAX,
            signed_discriminant: u32::MAX,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
