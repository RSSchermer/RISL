#![cfg(feature = "test_adt_enum_tagged_uninhabited_variant")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[gpu]
enum Never {}

#[repr(C)]
#[gpu]
struct ImpossiblePayload {
    before: u32,
    never: Never,
    after: u32,
}

#[repr(i32)]
#[gpu]
enum Tagged<E> {
    Impossible(E) = -7,
    Value(u32) = 3,
    Other(u32) = -11,
}

// Keep the match generic so MIR reads the retained tag before monomorphization.
#[gpu]
fn unwrap_or_zero<E>(value: Tagged<E>) -> u32 {
    match value {
        Tagged::Impossible(_) => 0,
        Tagged::Value(value) => value,
        Tagged::Other(value) => value ^ u32::MAX,
    }
}

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    empty_payload: u32,
    aggregate_payload: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        VALUE: u32 as Uniform<u32>,
        OTHER: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let empty_payload = unwrap_or_zero::<Never>(if *OTHER == 0 {
            Tagged::Value(*VALUE)
        } else {
            Tagged::Other(*VALUE)
        });

        let aggregate_payload = unwrap_or_zero::<ImpossiblePayload>(if *OTHER == 0 {
            Tagged::Value(*VALUE)
        } else {
            Tagged::Other(*VALUE)
        });

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                empty_payload,
                aggregate_payload,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    // OTHER = 0 selects Value (discriminant 3), which returns the payload unchanged.
    assert_eq!(
        runner.run(7, 0).await?,
        Results {
            empty_payload: 7,
            aggregate_payload: 7,
        }
    );

    // OTHER = 1 selects Other (discriminant -11), which XORs the payload with
    // u32::MAX: 7 becomes 0xfffffff8, or 4_294_967_288.
    assert_eq!(
        runner.run(7, 1).await?,
        Results {
            empty_payload: 4_294_967_288,
            aggregate_payload: 4_294_967_288,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
