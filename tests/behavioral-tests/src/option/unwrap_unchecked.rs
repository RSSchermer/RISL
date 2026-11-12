#![cfg(feature = "test_option_unwrap_unchecked")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[gpu]
struct Value {
    a: u32,
    b: u32,
}

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
#[repr(C)]
struct Results {
    scalar: u32,
    shared: u32,
    mutable: u32,
    pair: [u32; 2],
    array_aggregate: [u32; 3],
    struct_aggregate: [u32; 2],
    nested: u32,
    nested_none: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        A: u32 as Uniform<u32>,
        B: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let scalar = unsafe { Some(*A).unwrap_unchecked() };
        let shared = unsafe { Some(&*B).unwrap_unchecked() };
        let mut value = *A;
        let mutable = unsafe { Some(&mut value).unwrap_unchecked() };

        *mutable = *B;

        let pair = unsafe { Some((*A, *B)).unwrap_unchecked() };
        let array_aggregate = unsafe { Some([*A, *B, *A]).unwrap_unchecked() };
        let struct_aggregate = unsafe { Some(Value { a: *A, b: *B }).unwrap_unchecked() };
        let nested = unsafe { Some(Some(*A)).unwrap_unchecked() };
        let nested = match nested {
            Some(value) => value,
            None => u32::MAX,
        };
        let nested_none = unsafe { Some(None::<u32>).unwrap_unchecked() };
        let () = unsafe { Some(()).unwrap_unchecked() };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                scalar,
                shared: *shared,
                mutable: value,
                pair: [pair.0, pair.1],
                array_aggregate,
                struct_aggregate: [struct_aggregate.a, struct_aggregate.b],
                nested,
                nested_none: nested_none.is_none() as u32,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    // Scalar, niche-encoded references, scalar-pair and indirect aggregate payloads.
    assert_eq!(
        runner.run(7, 42).await?,
        Results {
            scalar: 7,
            shared: 42,
            mutable: 42,
            pair: [7, 42],
            array_aggregate: [7, 42, 7],
            struct_aggregate: [7, 42],
            nested: 7,
            nested_none: 1,
        }
    );
    // Zero remains a valid payload, including inside the nested Option.
    assert_eq!(
        runner.run(0, 0).await?,
        Results {
            scalar: 0,
            shared: 0,
            mutable: 0,
            pair: [0, 0],
            array_aggregate: [0, 0, 0],
            struct_aggregate: [0, 0],
            nested: 0,
            nested_none: 1,
        }
    );
    // The full scalar range is preserved, and the mutable reference aliases its source.
    assert_eq!(
        runner.run(u32::MAX, 1).await?,
        Results {
            scalar: 4294967295,
            shared: 1,
            mutable: 1,
            pair: [4294967295, 1],
            array_aggregate: [4294967295, 1, 4294967295],
            struct_aggregate: [4294967295, 1],
            nested: 4294967295,
            nested_none: 1,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
