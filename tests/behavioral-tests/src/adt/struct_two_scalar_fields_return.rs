#![cfg(feature = "test_adt_struct_two_scalar_fields_return")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;
use risl::gpu;

#[derive(Clone, Copy)]
#[gpu]
struct Pair {
    first: u32,
    second: u32,
}

#[derive(Clone, Copy)]
#[gpu]
struct Wrapper {
    marker: (),
    value: Pair,
}

#[derive(Clone, Copy)]
#[gpu]
struct NestedWrapper {
    inner: Wrapper,
}

#[gpu]
fn swap(value: Pair) -> Pair {
    Pair {
        first: value.second,
        second: value.first,
    }
}

#[gpu]
fn wrap(value: Pair) -> Wrapper {
    Wrapper { marker: (), value }
}

#[gpu]
fn nest(value: Pair) -> NestedWrapper {
    NestedWrapper { inner: wrap(value) }
}

test_runner! {
    name: Runner,
    inputs: {
        FIRST: u32 as Uniform<u32>,
        SECOND: u32 as Uniform<u32>,
    },
    result: [u32; 6],
    shader: {
        let value = swap(Pair { first: *FIRST, second: *SECOND });
        let wrapped = wrap(value);
        let nested = nest(value);

        unsafe {
            *RESULT.as_mut_unchecked() = [
                value.first,
                value.second,
                wrapped.value.first,
                wrapped.value.second,
                nested.inner.value.first,
                nested.inner.value.second,
            ];
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run(11, 37).await?, [37, 11, 37, 11, 37, 11]);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
