#![cfg(feature = "test_adt_struct_single_struct_field")]

// Rustc likes to represent single field structs with the layout of only the field. This test
// verifies that we generate the correct SLIR in such cases.

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Clone, Copy)]
#[gpu]
struct Value {
    a: u32,
    b: u32,
    c: u32,
}

#[derive(Clone, Copy)]
#[gpu]
struct Wrapper {
    inner: Value,
}

#[gpu]
fn increment_value_by_ref(wrapper: &mut Wrapper) {
    wrapper.inner.a += 1;
}

#[gpu]
fn increment_value_by_val(mut wrapper: Wrapper) -> Wrapper {
    wrapper.inner.a += 1;

    wrapper
}

test_runner! {
    name: TestRunner,
    inputs: {
        VALUE: u32 as Storage<u32>,
    },
    result: u32,
    shader: {
        let inner = Value {
            a: *VALUE,
            b: 0,
            c: 0,
        };
        let mut wrapper = Wrapper { inner };

        increment_value_by_ref(&mut wrapper);

        let wrapper = increment_value_by_val(wrapper);

        unsafe {
            *RESULT.as_mut_unchecked() = wrapper.inner.a;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = TestRunner::init().await?;

    assert_eq!(runner.run(0u32).await?, 2u32);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
