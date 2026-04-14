#![cfg(feature = "test_struct_single_scalar_field")]

// Rustc likes to represent single field structs with the layout of only the field. This test
// verifies that we generate the correct SLIR in such cases.

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Clone, Copy, abi::Sized)]
#[gpu]
struct Value {
    a: u32,
}

#[gpu]
fn increment_value(value: &mut Value) {
    value.a += 1;
}

test_runner! {
    name: TestRunner,
    inputs: {
        VALUE: Value as Storage<Value>,
    },
    result: u32,
    shader: {
        let mut value = *VALUE;

        increment_value(&mut value);

        unsafe {
            *RESULT.as_mut_unchecked() = value.a;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = TestRunner::init().await?;

    assert_eq!(runner.run(Value { a: 0 }).await?, 1u32);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
