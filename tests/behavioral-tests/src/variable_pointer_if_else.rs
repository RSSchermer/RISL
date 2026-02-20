use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Clone, Copy, abi::Sized)]
#[gpu]
struct Values {
    a: u32,
    b: u32,
}

test_runner! {
    name: VarPtrIfElseRunner,
    inputs: {
        CONDITION: u32 as Uniform<u32>,
        VALUES: Values as Storage<Values>,
    },
    result: u32,
    shader: {
        let values = VALUES.as_ref();

        let value = if *CONDITION == 1 {
            &values.a
        } else {
            &values.b
        };

        unsafe {
            *RESULT.as_mut_unchecked() = *value;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = VarPtrIfElseRunner::init().await?;

    assert_eq!(runner.run(0u32, Values { a: 0, b: 1 }).await?, 1u32);
    assert_eq!(runner.run(1u32, Values { a: 0, b: 1 }).await?, 0u32);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
