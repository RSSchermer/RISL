#![cfg(feature = "test_adt_enum_copy_reference_loop")]

// Regression for a minimal reproduction of a Fuse adapter test that overflowed the compiler stack
// when carrying a reference to an enum through a loop.

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

const NONE: u32 = u32::MAX;

test_runner! {
    name: Runner,
    inputs: {
        VALUE: u32 as Uniform<u32>,
        INDEX: u32 as Uniform<u32>,
    },
    result: u32,
    shader: {
        let values = [Some(*VALUE), None];

        let mut selected = unsafe { values.get_unchecked(0) };
        let mut i = 0;

        while i < (*INDEX & 1) {
            selected = unsafe { values.get_unchecked(1) };
            i += 1;
        }

        let copied = *selected;

        unsafe {
            *RESULT.as_mut_unchecked() = match copied {
                Some(value) => value,
                None => NONE,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run(7, 0).await?, 7);
    assert_eq!(runner.run(7, 1).await?, NONE);
    assert_eq!(runner.run(0, 0).await?, 0);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
