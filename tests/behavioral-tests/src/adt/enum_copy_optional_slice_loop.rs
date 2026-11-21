#![cfg(feature = "test_adt_enum_copy_optional_slice_loop")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;
use risl::gpu;

const NONE: u32 = u32::MAX;

#[gpu]
fn copy_next(cursor: &mut Option<&[Option<u32>]>) -> Option<u32> {
    let remaining = cursor.as_mut()?;

    let copied = if let [first, rest @ ..] = *remaining {
        *remaining = rest;
        *first
    } else {
        None
    };

    if copied.is_none() {
        *cursor = None;
    }

    copied
}

test_runner! {
    name: Runner,
    inputs: {
        VALUE: u32 as Uniform<u32>,
        STEPS: u32 as Uniform<u32>,
    },
    result: u32,
    shader: {
        let values = [Some(*VALUE), None];
        let mut cursor = Some(values.as_slice());

        let mut copied = None;
        let mut i = 0;

        while i < *STEPS {
            copied = copy_next(&mut cursor);
            i += 1;
        }

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

    // Copy the first enum's payload.
    assert_eq!(runner.run(7, 1).await?, 7);
    // Copy None from the second element and clear the cursor.
    assert_eq!(runner.run(7, 2).await?, NONE);
    // The next call returns early because the cursor has been cleared.
    assert_eq!(runner.run(7, 3).await?, NONE);
    // A zero payload must remain distinguishable from None.
    assert_eq!(runner.run(0, 1).await?, 0);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
