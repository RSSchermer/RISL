#![cfg(feature = "test_slice_first_mut")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: Runner,
    inputs: {
        START: u32 as Uniform<u32>,
        VALUES: [u32] as StorageMut<[u32]>,
    },
    result: u32,
    shader: {
        let values = unsafe {
            VALUES.as_mut_unchecked()
        };
        let start = *START as usize;

        let slice = if let Some(slice) = values.get_mut(start..) {
            slice
        } else {
            &mut []
        };

        if let Some(v) = slice.first_mut() {
            *v += 1;
        }

        let first = if let Some(v) = slice.first() {
            *v
        } else {
            99
        };

        unsafe {
            *RESULT.as_mut_unchecked() = first;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run(0, vec![10, 20, 30]).await?, 11);
    assert_eq!(runner.run(1, vec![10, 20, 30]).await?, 21);
    assert_eq!(runner.run(2, vec![10, 20, 30]).await?, 31);
    assert_eq!(runner.run(3, vec![10, 20, 30]).await?, 99);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
