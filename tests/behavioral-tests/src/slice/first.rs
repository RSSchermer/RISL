#![cfg(feature = "test_slice_first")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: Runner,
    inputs: {
        START: u32 as Uniform<u32>,
        VALUES: [u32] as Storage<[u32]>,
    },
    result: u32,
    shader: {
        let values = VALUES.as_ref();
        let start = *START as usize;

        let slice = if let Some(slice) = values.get(start..) {
            slice
        } else {
            &[]
        };

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

    assert_eq!(runner.run(0, vec![10, 20, 30]).await?, 10);
    assert_eq!(runner.run(1, vec![10, 20, 30]).await?, 20);
    assert_eq!(runner.run(2, vec![10, 20, 30]).await?, 30);
    assert_eq!(runner.run(3, vec![10, 20, 30]).await?, 99);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
