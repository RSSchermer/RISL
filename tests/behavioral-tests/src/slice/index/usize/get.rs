#![cfg(feature = "test_slice_index_usize_get")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: Runner,
    inputs: {
        INDEX: u32 as Uniform<u32>,
        VALUES: [u32] as Storage<[u32]>,
    },
    result: u32,
    shader: {
        let values = VALUES.as_ref();
        let index = *INDEX as usize;

        let value = if let Some(value) = values.get(index) {
            *value
        } else {
            99
        };

        unsafe {
            *RESULT.as_mut_unchecked() = value;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run(0, vec![10, 20]).await?, 10);
    assert_eq!(runner.run(1, vec![10, 20]).await?, 20);
    assert_eq!(runner.run(3, vec![10, 20]).await?, 99);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
