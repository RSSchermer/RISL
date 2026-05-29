#![cfg(feature = "test_slice_last")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

test_runner! {
    name: Runner,
    inputs: {
        END: u32 as Uniform<u32>,
        VALUES: [u32] as Storage<[u32]>,
    },
    result: u32,
    shader: {
        let values = VALUES.as_ref();
        let end = *END as usize;

        let slice = if let Some(slice) = values.get(..end) {
            slice
        } else {
            &[]
        };

        let result = if let Some(last) = slice.last() {
            *last
        } else {
            u32::MAX
        };

        unsafe {
            *RESULT.as_mut_unchecked() = result;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let values = vec![10, 20, 30, 40];

    // Case 1: Sub-slice [] -> last: u32::MAX
    assert_eq!(runner.run(0, values.clone()).await?, u32::MAX);

    // Case 2: Sub-slice [10] -> last: 10
    assert_eq!(runner.run(1, values.clone()).await?, 10);

    // Case 3: Sub-slice [10, 20, 30] -> last: 30
    assert_eq!(runner.run(3, values.clone()).await?, 30);

    // Case 4: Sub-slice [10, 20, 30, 40] -> last: 40
    assert_eq!(runner.run(4, values.clone()).await?, 40);

    // Case 5: Out of bounds -> last: u32::MAX
    assert_eq!(runner.run(5, values.clone()).await?, u32::MAX);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
