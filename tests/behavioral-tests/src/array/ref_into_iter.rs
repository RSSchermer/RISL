#![cfg(feature = "test_array_ref_into_iter")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

const NONE: u32 = u32::MAX;

test_runner! {
    name: Runner,
    inputs: {
        VALUES: [u32; 4] as Storage<[u32; 4]>,
    },
    result: [u32; 8],
    shader: {
        let mut iter: std::slice::Iter<'_, u32> =
            <&[u32; 4] as IntoIterator>::into_iter(VALUES.as_ref());
        let empty: [u32; 0] = [];
        let mut empty_iter = <&[u32; 0] as IntoIterator>::into_iter(&empty);

        unsafe {
            *RESULT.as_mut_unchecked() = [
                *iter.next().unwrap_or(&NONE),
                *iter.next_back().unwrap_or(&NONE),
                *iter.next().unwrap_or(&NONE),
                *iter.next_back().unwrap_or(&NONE),
                *iter.next().unwrap_or(&NONE),
                *iter.next_back().unwrap_or(&NONE),
                *empty_iter.next().unwrap_or(&NONE),
                *empty_iter.next_back().unwrap_or(&NONE),
            ];
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    // Both ends share the remaining items; exhausted and empty arrays yield no references.
    assert_eq!(
        runner.run([10, 20, 30, 40]).await?,
        [10, 40, 20, 30, NONE, NONE, NONE, NONE]
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
