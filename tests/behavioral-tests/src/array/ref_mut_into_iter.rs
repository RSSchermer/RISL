#![cfg(feature = "test_array_ref_mut_into_iter")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

const NONE: u32 = u32::MAX;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    visited: [u32; 8],
    values: [u32; 4],
}

test_runner! {
    name: Runner,
    inputs: {
        VALUES: [u32; 4] as Storage<[u32; 4]>,
    },
    result: Results,
    shader: {
        let mut values = *VALUES.as_ref();
        let mut iter: std::slice::IterMut<'_, u32> =
            <&mut [u32; 4] as IntoIterator>::into_iter(&mut values);
        let mut empty: [u32; 0] = [];
        let mut empty_iter = <&mut [u32; 0] as IntoIterator>::into_iter(&mut empty);

        let visited = [
            iter.next().map(|value| {
                *value += 1;

                *value
            }).unwrap_or(NONE),
            iter.next_back().map(|value| {
                *value += 4;

                *value
            }).unwrap_or(NONE),
            iter.next().map(|value| {
                *value += 2;

                *value
            }).unwrap_or(NONE),
            iter.next_back().map(|value| {
                *value += 3;

                *value
            }).unwrap_or(NONE),
            iter.next().map(|value| *value).unwrap_or(NONE),
            iter.next_back().map(|value| *value).unwrap_or(NONE),
            empty_iter.next().map(|value| *value).unwrap_or(NONE),
            empty_iter.next_back().map(|value| *value).unwrap_or(NONE),
        ];

        unsafe {
            *RESULT.as_mut_unchecked() = Results { visited, values };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    // Both ends yield mutable references into the array; exhausted and empty arrays yield none.
    assert_eq!(
        runner.run([10, 20, 30, 40]).await?,
        Results {
            visited: [11, 44, 22, 33, NONE, NONE, NONE, NONE],
            values: [11, 22, 33, 44],
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
