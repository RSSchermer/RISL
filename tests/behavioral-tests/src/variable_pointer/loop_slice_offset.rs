#![cfg(feature = "test_variable_pointer_loop_slice_offset")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

test_runner! {
    name: Runner,
    inputs: {
        VALUES: [u32] as Storage<[u32]>,
    },
    result: [u32; 4],
    shader: {
        let mut slice = VALUES.as_ref();
        let result = unsafe {
            RESULT.as_mut_unchecked()
        };

        let mut i = 0;

        loop {
            unsafe {
                *result.get_unchecked_mut(i as usize) = *slice.get_unchecked(0);

                slice = slice.get_unchecked(1..);
            }

            i += 1;

            if slice.len() < 2 {
                break;
            }
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run(vec![1, 2, 3, 4]).await?, [1, 2, 3, 0]);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
