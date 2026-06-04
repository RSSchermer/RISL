#![cfg(feature = "test_iter_adapters_map_next_back")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: Runner,
    inputs: {
        VALUES: [u32] as Storage<[u32]>,
        MULT: u32 as Uniform<u32>,
    },
    result: [u32; 5],
    shader: {
        let mult = *MULT;
        let mut iter = VALUES.iter().map(|v| *v * mult);

        unsafe {
            let res = RESULT.as_mut_unchecked();
            let mut i = 0;

            while let Some(value) = iter.next_back() {
                *res.get_unchecked_mut(i) = value;

                i += 1;
            }
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run(vec![1, 2, 3, 4], 2).await?, [8, 6, 4, 2, 0]);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
