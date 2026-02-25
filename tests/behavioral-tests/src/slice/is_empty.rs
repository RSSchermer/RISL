use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: Runner,
    inputs: {
        SELECTOR: u32 as Uniform<u32>,
    },
    result: u32,
    shader: {
        let empty = [];
        let not_empty = [1u32];

        let slice: &[u32] = if *SELECTOR == 1 {
            &not_empty
        } else {
            &empty
        };

        unsafe {
            *RESULT.as_mut_unchecked() = slice.is_empty() as u32;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run(0).await?, 1);
    assert_eq!(runner.run(1).await?, 0);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
