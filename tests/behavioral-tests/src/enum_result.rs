use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: EnumRunner,
    inputs: {
        VALUE: u32 as StorageMut<u32>,
    },
    result: u32,
    shader: {
        fn test(v: u32) -> Result<u32, u32> {
            if v > 10 { Ok(1) } else { Err(0) }
        }

        unsafe {
            *RESULT.as_mut_unchecked() = match test(*VALUE.as_ref_unchecked()) {
                Ok(v) => v,
                Err(v) => v,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = EnumRunner::init().await?;

    assert_eq!(runner.run(10u32).await?, 0u32);
    assert_eq!(runner.run(11u32).await?, 1u32);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
