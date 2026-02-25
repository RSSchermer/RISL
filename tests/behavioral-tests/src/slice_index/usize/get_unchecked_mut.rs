use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: Runner,
    inputs: {
        INDEX: u32 as Uniform<u32>,
        VALUES: [u32] as StorageMut<[u32]>,
    },
    result: u32,
    shader: {
        let values = unsafe {
            VALUES.as_mut_unchecked()
        };
        let index = *INDEX as usize;

        unsafe {
            *RESULT.as_mut_unchecked() = *values.get_unchecked_mut(index);
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run(0u32, vec![10u32, 20u32, 30u32]).await?, 10u32);
    assert_eq!(runner.run(1u32, vec![10u32, 20u32, 30u32]).await?, 20u32);
    assert_eq!(runner.run(2u32, vec![10u32, 20u32, 30u32]).await?, 30u32);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
