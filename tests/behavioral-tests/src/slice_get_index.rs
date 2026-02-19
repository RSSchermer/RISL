use std::error::Error;

use futures::FutureExt;

use crate::gen_test_runner;

gen_test_runner! {
    name: SliceRunner,
    inputs: {
        INDEX: u32 as Uniform,
        VALUES: [u32] as Storage,
    },
    result: u32,
    shader: {
        let values = VALUES.as_ref();

        let value = if let Some(value) = values.get(*INDEX as usize) {
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
    let runner = SliceRunner::init().await?;

    assert_eq!(runner.run(0u32, vec![10u32, 20u32]).await?, 10u32);
    assert_eq!(runner.run(1u32, vec![10u32, 20u32]).await?, 20u32);
    assert_eq!(runner.run(3u32, vec![10u32, 20u32]).await?, 99u32);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
