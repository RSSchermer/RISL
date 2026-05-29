#![cfg(feature = "test_slice_iter_count")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    initial: u32,
    after_one: u32,
    after_all: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        VALUES: [u32] as Storage<[u32]>,
    },
    result: Results,
    shader: {
        let initial = VALUES.as_ref().iter().count() as u32;

        let mut iter = VALUES.as_ref().iter();
        iter.next();
        let after_one = iter.count() as u32;

        let mut iter = VALUES.as_ref().iter();
        while let Some(_) = iter.next() {}
        let after_all = iter.count() as u32;

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                initial,
                after_one,
                after_all,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    let values = vec![10, 20, 30, 40];
    let results = runner.run(values).await?;

    assert_eq!(
        results,
        Results {
            initial: 4,
            after_one: 3,
            after_all: 0,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
