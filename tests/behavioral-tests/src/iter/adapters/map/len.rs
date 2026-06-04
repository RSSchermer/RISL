#![cfg(feature = "test_iter_adapters_map_len")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    len_before: u32,
    len_after: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        VALUES: [u32; 4] as Storage<[u32; 4]>,
    },
    result: Results,
    shader: {
        let mut iter = VALUES.iter().map(|x| *x * 2);

        let len_before = iter.len() as u32;
        iter.next();
        let len_after = iter.len() as u32;

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                len_before,
                len_after,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(
        runner.run([1, 2, 3, 4]).await?,
        Results {
            len_before: 4,
            len_after: 3,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
