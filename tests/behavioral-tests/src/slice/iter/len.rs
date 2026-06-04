#![cfg(feature = "test_slice_iter_len")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    len_0: u32,
    len_1: u32,
    len_2: u32,
    len_3: u32,
    len_4: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        VALUES: [u32; 4] as Storage<[u32; 4]>,
    },
    result: Results,
    shader: {
        let mut iter = VALUES.iter();

        let len_0 = iter.len() as u32;
        iter.next();
        let len_1 = iter.len() as u32;
        iter.next_back();
        let len_2 = iter.len() as u32;
        iter.next();
        let len_3 = iter.len() as u32;
        iter.next_back();
        let len_4 = iter.len() as u32;

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                len_0,
                len_1,
                len_2,
                len_3,
                len_4,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(
        runner.run([1, 2, 3, 4]).await?,
        Results {
            len_0: 4,
            len_1: 3,
            len_2: 2,
            len_3: 1,
            len_4: 0,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
