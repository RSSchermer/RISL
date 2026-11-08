#![cfg(feature = "test_iter_adapters_cycle_size_hint")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

const NONE: u32 = u32::MAX;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    lower_bound: u32,
    upper_bound: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        LEN: u32 as Uniform<u32>,
        FRONT: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let a = unsafe { A.get_unchecked(..(*LEN as usize)) };
        let mut iter = a.iter().cycle();
        let mut i = 0;

        while i < *FRONT {
            iter.next();
            i += 1;
        }

        let (lower_bound, upper_bound) = iter.size_hint();
        let upper_bound = match upper_bound {
            Some(value) => value as u32,
            None => NONE,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                lower_bound: lower_bound as u32,
                upper_bound,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3, 4];

    // An exactly empty source keeps both bounds at zero.
    assert_eq!(
        runner.run(a, 0, 0).await?,
        Results {
            lower_bound: 0,
            upper_bound: 0,
        }
    );
    // A known nonempty source has the GPU usize maximum as its lower bound.
    assert_eq!(
        runner.run(a, 4, 0).await?,
        Results {
            lower_bound: u32::MAX,
            upper_bound: NONE,
        }
    );
    // Exhausting the current iteration still has the GPU usize maximum as its lower bound.
    assert_eq!(
        runner.run(a, 4, 4).await?,
        Results {
            lower_bound: u32::MAX,
            upper_bound: NONE,
        }
    );
    // Wrapping to the original source also still has the GPU usize maximum as its lower bound.
    assert_eq!(
        runner.run(a, 4, 5).await?,
        Results {
            lower_bound: u32::MAX,
            upper_bound: NONE,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
