#![cfg(feature = "test_iter_adapters_inspect_rfold")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    value: u32,
    inspected: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        PRECONSUME: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut inspected = 0;
        let mut iter = A.iter().copied().inspect(|value| {
            inspected = inspected * 10 + *value;
        });

        let mut i = 0;

        while i < *PRECONSUME {
            iter.next_back();
            i += 1;
        }

        let value = iter.rfold(9, |acc, value| acc * 10 + value);

        unsafe {
            *RESULT.as_mut_unchecked() = Results { value, inspected };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3, 4];

    // The fold accumulator and inspector both record reverse visitation order.
    assert_eq!(
        runner.run(a, 0).await?,
        Results {
            value: 94321,
            inspected: 4321,
        }
    );
    // Backward pre-consumed items are inspected once, and rfold sees only the remaining items.
    assert_eq!(
        runner.run(a, 2).await?,
        Results {
            value: 921,
            inspected: 4321,
        }
    );
    // Folding an exhausted adapter preserves the initial accumulator.
    assert_eq!(
        runner.run(a, 4).await?,
        Results {
            value: 9,
            inspected: 4321,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
