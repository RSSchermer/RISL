#![cfg(feature = "test_iter_adapters_map_while_fold")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    value: u32,
    visited: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        STOP: u32 as Uniform<u32>,
        PRECONSUME: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut visited = 0;
        let mut iter = A.iter().copied().map_while(|value| {
            visited = visited * 10 + value;

            if value == *STOP {
                None
            } else {
                Some(value * 2)
            }
        });

        let mut i = 0;

        while i < *PRECONSUME {
            iter.next();
            i += 1;
        }

        let value = iter.fold(8, |acc, value| acc * 10 + value);

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                value,
                visited,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3];

    // Without map termination, fold consumes all doubled values in forward order.
    assert_eq!(
        runner.run(a, 0, 0).await?,
        Results {
            value: 8246,
            visited: 123,
        }
    );
    // Termination on the first value preserves the accumulator; the predicate visits the first
    // value.
    assert_eq!(
        runner.run(a, 1, 0).await?,
        Results {
            value: 8,
            visited: 1,
        }
    );
    // Termination on the second value only add the first value to the accumulator; the predicate
    // visits the first two values.
    assert_eq!(
        runner.run(a, 2, 0).await?,
        Results {
            value: 82,
            visited: 12,
        }
    );
    // Preconsuming an accepted item leaves only the remaining mapped values for fold.
    assert_eq!(
        runner.run(a, 0, 1).await?,
        Results {
            value: 846,
            visited: 123,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
