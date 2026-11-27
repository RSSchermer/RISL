#![cfg(feature = "test_iter_adapters_peekable_next")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

const NONE: u32 = u32::MAX;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    peeked_value: u32,
    values: [u32; 5],
}

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        PRECONSUME: u32 as Uniform<u32>,
        PEEK: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut iter = A.iter().copied().peekable();
        let mut i = 0;

        while i < *PRECONSUME {
            iter.next();
            i += 1;
        }

        let peeked_value = if *PEEK != 0 {
            iter.peek().copied().unwrap_or(NONE)
        } else {
            NONE
        };

        unsafe {
            let result = RESULT.as_mut_unchecked();

            result.peeked_value = peeked_value;

            let mut i = 0;

            while i < 5 {
                *result.values.get_unchecked_mut(i) = match iter.next() {
                    Some(value) => value,
                    None => NONE,
                };

                i += 1;
            }
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3];

    // Without a cached peek, next delegates to the source in forward order.
    assert_eq!(
        runner.run(a, 0, 0).await?,
        Results {
            peeked_value: NONE,
            values: [1, 2, 3, NONE, NONE],
        }
    );
    // The cached front item is returned exactly once, before the remaining source.
    assert_eq!(
        runner.run(a, 0, 1).await?,
        Results {
            peeked_value: 1,
            values: [1, 2, 3, NONE, NONE],
        }
    );
    // Peeking after partial consumption caches the new front item.
    assert_eq!(
        runner.run(a, 1, 1).await?,
        Results {
            peeked_value: 2,
            values: [2, 3, NONE, NONE, NONE],
        }
    );
    // The final item remains available even when peeking exhausts the source.
    assert_eq!(
        runner.run(a, 2, 1).await?,
        Results {
            peeked_value: 3,
            values: [3, NONE, NONE, NONE, NONE],
        }
    );
    // Peeking an exhausted iterator peeks None and leaves the iterator exhausted.
    assert_eq!(
        runner.run(a, 3, 1).await?,
        Results {
            peeked_value: NONE,
            values: [NONE, NONE, NONE, NONE, NONE],
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
