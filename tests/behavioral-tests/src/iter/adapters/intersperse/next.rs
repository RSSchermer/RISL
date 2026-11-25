#![cfg(feature = "test_iter_adapters_intersperse_next")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

const NONE: u32 = u32::MAX;

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        LEN: u32 as Uniform<u32>,
        PRECONSUME: u32 as Uniform<u32>,
    },
    result: [u32; 7],
    shader: {
        let source = unsafe { A.get_unchecked(..(*LEN as usize)) };
        let mut iter = source.iter().copied().intersperse(9);

        let mut i = 0;

        while i < *PRECONSUME {
            iter.next();
            i += 1;
        }

        unsafe {
            let result = RESULT.as_mut_unchecked();

            let mut i = 0;

            while i < 7 {
                *result.get_unchecked_mut(i) = match iter.next() {
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

    // Separators occur only between source items, followed by repeated None results.
    assert_eq!(runner.run(a, 3, 0).await?, [1, 9, 2, 9, 3, NONE, NONE]);
    // Consuming the first item leaves a separator as the next output.
    assert_eq!(runner.run(a, 3, 1).await?, [9, 2, 9, 3, NONE, NONE, NONE]);
    // Consuming the first separator leaves the second inner item as the next output.
    assert_eq!(
        runner.run(a, 3, 2).await?,
        [2, 9, 3, NONE, NONE, NONE, NONE]
    );
    // An exhausted adapter stays exhausted on subsequent requests.
    assert_eq!(
        runner.run(a, 3, 5).await?,
        [NONE, NONE, NONE, NONE, NONE, NONE, NONE]
    );
    // A singleton has neither a leading nor a trailing separator.
    assert_eq!(
        runner.run(a, 1, 0).await?,
        [1, NONE, NONE, NONE, NONE, NONE, NONE]
    );
    // An empty inner iterator produces no separator values.
    assert_eq!(
        runner.run(a, 0, 0).await?,
        [NONE, NONE, NONE, NONE, NONE, NONE, NONE]
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
