#![cfg(feature = "test_iter_adapters_cycle_next")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

const NONE: u32 = u32::MAX;

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        LEN: u32 as Uniform<u32>,
        SKIP_SOURCE: u32 as Uniform<u32>,
    },
    result: [u32; 10],
    shader: {
        let source = unsafe { A.get_unchecked(..(*LEN as usize)) };
        let mut source = source.iter();
        let mut i = 0;

        while i < *SKIP_SOURCE {
            source.next();
            i += 1;
        }

        let mut iter = source.cycle();

        unsafe {
            let res = RESULT.as_mut_unchecked();

            let mut i = 0;

            while i < 10 {
                *res.get_unchecked_mut(i) = match iter.next() {
                    Some(value) => *value,
                    None => NONE,
                };

                i += 1;
            }
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3, 4];

    // Repeat in order through two resets without dropping or duplicating boundary items.
    assert_eq!(runner.run(a, 4, 0).await?, [1, 2, 3, 4, 1, 2, 3, 4, 1, 2]);
    // Reset to the source state captured at construction, not its original beginning.
    assert_eq!(runner.run(a, 4, 2).await?, [3, 4, 3, 4, 3, 4, 3, 4, 3, 4]);
    // A singleton restarts on every subsequent call.
    assert_eq!(runner.run(a, 1, 0).await?, [1; 10]);
    // An empty source keeps returning None after repeated resets.
    assert_eq!(runner.run(a, 0, 0).await?, [NONE; 10]);
    // A source exhausted before construction must not resurrect consumed items.
    assert_eq!(runner.run(a, 4, 4).await?, [NONE; 10]);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
