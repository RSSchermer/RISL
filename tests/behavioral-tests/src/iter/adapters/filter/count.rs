#![cfg(feature = "test_iter_adapters_filter_count")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: Runner,
    inputs: {
        A: [u32] as Storage<[u32]>,
        MIN: u32 as Uniform<u32>,
        PRECONSUME: u32 as Uniform<u32>,
    },
    result: u32,
    shader: {
        let mut iter = A.iter().filter(|value| **value % 2 == 0 && **value >= *MIN);
        let mut i = 0;

        while i < *PRECONSUME {
            iter.next();
            i += 1;
        }

        let result = iter.count() as u32;

        unsafe {
            *RESULT.as_mut_unchecked() = result;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let a: &[u32] = &[1, 2, 3, 4, 5, 6];

    // Count only the three even matches, not all six source items.
    assert_eq!(runner.run(a, 0, 0).await?, 3);
    // The threshold also excludes otherwise matching evens.
    assert_eq!(runner.run(a, 5, 0).await?, 1);
    // Rejecting every item produces a zero count.
    assert_eq!(runner.run(a, 7, 0).await?, 0);
    // A prior next removes one match from the remaining count.
    assert_eq!(runner.run(a, 0, 1).await?, 2);
    // Consuming all matches leaves no items for count.
    assert_eq!(runner.run(a, 0, 3).await?, 0);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}

mod stateful {
    use std::error::Error;

    use behavioral_tests_macros::test_runner;
    use futures::FutureExt;

    test_runner! {
        name: Runner,
        inputs: {
            A: [u32] as Storage<[u32]>,
            PRECONSUME: u32 as Uniform<u32>,
        },
        result: [u32; 2],
        shader: {
            let mut predicate_calls = 0;
            let mut iter = A.iter().filter(|_| {
                predicate_calls += 1;

                predicate_calls % 3 != 0
            });
            let mut i = 0;

            while i < *PRECONSUME {
                iter.next();
                i += 1;
            }

            let count = iter.count() as u32;

            unsafe {
                *RESULT.as_mut_unchecked() = [count, predicate_calls];
            }
        },
    }

    async fn run() -> Result<(), Box<dyn Error>> {
        let runner = Runner::init().await?;
        let a: &[u32] = &[1, 2, 3, 4, 5, 6];

        // Counting calls the stateful predicate once for every source item, including rejections.
        assert_eq!(runner.run(a, 0).await?, [4, 6]);
        // State established by next is retained rather than restarting the predicate.
        assert_eq!(runner.run(a, 2).await?, [2, 6]);
        // An exhausted source makes no additional predicate calls.
        assert_eq!(runner.run(a, 5).await?, [0, 6]);

        Ok(())
    }

    #[test]
    fn test() {
        pollster::block_on(run().map(|res| res.unwrap()));
    }
}

mod small_sources {
    use std::error::Error;

    use behavioral_tests_macros::test_runner;
    use futures::FutureExt;
    use risl::gpu;

    // Cover a custom source with non-capturing, owned-capture, and borrowed-capture predicates.
    #[gpu]
    struct Countdown {
        remaining: u32,
    }

    #[gpu]
    impl Iterator for Countdown {
        type Item = u32;

        fn next(&mut self) -> Option<u32> {
            if self.remaining == 0 {
                None
            } else {
                self.remaining -= 1;

                Some(self.remaining)
            }
        }
    }

    test_runner! {
        name: Runner,
        inputs: {
            LEN: u32 as Uniform<u32>,
            MIN: u32 as Uniform<u32>,
        },
        result: [u32; 4],
        shader: {
            let direct = Countdown { remaining: *LEN }.filter(|value| *value % 2 == 0).count();
            let min = *MIN;
            let pair = Countdown { remaining: *LEN }.filter(move |value| *value >= min).count();
            let empty = std::iter::empty::<u32>().filter(|_| true).count();
            let borrowed = Countdown { remaining: *LEN }.filter(|value| *value >= min).count();

            unsafe {
                *RESULT.as_mut_unchecked() = [direct as u32, pair as u32, empty as u32, borrowed as u32];
            }
        },
    }

    async fn run() -> Result<(), Box<dyn Error>> {
        let runner = Runner::init().await?;

        // Empty custom and standard-library sources count no items.
        assert_eq!(runner.run(0, 0).await?, [0, 0, 0, 0]);
        // Owned and borrowed captures select the same items.
        assert_eq!(runner.run(6, 3).await?, [3, 3, 0, 3]);
        // The captured threshold can reject every item while the even predicate still matches.
        assert_eq!(runner.run(6, 6).await?, [3, 0, 0, 0]);

        Ok(())
    }

    #[test]
    fn test() {
        pollster::block_on(run().map(|res| res.unwrap()));
    }
}
