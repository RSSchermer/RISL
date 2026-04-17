#![cfg(feature = "test_slice_destructure_first")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

struct Iter<'a, T> {
    slice: &'a [T],
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if let [first, rest @ ..] = self.slice {
            self.slice = rest;

            Some(first)
        } else {
            None
        }
    }
}

test_runner! {
    name: Runner,
    inputs: {
        VALUES: [u32] as Storage<[u32]>,
    },
    result: u32,
    shader: {
        let values = VALUES.as_ref();
        let mut iter = Iter { slice: values };
        let mut sum = 0;

        for v in iter {
            sum += v;
        }

        unsafe {
            *RESULT.as_mut_unchecked() = sum;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run(vec![10, 20, 30, 40]).await?, 100);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
