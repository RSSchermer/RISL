#![cfg(feature = "test_option_get_or_insert_with")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    returned: u32,
    stored: u32,
    calls: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        HAS_VALUE: u32 as Uniform<u32>,
        BEFORE: u32 as Uniform<u32>,
        VALUE: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let mut option = if *HAS_VALUE != 0 {
            Some(*BEFORE)
        } else {
            None
        };

        let mut calls = 0;

        let value = option.get_or_insert_with(|| {
            calls += 1;

            *VALUE
        });

        let returned = *value;

        *value += 1;

        let stored = match option {
            Some(value) => value,
            None => u32::MAX,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results { returned, stored, calls };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    // None invokes the closure once; the returned reference mutates the inserted value.
    assert_eq!(
        runner.run(0, 0, 7).await?,
        Results {
            returned: 7,
            stored: 8,
            calls: 1,
        }
    );
    // Some preserves its value without invoking the closure, and returns a reference to it.
    assert_eq!(
        runner.run(1, 42, 3).await?,
        Results {
            returned: 42,
            stored: 43,
            calls: 0,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
