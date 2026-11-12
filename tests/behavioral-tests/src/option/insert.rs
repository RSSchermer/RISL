#![cfg(feature = "test_option_insert")]

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

        let inserted = option.insert(*VALUE);
        let returned = *inserted;

        *inserted += 1;

        let stored = match option {
            Some(value) => value,
            None => u32::MAX,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results { returned, stored };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    // Inserting into None returns a reference to the newly stored value.
    assert_eq!(
        runner.run(0, 0, 7).await?,
        Results {
            returned: 7,
            stored: 8
        }
    );
    // Inserting into Some replaces its old value, and the returned reference aliases the option.
    assert_eq!(
        runner.run(1, 42, 3).await?,
        Results {
            returned: 3,
            stored: 4
        }
    );
    // Zero acts as a valid inserted payload, not a None sentinel.
    assert_eq!(
        runner.run(1, 42, 0).await?,
        Results {
            returned: 0,
            stored: 1
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
