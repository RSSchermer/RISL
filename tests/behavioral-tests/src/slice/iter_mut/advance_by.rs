#![cfg(feature = "test_slice_iter_mut_advance_by")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Results {
    advance_2_ok: u32,
    first_after_advance: u32,
    advance_10_rem: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        VALUES: [u32] as StorageMut<[u32]>,
    },
    result: Results,
    shader: {
        let mut iter = unsafe { VALUES.as_mut_unchecked().iter_mut() };

        let advance_2_ok = if iter.advance_by(2).is_ok() { 1 } else { 0 };
        let first_after_advance = match iter.next() {
            Some(v) => {
                *v += 1;
                *v
            }
            None => u32::MAX,
        };

        let mut iter = unsafe { VALUES.as_mut_unchecked().iter_mut() };
        let advance_10_rem = match iter.advance_by(10) {
            Ok(_) => 0,
            Err(n) => n.get() as u32,
        };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                advance_2_ok,
                first_after_advance,
                advance_10_rem,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    let values = vec![10, 20, 30, 40];
    let results = runner.run(values).await?;

    assert_eq!(
        results,
        Results {
            advance_2_ok: 1,
            first_after_advance: 31,
            advance_10_rem: 6,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
