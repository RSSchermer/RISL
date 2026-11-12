#![cfg(feature = "test_adt_niche_enum_aggregates")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
#[repr(C)]
struct Results {
    pair_marker: u32,
    pair_value: u32,
    reversed_marker: u32,
    reversed_value: u32,
    wrapped_marker: u32,
    wrapped_value: u32,
    scalar_value: u32,
    mutable_marker: u32,
    mutated_value: u32,
    bool_pair_marker: u32,
    bool_pair_value: u32,
    constant_pair_marker: u32,
    constant_pair_value: u32,
}

#[gpu]
struct Pair<'a> {
    marker: bool,
    value: Option<&'a u32>,
}

#[gpu]
struct Wrapper<'a> {
    marker: (),
    pair: Pair<'a>,
}

#[gpu]
struct ScalarWrapper<'a> {
    marker: (),
    value: Option<&'a u32>,
}

#[gpu]
fn make_pair(marker: bool, value: &u32, present: bool) -> (bool, Option<&u32>) {
    (marker, if present { Some(value) } else { None })
}

#[gpu]
fn reverse_pair(pair: (bool, Option<&u32>)) -> (Option<&u32>, bool) {
    (pair.1, !pair.0)
}

#[gpu]
fn wrap_pair(pair: (bool, Option<&u32>)) -> Wrapper<'_> {
    Wrapper {
        marker: (),
        pair: Pair {
            marker: pair.0,
            value: pair.1,
        },
    }
}

#[gpu]
fn wrap_scalar(value: Option<&u32>) -> ScalarWrapper<'_> {
    ScalarWrapper { marker: (), value }
}

#[gpu]
fn mutate_pair(pair: (bool, Option<&mut u32>), replacement: u32) -> bool {
    if let Some(value) = pair.1 {
        *value = replacement;
    }

    pair.0
}

#[gpu]
fn bool_pair(marker: bool, present: bool) -> (bool, Option<bool>) {
    (marker, if present { Some(!marker) } else { None })
}

#[gpu]
fn constant_pair(present: bool) -> (bool, Option<bool>) {
    if present {
        (false, Some(true))
    } else {
        (true, None)
    }
}

test_runner! {
    name: Runner,
    inputs: {
        A: u32 as Uniform<u32>,
        B: u32 as Uniform<u32>,
        MARKER: u32 as Uniform<u32>,
        PRESENT: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let pair = make_pair(*MARKER != 0, &*A, *PRESENT != 0);
        let reversed = reverse_pair(pair);
        let wrapped = wrap_pair(pair);
        let scalar = wrap_scalar(pair.1);
        let mut value = *A;
        let mutable = if *PRESENT != 0 { Some(&mut value) } else { None };
        let marker = mutate_pair((pair.0, mutable), *B);
        let bools = bool_pair(pair.0, *PRESENT != 0);
        let constant = constant_pair(*PRESENT != 0);

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                pair_marker: pair.0 as u32,
                pair_value: match pair.1 {
                    Some(value) => *value,
                    None => u32::MAX,
                },
                reversed_marker: reversed.1 as u32,
                reversed_value: match reversed.0 {
                    Some(value) => *value,
                    None => u32::MAX,
                },
                wrapped_marker: wrapped.pair.marker as u32,
                wrapped_value: match wrapped.pair.value {
                    Some(value) => *value,
                    None => u32::MAX,
                },
                scalar_value: match scalar.value {
                    Some(value) => *value,
                    None => u32::MAX,
                },
                mutable_marker: marker as u32,
                mutated_value: value,
                bool_pair_marker: bools.0 as u32,
                bool_pair_value: match bools.1 {
                    Some(value) => value as u32,
                    None => u32::MAX,
                },
                constant_pair_marker: constant.0 as u32,
                constant_pair_value: match constant.1 {
                    Some(value) => value as u32,
                    None => u32::MAX,
                },
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    // Both marker values remain independent of the niche-encoded Some payload.
    assert_eq!(
        runner.run(7, 42, 0, 1).await?,
        Results {
            pair_marker: 0,
            pair_value: 7,
            reversed_marker: 1,
            reversed_value: 7,
            wrapped_marker: 0,
            wrapped_value: 7,
            scalar_value: 7,
            mutable_marker: 0,
            mutated_value: 42,
            bool_pair_marker: 0,
            bool_pair_value: 1,
            constant_pair_marker: 0,
            constant_pair_value: 1,
        }
    );
    assert_eq!(
        runner.run(7, 42, 1, 1).await?,
        Results {
            pair_marker: 1,
            pair_value: 7,
            reversed_marker: 0,
            reversed_value: 7,
            wrapped_marker: 1,
            wrapped_value: 7,
            scalar_value: 7,
            mutable_marker: 1,
            mutated_value: 42,
            bool_pair_marker: 1,
            bool_pair_value: 0,
            constant_pair_marker: 0,
            constant_pair_value: 1,
        }
    );
    // `None` must survive passing, returning and wrapping without dereferencing a payload.
    assert_eq!(
        runner.run(7, 42, 0, 0).await?,
        Results {
            pair_marker: 0,
            pair_value: u32::MAX,
            reversed_marker: 1,
            reversed_value: u32::MAX,
            wrapped_marker: 0,
            wrapped_value: u32::MAX,
            scalar_value: u32::MAX,
            mutable_marker: 0,
            mutated_value: 7,
            bool_pair_marker: 0,
            bool_pair_value: u32::MAX,
            constant_pair_marker: 1,
            constant_pair_value: u32::MAX,
        }
    );
    assert_eq!(
        runner.run(7, 42, 1, 0).await?,
        Results {
            pair_marker: 1,
            pair_value: u32::MAX,
            reversed_marker: 0,
            reversed_value: u32::MAX,
            wrapped_marker: 1,
            wrapped_value: u32::MAX,
            scalar_value: u32::MAX,
            mutable_marker: 1,
            mutated_value: 7,
            bool_pair_marker: 1,
            bool_pair_value: u32::MAX,
            constant_pair_marker: 1,
            constant_pair_value: u32::MAX,
        }
    );
    // A reference to zero is still `Some`, and mutable payloads alias the original value.
    assert_eq!(
        runner.run(0, 9, 1, 1).await?,
        Results {
            pair_marker: 1,
            pair_value: 0,
            reversed_marker: 0,
            reversed_value: 0,
            wrapped_marker: 1,
            wrapped_value: 0,
            scalar_value: 0,
            mutable_marker: 1,
            mutated_value: 9,
            bool_pair_marker: 1,
            bool_pair_value: 0,
            constant_pair_marker: 0,
            constant_pair_value: 1,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
