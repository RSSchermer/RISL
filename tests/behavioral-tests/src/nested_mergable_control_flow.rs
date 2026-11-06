#![cfg(feature = "test_nested_mergable_control_flow")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

// This test checks for a historically failing case, where the compiler tried to merge two switch
// nodes (`encoded` and `selected` below), but a permutation of the branch order was required; this
// permutation was not properly applied if the later switch node was nested inside another switch
// (`nested` below).

test_runner! {
    name: Runner,
    inputs: {
        VALUE: u32 as Uniform<u32>,
        GATE: u32 as Uniform<u32>,
    },
    result: u32,
    shader: {
        let a = 111;
        let b = 222;
        let c = 333;

        let encoded = match *VALUE {
            0 => 1u32,
            1 => 0u32,
            _ => 2u32,
        };

        let selected = match encoded {
            0 => &a,
            1 => &b,
            _ => &c,
        };

        let nested = if *GATE == 0 { selected } else { &c };

        unsafe {
            *RESULT.as_mut_unchecked() = *nested;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let mut actual = Vec::new();
    let mut expected = Vec::new();

    for (value, selected) in [(0, 222), (1, 111), (2, 333), (u32::MAX, 333)] {
        for gate in [0, 1] {
            actual.push((value, gate, runner.run(value, gate).await?));
            expected.push((value, gate, if gate == 0 { selected } else { 333 }));
        }
    }

    assert_eq!(actual, expected, "(value, gate, result)");

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
