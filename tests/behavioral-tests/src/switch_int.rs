#![cfg(feature = "test_switch_int")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;

test_runner! {
    name: Runner,
    inputs: {
        SIGNED: i32 as Uniform<i32>,
        UNSIGNED: u32 as Uniform<u32>,
    },
    result: [u32; 6],
    shader: {
        unsafe {
            let res = RESULT.as_mut_unchecked();

            *res.get_unchecked_mut(0) = match *SIGNED {
                0 => 11,
                _ => 12,
            };

            *res.get_unchecked_mut(1) = match *SIGNED {
                -1 => 21,
                _ => 22,
            };

            *res.get_unchecked_mut(2) = match *SIGNED {
                i32::MIN => 31,
                _ => 32,
            };

            *res.get_unchecked_mut(3) = match *SIGNED {
                i32::MAX => 41,
                _ => 42,
            };

            *res.get_unchecked_mut(4) = match *UNSIGNED {
                0 => 51,
                _ => 52,
            };

            *res.get_unchecked_mut(5) = match *UNSIGNED {
                u32::MAX => 61,
                _ => 62,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    for signed in [i32::MIN, -2, -1, 0, 1, i32::MAX] {
        for unsigned in [0, 1, u32::MAX - 1, u32::MAX] {
            let expected = [
                if signed == 0 { 11 } else { 12 },
                if signed == -1 { 21 } else { 22 },
                if signed == i32::MIN { 31 } else { 32 },
                if signed == i32::MAX { 41 } else { 42 },
                if unsigned == 0 { 51 } else { 52 },
                if unsigned == u32::MAX { 61 } else { 62 },
            ];

            assert_eq!(
                runner.run(signed, unsigned).await?,
                expected,
                "signed={signed}, unsigned={unsigned}"
            );
        }
    }

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
