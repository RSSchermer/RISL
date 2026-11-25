#![cfg(feature = "test_adt_abi_over_alignment")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[repr(C, align(32))]
#[gpu]
struct Padded {
    first: u32,
    second: u32,
}

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[repr(C)]
#[gpu]
struct Outer {
    padded: Padded,
    tail: u32,
}

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[repr(C)]
#[gpu]
struct Results {
    read_first: u32,
    read_second: u32,
    read_tail: u32,
    read_array_first: u32,
    read_array_second: u32,
    written: Outer,
    array: [Padded; 2],
}

test_runner! {
    name: Runner,
    inputs: {
        VALUE: Outer as Storage<Outer>,
        VALUES: [Padded; 2] as Storage<[Padded; 2]>,
        INDEX: u32 as Uniform<u32>,
        BASE: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let value = VALUE.as_ref();
        let index = *INDEX as usize;
        let element = unsafe { VALUES.as_ref().get_unchecked(index) };
        let base = *BASE;
        let result = unsafe { RESULT.as_mut_unchecked() };

        *result = Results {
            read_first: value.padded.first,
            read_second: value.padded.second,
            read_tail: value.tail,
            read_array_first: element.first,
            read_array_second: element.second,
            written: Outer {
                padded: Padded {
                    first: base,
                    second: base + 1,
                },
                tail: base + 2,
            },
            array: [
                Padded {
                    first: base + 3,
                    second: base + 4,
                },
                Padded {
                    first: base + 5,
                    second: base + 6,
                },
            ],
        };

        result.written.padded.second += 100;
        unsafe { result.array.get_unchecked_mut(index).first += 200 };
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let value = Outer {
        padded: Padded {
            first: 10,
            second: 11,
        },
        tail: 12,
    };
    let values = [
        Padded {
            first: 20,
            second: 21,
        },
        Padded {
            first: 30,
            second: 31,
        },
    ];

    assert_eq!(
        runner.run(value, values, 0, 100).await?,
        Results {
            read_first: 10,
            read_second: 11,
            read_tail: 12,
            read_array_first: 20,
            read_array_second: 21,
            written: Outer {
                padded: Padded {
                    first: 100,
                    second: 201,
                },
                tail: 102,
            },
            array: [
                Padded {
                    first: 303,
                    second: 104,
                },
                Padded {
                    first: 105,
                    second: 106,
                },
            ],
        }
    );
    assert_eq!(
        runner.run(value, values, 1, 200).await?,
        Results {
            read_first: 10,
            read_second: 11,
            read_tail: 12,
            read_array_first: 30,
            read_array_second: 31,
            written: Outer {
                padded: Padded {
                    first: 200,
                    second: 301,
                },
                tail: 202,
            },
            array: [
                Padded {
                    first: 203,
                    second: 204,
                },
                Padded {
                    first: 405,
                    second: 206,
                },
            ],
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
