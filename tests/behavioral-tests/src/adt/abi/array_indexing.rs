#![cfg(feature = "test_adt_abi_array_indexing")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Element {
    before: u32,
    values: [u32; 2],
    after: u32,
}

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[repr(C)]
#[gpu]
struct Results {
    fixed_before: u32,
    fixed_first: u32,
    fixed_second: u32,
    fixed_after: u32,
    fixed_neighbor_before: u32,
    nested_before: u32,
    nested_first: u32,
    nested_second: u32,
    nested_after: u32,
    nested_neighbor_second: u32,
    slice_before: u32,
    slice_first: u32,
    slice_second: u32,
    slice_after: u32,
    slice_neighbor_after: u32,
    slice_len: u32,
}

test_runner! {
    name: Runner,
    inputs: {
        FIXED: [Element; 2] as StorageMut<[Element; 2]>,
        NESTED: [[Element; 2]; 2] as StorageMut<[[Element; 2]; 2]>,
        SLICE: [Element] as StorageMut<[Element]>,
        INDEX: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let index = *INDEX as usize;
        let fixed = unsafe { FIXED.as_mut_unchecked() };
        let nested = unsafe { NESTED.as_mut_unchecked() };
        let slice = unsafe { SLICE.as_mut_unchecked() };

        // Mutate one field per container; observe every field and an untouched neighbor below.
        unsafe {
            fixed.get_unchecked_mut(index).before += 100;
            let nested_value = nested.get_unchecked_mut(index).get_unchecked_mut(1 - index);
            *nested_value.values.get_unchecked_mut(1) += 200;
            slice.get_unchecked_mut(index + 1).after += 300;
        }

        let fixed_value = unsafe { fixed.get_unchecked(index) };
        let nested_value = unsafe { nested.get_unchecked(index).get_unchecked(1 - index) };
        let slice_value = unsafe { slice.get_unchecked(index + 1) };

        unsafe {
            *RESULT.as_mut_unchecked() = Results {
                fixed_before: fixed_value.before,
                fixed_first: *fixed_value.values.get_unchecked(0),
                fixed_second: *fixed_value.values.get_unchecked(1),
                fixed_after: fixed_value.after,
                fixed_neighbor_before: fixed.get_unchecked(1 - index).before,
                nested_before: nested_value.before,
                nested_first: *nested_value.values.get_unchecked(0),
                nested_second: *nested_value.values.get_unchecked(1),
                nested_after: nested_value.after,
                nested_neighbor_second: *nested
                    .get_unchecked(1 - index)
                    .get_unchecked(index)
                    .values
                    .get_unchecked(1),
                slice_before: slice_value.before,
                slice_first: *slice_value.values.get_unchecked(0),
                slice_second: *slice_value.values.get_unchecked(1),
                slice_after: slice_value.after,
                slice_neighbor_after: slice.get_unchecked(0).after,
                slice_len: slice.len() as u32,
            };
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let fixed = [
        Element {
            before: 10,
            values: [11, 12],
            after: 13,
        },
        Element {
            before: 20,
            values: [21, 22],
            after: 23,
        },
    ];
    let nested = [
        [
            Element {
                before: 30,
                values: [31, 32],
                after: 33,
            },
            Element {
                before: 40,
                values: [41, 42],
                after: 43,
            },
        ],
        [
            Element {
                before: 50,
                values: [51, 52],
                after: 53,
            },
            Element {
                before: 60,
                values: [61, 62],
                after: 63,
            },
        ],
    ];
    let slice = [
        Element {
            before: 70,
            values: [71, 72],
            after: 73,
        },
        Element {
            before: 80,
            values: [81, 82],
            after: 83,
        },
        Element {
            before: 90,
            values: [91, 92],
            after: 93,
        },
        Element {
            before: 100,
            values: [101, 102],
            after: 103,
        },
    ];

    assert_eq!(
        runner.run(fixed, nested, &slice[..3], 0).await?,
        Results {
            fixed_before: 110,
            fixed_first: 11,
            fixed_second: 12,
            fixed_after: 13,
            fixed_neighbor_before: 20,
            nested_before: 40,
            nested_first: 41,
            nested_second: 242,
            nested_after: 43,
            nested_neighbor_second: 52,
            slice_before: 80,
            slice_first: 81,
            slice_second: 82,
            slice_after: 383,
            slice_neighbor_after: 73,
            slice_len: 3,
        }
    );
    assert_eq!(
        runner.run(fixed, nested, &slice[..], 1).await?,
        Results {
            fixed_before: 120,
            fixed_first: 21,
            fixed_second: 22,
            fixed_after: 23,
            fixed_neighbor_before: 10,
            nested_before: 50,
            nested_first: 51,
            nested_second: 252,
            nested_after: 53,
            nested_neighbor_second: 42,
            slice_before: 90,
            slice_first: 91,
            slice_second: 92,
            slice_after: 393,
            slice_neighbor_after: 73,
            slice_len: 4,
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
