#![cfg(feature = "test_adt_struct_single_slice_field")]

// Rustc likes to represent single field structs with the layout of only the field. This test
// verifies that we generate the correct SLIR in such cases.

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Clone, Copy)]
#[gpu]
struct SliceWrapper<'a> {
    slice: &'a [u32],
}

#[gpu]
fn slice_wrapper_len_by_ref(wrapper: &SliceWrapper) -> u32 {
    wrapper.slice.len() as u32
}

#[gpu]
fn slice_wrapper_len_by_val(wrapper: SliceWrapper) -> u32 {
    wrapper.slice.len() as u32
}

test_runner! {
    name: Runner,
    inputs: {
        VALUES: [u32] as Storage<[u32]>,
    },
    result: u32,
    shader: {
        let wrapper = SliceWrapper { slice: VALUES.as_ref() };
        let mut v = slice_wrapper_len_by_ref(&wrapper);

        v += slice_wrapper_len_by_val(wrapper);

        unsafe {
            *RESULT.as_mut_unchecked() = v;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(runner.run(vec![0, 1, 2, 3]).await?, 8);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
