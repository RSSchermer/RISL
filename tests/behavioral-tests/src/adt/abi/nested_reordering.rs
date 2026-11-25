#![cfg(feature = "test_adt_abi_nested_reordering")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct Inner {
    before: u32,
    values: [u32; 2],
    after: u32,
}

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[repr(C)]
#[gpu]
struct Outer {
    head: u32,
    inner: Inner,
    tail: u32,
}

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[repr(C)]
#[gpu]
struct Results {
    read_head: u32,
    read_before: u32,
    read_first: u32,
    read_second: u32,
    read_after: u32,
    read_tail: u32,
    written: Outer,
}

test_runner! {
    name: Runner,
    inputs: {
        VALUE: Outer as Storage<Outer>,
        BASE: u32 as Uniform<u32>,
    },
    result: Results,
    shader: {
        let value = VALUE.as_ref();
        let base = *BASE;
        let result = unsafe { RESULT.as_mut_unchecked() };

        // Observe input fields independently of the typed output to avoid cancelling ABI errors.
        *result = Results {
            read_head: value.head,
            read_before: value.inner.before,
            read_first: unsafe { *value.inner.values.get_unchecked(0) },
            read_second: unsafe { *value.inner.values.get_unchecked(1) },
            read_after: value.inner.after,
            read_tail: value.tail,
            written: Outer {
                head: base,
                inner: Inner {
                    before: base + 1,
                    values: [base + 2, base + 3],
                    after: base + 4,
                },
                tail: base + 5,
            },
        };

        result.written.inner.before += 200;
        unsafe { *result.written.inner.values.get_unchecked_mut(1) += 100 };
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    assert_eq!(
        runner
            .run(
                Outer {
                    head: 10,
                    inner: Inner {
                        before: 20,
                        values: [21, 22],
                        after: 23,
                    },
                    tail: 30,
                },
                100,
            )
            .await?,
        Results {
            read_head: 10,
            read_before: 20,
            read_first: 21,
            read_second: 22,
            read_after: 23,
            read_tail: 30,
            written: Outer {
                head: 100,
                inner: Inner {
                    before: 301,
                    values: [102, 203],
                    after: 104,
                },
                tail: 105,
            },
        }
    );

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
