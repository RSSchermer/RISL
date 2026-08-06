#![cfg(feature = "test_adt_struct_ref_array_field_dynamic_index_loop")]

// This test was added as the result of a defect in scalar replacement, where array-like switch
// output was not being split correctly. Note that the conditional break inside the loop is
// necessary to create switch-based control-flow.

use std::error::Error;

use behavioral_tests_macros::test_runner;
use futures::FutureExt;
use risl::gpu;

#[gpu]
struct Cursor<'a> {
    values: [&'a u32; 2],
    index: usize,
}

#[gpu]
impl<'a> Cursor<'a> {
    fn next(&mut self) -> Option<u32> {
        if self.index >= 2 {
            return None;
        }

        let value = unsafe { **self.values.get_unchecked(self.index) };

        self.index += 1;

        Some(value)
    }
}

test_runner! {
    name: Runner,
    inputs: {
        START: u32 as Uniform<u32>,
        VALUES: [u32; 2] as Storage<[u32; 2]>,
    },
    result: u32,
    shader: {
        let values = VALUES.as_ref();

        let mut cursor = Cursor {
            values: unsafe { [values.get_unchecked(0), values.get_unchecked(1)] },
            index: *START as usize,
        };

        let mut sum = 0;

        while let Some(value) = cursor.next() {
            if sum + value > 50 {
                break;
            }

            sum += value;
        }

        unsafe {
            *RESULT.as_mut_unchecked() = sum;
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;

    let values = [40, 10];

    assert_eq!(runner.run(0, values).await?, 50);
    assert_eq!(runner.run(1, values).await?, 10);
    assert_eq!(runner.run(2, values).await?, 0);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
