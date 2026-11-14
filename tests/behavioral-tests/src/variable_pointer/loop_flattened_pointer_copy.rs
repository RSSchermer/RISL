#![cfg(feature = "test_variable_pointer_loop_flattened_pointer_copy")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;

#[derive(Clone, Copy, abi::Sized)]
#[repr(C)]
#[gpu]
struct Values {
    saved: [u32; 1],
    row: [[u32; 1]; 2],
    matrix: [[[u32; 1]; 2]; 2],
    replacement: [[[u32; 1]; 2]; 2],
}

test_runner! {
    name: Runner,
    inputs: {
        ITERATIONS: u32 as Uniform<u32>,
        ROW: u32 as Uniform<u32>,
        COL: u32 as Uniform<u32>,
        COPY: u32 as Uniform<u32>,
        VALUES: Values as Storage<Values>,
    },
    result: u32,
    shader: {
        let values = VALUES.as_ref();
        let mut saved = &values.saved;
        let mut row = &values.row;
        let mut matrix = &values.matrix;
        let mut i = 0;

        while i < *ITERATIONS {
            let dynamic_row = (i + *ROW) % 2;
            let column = (i + *COL) % 2;

            if *COPY != 0 {
                saved = unsafe {
                    row.get_unchecked(column as usize)
                };
            }
            row = unsafe {
                matrix.get_unchecked(dynamic_row as usize)
            };
            matrix = &values.replacement;

            i += 1;
        }

        unsafe {
            *RESULT.as_mut_unchecked() = *saved.get_unchecked(0);
        }
    },
}

async fn run() -> Result<(), Box<dyn Error>> {
    let runner = Runner::init().await?;
    let values = Values {
        saved: [10],
        row: [[20], [21]],
        matrix: [[[30], [31]], [[40], [41]]],
        replacement: [[[50], [51]], [[60], [61]]],
    };

    // With no iterations, saved retains its initial target.
    assert_eq!(runner.run(0, 0, 0, 1, values).await?, 10);

    // Either column of the initial row can be copied inside the branch.
    assert_eq!(runner.run(1, 0, 0, 1, values).await?, 20);
    assert_eq!(runner.run(1, 1, 1, 1, values).await?, 21);

    // The row index is captured on the preceding iteration, before the column.
    assert_eq!(runner.run(2, 0, 0, 1, values).await?, 31);
    assert_eq!(runner.run(2, 1, 1, 1, values).await?, 40);

    // The replacement reaches saved transitively through matrix and then row.
    assert_eq!(runner.run(3, 0, 0, 1, values).await?, 60);

    // Later row and index updates must not retarget the saved reference.
    assert_eq!(runner.run(6, 1, 0, 1, values).await?, 61);

    // Updating row and matrix without copying must preserve the initial target.
    assert_eq!(runner.run(6, 1, 1, 0, values).await?, 10);

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
