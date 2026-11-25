#![cfg(feature = "test_adt_abi_matrix_layout")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;
use risl::primitive::{mat2x3_f32, vec3_f32};

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
#[repr(C)]
struct Value {
    matrix: mat2x3_f32,
    scalar: u32,
}

mod direct_value_write {
    use super::*;

    test_runner! {
        name: Runner,
        inputs: {
            BASE: f32 as Uniform<f32>,
            SCALAR: u32 as Uniform<u32>,
        },
        result: Value,
        shader: {
            let base = *BASE;

            unsafe {
                *RESULT.as_mut_unchecked() = Value {
                    matrix: mat2x3_f32(
                        vec3_f32(base, base + 1.0, base + 2.0),
                        vec3_f32(base + 3.0, base + 4.0, base + 5.0),
                    ),
                    scalar: *SCALAR,
                };
            }
        },
    }

    async fn run() -> Result<(), Box<dyn Error>> {
        let runner = Runner::init().await?;

        assert_eq!(
            runner.run(21.0, 121).await?,
            Value {
                matrix: mat2x3_f32(vec3_f32(21.0, 22.0, 23.0), vec3_f32(24.0, 25.0, 26.0)),
                scalar: 121,
            }
        );

        Ok(())
    }

    #[test]
    fn test() {
        pollster::block_on(run().map(|res| res.unwrap()));
    }
}

mod storage_reads_and_partial_writes {
    use super::*;

    #[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
    #[gpu]
    #[repr(C)]
    struct Results {
        first_col0_x: f32,
        first_col0_y: f32,
        first_col0_z: f32,
        first_col1_x: f32,
        first_col1_y: f32,
        first_col1_z: f32,
        first_scalar: u32,
        second_col0_x: f32,
        second_col0_y: f32,
        second_col0_z: f32,
        second_col1_x: f32,
        second_col1_y: f32,
        second_col1_z: f32,
        second_scalar: u32,
    }

    test_runner! {
        name: Runner,
        inputs: {
            VALUES: [Value; 2] as StorageMut<[Value; 2]>,
            COMPONENT: f32 as Uniform<f32>,
            SCALAR: u32 as Uniform<u32>,
            MODE: u32 as Uniform<u32>,
        },
        result: Results,
        shader: {
            unsafe {
                let values = VALUES.as_mut_unchecked();

                if *MODE == 1 {
                    values.get_unchecked_mut(0).matrix.0.1 = *COMPONENT;
                    values.get_unchecked_mut(1).scalar = *SCALAR;
                } else if *MODE == 2 {
                    values.get_unchecked_mut(1).matrix.1.2 = *COMPONENT;
                    values.get_unchecked_mut(0).scalar = *SCALAR;
                }

                *RESULT.as_mut_unchecked() = Results {
                    first_col0_x: values.get_unchecked(0).matrix.0.0,
                    first_col0_y: values.get_unchecked(0).matrix.0.1,
                    first_col0_z: values.get_unchecked(0).matrix.0.2,
                    first_col1_x: values.get_unchecked(0).matrix.1.0,
                    first_col1_y: values.get_unchecked(0).matrix.1.1,
                    first_col1_z: values.get_unchecked(0).matrix.1.2,
                    first_scalar: values.get_unchecked(0).scalar,
                    second_col0_x: values.get_unchecked(1).matrix.0.0,
                    second_col0_y: values.get_unchecked(1).matrix.0.1,
                    second_col0_z: values.get_unchecked(1).matrix.0.2,
                    second_col1_x: values.get_unchecked(1).matrix.1.0,
                    second_col1_y: values.get_unchecked(1).matrix.1.1,
                    second_col1_z: values.get_unchecked(1).matrix.1.2,
                    second_scalar: values.get_unchecked(1).scalar,
                };
            }
        },
    }

    async fn run() -> Result<(), Box<dyn Error>> {
        let runner = Runner::init().await?;
        let values = [
            Value {
                matrix: mat2x3_f32(vec3_f32(71.0, 72.0, 73.0), vec3_f32(74.0, 75.0, 76.0)),
                scalar: 171,
            },
            Value {
                matrix: mat2x3_f32(vec3_f32(81.0, 82.0, 83.0), vec3_f32(84.0, 85.0, 86.0)),
                scalar: 181,
            },
        ];

        assert_eq!(
            runner.run(values, 91.0, 191, 0).await?,
            Results {
                first_col0_x: 71.0,
                first_col0_y: 72.0,
                first_col0_z: 73.0,
                first_col1_x: 74.0,
                first_col1_y: 75.0,
                first_col1_z: 76.0,
                first_scalar: 171,
                second_col0_x: 81.0,
                second_col0_y: 82.0,
                second_col0_z: 83.0,
                second_col1_x: 84.0,
                second_col1_y: 85.0,
                second_col1_z: 86.0,
                second_scalar: 181,
            }
        );
        assert_eq!(
            runner.run(values, 91.0, 191, 1).await?,
            Results {
                first_col0_x: 71.0,
                first_col0_y: 91.0,
                first_col0_z: 73.0,
                first_col1_x: 74.0,
                first_col1_y: 75.0,
                first_col1_z: 76.0,
                first_scalar: 171,
                second_col0_x: 81.0,
                second_col0_y: 82.0,
                second_col0_z: 83.0,
                second_col1_x: 84.0,
                second_col1_y: 85.0,
                second_col1_z: 86.0,
                second_scalar: 191,
            }
        );
        assert_eq!(
            runner.run(values, 91.0, 191, 2).await?,
            Results {
                first_col0_x: 71.0,
                first_col0_y: 72.0,
                first_col0_z: 73.0,
                first_col1_x: 74.0,
                first_col1_y: 75.0,
                first_col1_z: 76.0,
                first_scalar: 191,
                second_col0_x: 81.0,
                second_col0_y: 82.0,
                second_col0_z: 83.0,
                second_col1_x: 84.0,
                second_col1_y: 85.0,
                second_col1_z: 91.0,
                second_scalar: 181,
            }
        );

        Ok(())
    }

    #[test]
    fn test() {
        pollster::block_on(run().map(|res| res.unwrap()));
    }
}

mod whole_value_write {
    use super::*;

    test_runner! {
        name: Runner,
        inputs: {
            FIRST_BASE: f32 as Uniform<f32>,
            FIRST_SCALAR: u32 as Uniform<u32>,
            SECOND_BASE: f32 as Uniform<f32>,
            SECOND_SCALAR: u32 as Uniform<u32>,
        },
        result: [Value; 2],
        shader: {
            unsafe {
                *RESULT.as_mut_unchecked() = [
                    Value {
                        matrix: mat2x3_f32(
                            vec3_f32(*FIRST_BASE, *FIRST_BASE + 1.0, *FIRST_BASE + 2.0),
                            vec3_f32(*FIRST_BASE + 3.0, *FIRST_BASE + 4.0, *FIRST_BASE + 5.0),
                        ),
                        scalar: *FIRST_SCALAR,
                    },
                    Value {
                        matrix: mat2x3_f32(
                            vec3_f32(*SECOND_BASE, *SECOND_BASE + 1.0, *SECOND_BASE + 2.0),
                            vec3_f32(*SECOND_BASE + 3.0, *SECOND_BASE + 4.0, *SECOND_BASE + 5.0),
                        ),
                        scalar: *SECOND_SCALAR,
                    },
                ];
            }
        },
    }

    async fn run() -> Result<(), Box<dyn Error>> {
        let runner = Runner::init().await?;

        assert_eq!(
            runner.run(201.0, 301, 211.0, 311).await?,
            [
                Value {
                    matrix: mat2x3_f32(
                        vec3_f32(201.0, 202.0, 203.0),
                        vec3_f32(204.0, 205.0, 206.0)
                    ),
                    scalar: 301,
                },
                Value {
                    matrix: mat2x3_f32(
                        vec3_f32(211.0, 212.0, 213.0),
                        vec3_f32(214.0, 215.0, 216.0)
                    ),
                    scalar: 311,
                },
            ]
        );

        Ok(())
    }

    #[test]
    fn test() {
        pollster::block_on(run().map(|res| res.unwrap()));
    }
}
