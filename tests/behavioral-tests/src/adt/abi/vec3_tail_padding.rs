#![cfg(feature = "test_adt_abi_vec3_tail_padding")]

use std::error::Error;

use behavioral_tests_macros::test_runner;
use empa::abi;
use futures::FutureExt;
use risl::gpu;
use risl::primitive::vec3_f32;

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
struct RustValue {
    vector: vec3_f32,
    scalar: u32,
}

#[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
#[gpu]
#[repr(C)]
struct CValue {
    vector: vec3_f32,
    scalar: u32,
}

mod storage_reads_and_partial_writes {
    use super::*;

    #[derive(Copy, Clone, abi::Sized, PartialEq, Default, Debug)]
    #[gpu]
    #[repr(C)]
    struct Results {
        rust_x: f32,
        rust_y: f32,
        rust_z: f32,
        rust_scalar: u32,
        c_x: f32,
        c_y: f32,
        c_z: f32,
        c_scalar: u32,
    }

    test_runner! {
        name: Runner,
        inputs: {
            RUST_VALUE: RustValue as StorageMut<RustValue>,
            C_VALUE: CValue as StorageMut<CValue>,
            RUST_COMPONENT: f32 as Uniform<f32>,
            C_COMPONENT: f32 as Uniform<f32>,
            RUST_SCALAR: u32 as Uniform<u32>,
            C_SCALAR: u32 as Uniform<u32>,
            MODE: u32 as Storage<u32>,
        },
        result: Results,
        shader: {
            unsafe {
                let rust_value = RUST_VALUE.as_mut_unchecked();
                let c_value = C_VALUE.as_mut_unchecked();

                if *MODE == 1 {
                    rust_value.vector.1 = *RUST_COMPONENT;
                    c_value.vector.1 = *C_COMPONENT;
                } else if *MODE == 2 {
                    rust_value.scalar = *RUST_SCALAR;
                    c_value.scalar = *C_SCALAR;
                }

                *RESULT.as_mut_unchecked() = Results {
                    rust_x: rust_value.vector.0,
                    rust_y: rust_value.vector.1,
                    rust_z: rust_value.vector.2,
                    rust_scalar: rust_value.scalar,
                    c_x: c_value.vector.0,
                    c_y: c_value.vector.1,
                    c_z: c_value.vector.2,
                    c_scalar: c_value.scalar,
                };
            }
        },
    }

    async fn run() -> Result<(), Box<dyn Error>> {
        let runner = Runner::init().await?;
        let rust_value = RustValue {
            vector: vec3_f32(1.0, 2.0, 3.0),
            scalar: 101,
        };
        let c_value = CValue {
            vector: vec3_f32(11.0, 12.0, 13.0),
            scalar: 111,
        };

        assert_eq!(
            runner
                .run(rust_value, c_value, 21.0, 31.0, 121, 131, 0)
                .await?,
            Results {
                rust_x: 1.0,
                rust_y: 2.0,
                rust_z: 3.0,
                rust_scalar: 101,
                c_x: 11.0,
                c_y: 12.0,
                c_z: 13.0,
                c_scalar: 111,
            }
        );
        assert_eq!(
            runner
                .run(rust_value, c_value, 21.0, 31.0, 121, 131, 1)
                .await?,
            Results {
                rust_x: 1.0,
                rust_y: 21.0,
                rust_z: 3.0,
                rust_scalar: 101,
                c_x: 11.0,
                c_y: 31.0,
                c_z: 13.0,
                c_scalar: 111,
            }
        );
        assert_eq!(
            runner
                .run(rust_value, c_value, 21.0, 31.0, 121, 131, 2)
                .await?,
            Results {
                rust_x: 1.0,
                rust_y: 2.0,
                rust_z: 3.0,
                rust_scalar: 121,
                c_x: 11.0,
                c_y: 12.0,
                c_z: 13.0,
                c_scalar: 131,
            }
        );

        Ok(())
    }

    #[test]
    fn test() {
        pollster::block_on(run().map(|res| res.unwrap()));
    }
}

mod rust_whole_value_write {
    use super::*;

    test_runner! {
        name: Runner,
        inputs: {
            X: f32 as Uniform<f32>,
            Y: f32 as Uniform<f32>,
            Z: f32 as Uniform<f32>,
            SCALAR: u32 as Uniform<u32>,
        },
        result: RustValue,
        shader: {
            unsafe {
                *RESULT.as_mut_unchecked() = RustValue {
                    vector: vec3_f32(*X, *Y, *Z),
                    scalar: *SCALAR,
                };
            }
        },
    }

    async fn run() -> Result<(), Box<dyn Error>> {
        let runner = Runner::init().await?;

        assert_eq!(
            runner.run(41.0, 42.0, 43.0, 141).await?,
            RustValue {
                vector: vec3_f32(41.0, 42.0, 43.0),
                scalar: 141,
            }
        );

        Ok(())
    }

    #[test]
    fn test() {
        pollster::block_on(run().map(|res| res.unwrap()));
    }
}

mod c_whole_value_write {
    use super::*;

    test_runner! {
        name: Runner,
        inputs: {
            X: f32 as Uniform<f32>,
            Y: f32 as Uniform<f32>,
            Z: f32 as Uniform<f32>,
            SCALAR: u32 as Uniform<u32>,
        },
        result: CValue,
        shader: {
            unsafe {
                *RESULT.as_mut_unchecked() = CValue {
                    vector: vec3_f32(*X, *Y, *Z),
                    scalar: *SCALAR,
                };
            }
        },
    }

    async fn run() -> Result<(), Box<dyn Error>> {
        let runner = Runner::init().await?;

        assert_eq!(
            runner.run(51.0, 52.0, 53.0, 151).await?,
            CValue {
                vector: vec3_f32(51.0, 52.0, 53.0),
                scalar: 151,
            }
        );

        Ok(())
    }

    #[test]
    fn test() {
        pollster::block_on(run().map(|res| res.unwrap()));
    }
}
