#![cfg(feature = "test_cmp_ord_clamp")]

use std::error::Error;

use futures::FutureExt;

mod test_u32 {
    use std::error::Error;

    use behavioral_tests_macros::test_runner;
    use empa::abi;
    use risl::intrinsic;

    test_runner! {
        name: Runner,
        inputs: {
            A: u32 as Uniform<u32>,
            MIN: u32 as Uniform<u32>,
            MAX: u32 as Uniform<u32>,
        },
        result: u32,
        shader: {
            let res = unsafe { A.clamp(*MIN, *MAX) };

            unsafe {
                *RESULT.as_mut_unchecked() = res;
            }
        },
    }

    pub async fn run() -> Result<(), Box<dyn Error>> {
        let runner = Runner::init().await?;

        assert_eq!(runner.run(10, 5, 15).await?, 10);
        assert_eq!(runner.run(2, 5, 15).await?, 5);
        assert_eq!(runner.run(20, 5, 15).await?, 15);
        assert_eq!(runner.run(5, 5, 15).await?, 5);
        assert_eq!(runner.run(15, 5, 15).await?, 15);

        Ok(())
    }
}

mod test_i32 {
    use std::error::Error;

    use behavioral_tests_macros::test_runner;
    use empa::abi;
    use risl::intrinsic;

    test_runner! {
        name: Runner,
        inputs: {
            A: i32 as Uniform<i32>,
            MIN: i32 as Uniform<i32>,
            MAX: i32 as Uniform<i32>,
        },
        result: i32,
        shader: {
            let res = unsafe { A.clamp(*MIN, *MAX) };

            unsafe {
                *RESULT.as_mut_unchecked() = res;
            }
        },
    }

    pub async fn run() -> Result<(), Box<dyn Error>> {
        let runner = Runner::init().await?;

        assert_eq!(runner.run(10, 5, 15).await?, 10);
        assert_eq!(runner.run(2, 5, 15).await?, 5);
        assert_eq!(runner.run(20, 5, 15).await?, 15);
        assert_eq!(runner.run(-10, -15, -5).await?, -10);
        assert_eq!(runner.run(-20, -15, -5).await?, -15);
        assert_eq!(runner.run(0, -15, -5).await?, -5);

        Ok(())
    }
}

mod test_usize {
    use std::error::Error;

    use behavioral_tests_macros::test_runner;
    use empa::abi;
    use risl::intrinsic;

    test_runner! {
        name: Runner,
        inputs: {
            A: u32 as Uniform<u32>,
            MIN: u32 as Uniform<u32>,
            MAX: u32 as Uniform<u32>,
        },
        result: u32,
        shader: {
            let res = unsafe { (*A as usize).clamp(*MIN as usize, *MAX as usize) };

            unsafe {
                *RESULT.as_mut_unchecked() = res as u32;
            }
        },
    }

    pub async fn run() -> Result<(), Box<dyn Error>> {
        let runner = Runner::init().await?;

        assert_eq!(runner.run(10, 5, 15).await?, 10);
        assert_eq!(runner.run(2, 5, 15).await?, 5);
        assert_eq!(runner.run(20, 5, 15).await?, 15);

        Ok(())
    }
}

mod test_isize {
    use std::error::Error;

    use behavioral_tests_macros::test_runner;
    use empa::abi;
    use risl::intrinsic;

    test_runner! {
        name: Runner,
        inputs: {
            A: i32 as Uniform<i32>,
            MIN: i32 as Uniform<i32>,
            MAX: i32 as Uniform<i32>,
        },
        result: i32,
        shader: {
            let res = unsafe { (*A as isize).clamp(*MIN as isize, *MAX as isize) };

            unsafe {
                *RESULT.as_mut_unchecked() = res as i32;
            }
        },
    }

    pub async fn run() -> Result<(), Box<dyn Error>> {
        let runner = Runner::init().await?;

        assert_eq!(runner.run(10, 5, 15).await?, 10);
        assert_eq!(runner.run(2, 5, 15).await?, 5);
        assert_eq!(runner.run(20, 5, 15).await?, 15);
        assert_eq!(runner.run(-10, -15, -5).await?, -10);
        assert_eq!(runner.run(-20, -15, -5).await?, -15);
        assert_eq!(runner.run(0, -15, -5).await?, -5);

        Ok(())
    }
}

async fn run() -> Result<(), Box<dyn Error>> {
    test_u32::run().await?;
    test_i32::run().await?;
    test_usize::run().await?;
    test_isize::run().await?;

    Ok(())
}

#[test]
fn test() {
    pollster::block_on(run().map(|res| res.unwrap()));
}
