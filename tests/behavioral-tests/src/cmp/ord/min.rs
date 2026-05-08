#![cfg(feature = "test_cmp_ord_min")]

use std::error::Error;

use futures::FutureExt;

mod test_u32 {
    use std::error::Error;

    use behavioral_tests_macros::test_runner;
    use empa::abi;

    test_runner! {
        name: Runner,
        inputs: {
            A: u32 as Uniform<u32>,
            B: u32 as Uniform<u32>,
        },
        result: u32,
        shader: {
            let res = u32::min(*A, *B);

            unsafe {
                *RESULT.as_mut_unchecked() = res;
            }
        },
    }

    pub async fn run() -> Result<(), Box<dyn Error>> {
        let runner = Runner::init().await?;

        assert_eq!(runner.run(10, 20).await?, 10);
        assert_eq!(runner.run(20, 10).await?, 10);
        assert_eq!(runner.run(15, 15).await?, 15);

        Ok(())
    }
}

mod test_i32 {
    use std::error::Error;

    use behavioral_tests_macros::test_runner;
    use empa::abi;

    test_runner! {
        name: Runner,
        inputs: {
            A: i32 as Uniform<i32>,
            B: i32 as Uniform<i32>,
        },
        result: i32,
        shader: {
            let res = i32::min(*A, *B);

            unsafe {
                *RESULT.as_mut_unchecked() = res;
            }
        },
    }

    pub async fn run() -> Result<(), Box<dyn Error>> {
        let runner = Runner::init().await?;

        assert_eq!(runner.run(10, 20).await?, 10);
        assert_eq!(runner.run(20, 10).await?, 10);
        assert_eq!(runner.run(-10, -20).await?, -20);
        assert_eq!(runner.run(-20, -10).await?, -20);
        assert_eq!(runner.run(-5, 5).await?, -5);

        Ok(())
    }
}

mod test_usize {
    use std::error::Error;

    use behavioral_tests_macros::test_runner;
    use empa::abi;

    test_runner! {
        name: Runner,
        inputs: {
            A: u32 as Uniform<u32>,
            B: u32 as Uniform<u32>,
        },
        result: u32,
        shader: {
            let res = usize::min(*A as usize, *B as usize);

            unsafe {
                *RESULT.as_mut_unchecked() = res as u32;
            }
        },
    }

    pub async fn run() -> Result<(), Box<dyn Error>> {
        let runner = Runner::init().await?;

        assert_eq!(runner.run(10, 20).await?, 10);
        assert_eq!(runner.run(20, 10).await?, 10);
        assert_eq!(runner.run(15, 15).await?, 15);

        Ok(())
    }
}

mod test_isize {
    use std::error::Error;

    use behavioral_tests_macros::test_runner;
    use empa::abi;

    test_runner! {
        name: Runner,
        inputs: {
            A: i32 as Uniform<i32>,
            B: i32 as Uniform<i32>,
        },
        result: i32,
        shader: {
            let res = isize::min(*A as isize, *B as isize);

            unsafe {
                *RESULT.as_mut_unchecked() = res as i32;
            }
        },
    }

    pub async fn run() -> Result<(), Box<dyn Error>> {
        let runner = Runner::init().await?;

        assert_eq!(runner.run(10, 20).await?, 10);
        assert_eq!(runner.run(20, 10).await?, 10);
        assert_eq!(runner.run(-10, -20).await?, -20);
        assert_eq!(runner.run(-20, -10).await?, -20);
        assert_eq!(runner.run(-5, 5).await?, -5);

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
