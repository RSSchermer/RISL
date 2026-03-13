# Behavioral Tests

This crate contains the behavioral tests for the RISL compiler. These tests verify that the
compiler translates Rust code into programs behave as expected when executed on a GPU.

## Running Tests

The recommended way to run the behavioral tests is by using the custom ["xtask"](https://github.com/matklad/cargo-xtask) 
command provided by this crate:

```sh
cargo xtask run-behavioral-tests
```

As the purpose of this crate is to test the RISL compiler, this command first ensures that a fresh
build of the RISL compiler is available. It then uses this fresh build to run the tests in this crate.

## Selective Testing via Features

The behavioral tests are organized such that each test or group of related tests is behind a Cargo
feature flag. The general guideline is for each shader-module and its associated tests to be behind 
an individual feature flag. 

While `cargo test` comes with its own mechanism for selectively running tests, this is not sufficient
for our use-case: while this will run only a single test, all tests are still compiled. As the object 
of our tests is the compiler itself, we want a mechanism that can ensure only a single shader-module 
is being compiled. This ensures that any debug information output by the compiler process (e.g., an
"RVSDG dump-file") unambiguously pertains to the shader-module under test.

## Running Specific Tests

You can run a specific subset of tests by providing a list of features to the `xtask` command using
the `--features` (or `-f`) flag.

To run a specific test:

```sh
cargo xtask run-behavioral-tests --features test_enum_result
```

To run multiple tests:

```sh
cargo xtask run-behavioral-tests --features test_slice_len,test_slice_is_empty
```

If no feature list is provided, then all features are enabled.

Refer to the `[features]` section of the [Cargo.toml](Cargo.toml) file for a list of all available 
features.
