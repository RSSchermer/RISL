#![feature(
    stmt_expr_attributes,
    maybe_uninit_array_assume_init,
    maybe_uninit_uninit_array_transpose,
    macro_metavar_expr
)]

use behavioral_tests_macros::gen_test_runner;

mod enum_result;
mod slice_get_index;
mod variable_pointer_if_else;
