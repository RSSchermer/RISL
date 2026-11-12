#![feature(
    stmt_expr_attributes,
    maybe_uninit_array_assume_init,
    maybe_uninit_uninit_array_transpose,
    macro_metavar_expr,
    iter_advance_by,
    exact_size_is_empty
)]

mod adt;
mod array;
mod closure;
mod cmp;
mod condition_partial_initialization;
mod enum_constant_copy_merge;
mod enum_result;
mod iter;
mod nested_mergable_control_flow;
mod num;
mod ops;
mod option;
mod slice;
mod switch_int;
mod variable_pointer;
mod while_loop;
