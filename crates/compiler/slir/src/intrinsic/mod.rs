mod op_acos;
mod op_acosh;
mod op_alloca;
mod op_array_length;
mod op_asin;
mod op_asinh;
mod op_atan;
mod op_atanh;
mod op_binary;
mod op_bool_to_branch_selector;
mod op_branch_selector_to_case;
mod op_case_to_branch_selector;
mod op_ceil;
mod op_clamp;
mod op_convert_to_bool;
mod op_convert_to_f32;
mod op_convert_to_i32;
mod op_convert_to_u32;
mod op_cos;
mod op_cosh;
mod op_discriminant_ptr;
mod op_element_ptr;
mod op_exp;
mod op_exp2;
mod op_extract_element;
mod op_extract_field;
mod op_field_ptr;
mod op_floor;
mod op_fract;
mod op_get_discriminant;
mod op_get_slice_offset;
mod op_inverse_sqrt;
mod op_load;
mod op_log;
mod op_log2;
mod op_matrix;
mod op_max;
mod op_min;
mod op_offset_slice;
mod op_round;
mod op_set_discriminant;
mod op_sin;
mod op_sinh;
mod op_sqrt;
mod op_store;
mod op_tan;
mod op_tanh;
mod op_trunc;
mod op_unary;
mod op_variant_ptr;
mod op_vector;

pub use self::op_acos::*;
pub use self::op_acosh::*;
pub use self::op_alloca::*;
pub use self::op_array_length::*;
pub use self::op_asin::*;
pub use self::op_asinh::*;
pub use self::op_atan::*;
pub use self::op_atanh::*;
pub use self::op_binary::*;
pub use self::op_bool_to_branch_selector::*;
pub use self::op_branch_selector_to_case::*;
pub use self::op_case_to_branch_selector::*;
pub use self::op_ceil::*;
pub use self::op_clamp::*;
pub use self::op_convert_to_bool::*;
pub use self::op_convert_to_f32::*;
pub use self::op_convert_to_i32::*;
pub use self::op_convert_to_u32::*;
pub use self::op_cos::*;
pub use self::op_cosh::*;
pub use self::op_discriminant_ptr::*;
pub use self::op_element_ptr::*;
pub use self::op_exp::*;
pub use self::op_exp2::*;
pub use self::op_extract_element::*;
pub use self::op_extract_field::*;
pub use self::op_field_ptr::*;
pub use self::op_floor::*;
pub use self::op_fract::*;
pub use self::op_get_discriminant::*;
pub use self::op_get_slice_offset::*;
pub use self::op_inverse_sqrt::*;
pub use self::op_load::*;
pub use self::op_log::*;
pub use self::op_log2::*;
pub use self::op_matrix::*;
pub use self::op_max::*;
pub use self::op_min::*;
pub use self::op_offset_slice::*;
pub use self::op_round::*;
pub use self::op_set_discriminant::*;
pub use self::op_sin::*;
pub use self::op_sinh::*;
pub use self::op_sqrt::*;
pub use self::op_store::*;
pub use self::op_tan::*;
pub use self::op_tanh::*;
pub use self::op_trunc::*;
pub use self::op_unary::*;
pub use self::op_variant_ptr::*;
pub use self::op_vector::*;
use crate::ty::{Type, TypeRegistry};

macro_rules! expect_one_arg {
    ($intrinsic:literal, $args:expr) => {{
        let mut args = $args.into_iter();

        let Some(arg) = args.next() else {
            return Err(format!(
                "intrinsic `{}` expects exactly one argument, found none",
                $intrinsic
            ));
        };

        if args.next().is_some() {
            return Err(format!(
                "intrinsic `{}` expects exactly one argument, found multiple",
                $intrinsic
            ));
        }

        arg
    }};
}

pub(crate) use expect_one_arg;

macro_rules! expect_two_args {
    ($intrinsic:literal, $args:expr) => {{
        let mut args = $args.into_iter();

        let Some(arg_0) = args.next() else {
            return Err(format!(
                "intrinsic `{}` expects exactly two arguments, found none",
                $intrinsic
            ));
        };

        let Some(arg_1) = args.next() else {
            return Err(format!(
                "intrinsic `{}` expects exactly two arguments, found one",
                $intrinsic
            ));
        };

        if args.next().is_some() {
            return Err(format!(
                "intrinsic `{}` expects exactly two arguments, found more",
                $intrinsic
            ));
        }

        (arg_0, arg_1)
    }};
}

pub(crate) use expect_two_args;

macro_rules! expect_three_args {
    ($intrinsic:literal, $args:expr) => {{
        let mut args = $args.into_iter();

        let Some(arg_0) = args.next() else {
            return Err(format!(
                "intrinsic `{}` expects exactly three arguments, found none",
                $intrinsic
            ));
        };

        let Some(arg_1) = args.next() else {
            return Err(format!(
                "intrinsic `{}` expects exactly three arguments, found one",
                $intrinsic
            ));
        };

        let Some(arg_2) = args.next() else {
            return Err(format!(
                "intrinsic `{}` expects exactly three arguments, found two",
                $intrinsic
            ));
        };

        if args.next().is_some() {
            return Err(format!(
                "intrinsic `{}` expects exactly three arguments, found more",
                $intrinsic
            ));
        }

        (arg_0, arg_1, arg_2)
    }};
}

pub(crate) use expect_three_args;

pub trait Intrinsic {
    fn process_args(
        &self,
        ty_registry: &TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String>;

    fn affects_state(&self) -> bool;
}
