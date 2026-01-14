mod op_alloca;
mod op_array_length;
mod op_binary;
mod op_bool_to_branch_selector;
mod op_branch_selector_to_case;
mod op_case_to_branch_selector;
mod op_convert_to_bool;
mod op_convert_to_f32;
mod op_convert_to_i32;
mod op_convert_to_u32;
mod op_discriminant_ptr;
mod op_element_ptr;
mod op_extract_element;
mod op_extract_field;
mod op_field_ptr;
mod op_get_discriminant;
mod op_get_slice_offset;
mod op_load;
mod op_matrix;
mod op_offset_slice;
mod op_set_discriminant;
mod op_store;
mod op_u32_to_branch_selector;
mod op_unary;
mod op_variant_ptr;
mod op_vector;

pub use self::op_alloca::*;
pub use self::op_array_length::*;
pub use self::op_binary::*;
pub use self::op_bool_to_branch_selector::*;
pub use self::op_case_to_branch_selector::*;
pub use self::op_discriminant_ptr::*;
pub use self::op_element_ptr::*;
pub use self::op_extract_element::*;
pub use self::op_extract_field::*;
pub use self::op_field_ptr::*;
pub use self::op_get_discriminant::*;
pub use self::op_load::*;
pub use self::op_set_discriminant::*;
pub use self::op_store::*;
pub use self::op_unary::*;
pub use self::op_variant_ptr::*;
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

pub trait Intrinsic {
    fn process_args(
        &self,
        ty_registry: &TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String>;
}
