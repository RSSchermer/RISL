use crate::intrinsic::{Intrinsic, expect_one_arg};
use crate::ty::{TY_I32, Type, TypeRegistry};

pub struct OpConvertToI32;

impl Intrinsic for OpConvertToI32 {
    fn process_args(
        &self,
        ty_registry: &TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        let arg = expect_one_arg!("op-convert-to-i32", args);

        if !arg.is_scalar() {
            return Err(format!(
                "convert-to-i32 operations expected a scalar (`u32`, `i32`, `f32`, `bool`), found \
                `{}`",
                arg.to_string(ty_registry)
            ));
        };

        Ok(Some(TY_I32))
    }
}
