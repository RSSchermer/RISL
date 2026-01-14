use serde::{Deserialize, Serialize};

use crate::intrinsic::{Intrinsic, expect_one_arg};
use crate::ty::{TY_F32, Type, TypeRegistry};

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpConvertToF32;

impl Intrinsic for OpConvertToF32 {
    fn process_args(
        &self,
        ty_registry: &TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        let arg = expect_one_arg!("op-convert-to-f32", args);

        if !arg.is_numeric_scalar() {
            return Err(format!(
                "convert-to-f32 operations expected a numeric scalar (`u32`, `i32`, `f32`), found \
                `{}`",
                arg.to_string(ty_registry)
            ));
        };

        Ok(Some(TY_F32))
    }
}
