use serde::{Deserialize, Serialize};

use crate::intrinsic::{Intrinsic, expect_one_arg};
use crate::ty::{TY_BOOL, TY_U32, Type, TypeRegistry};

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpConvertToBool;

impl Intrinsic for OpConvertToBool {
    fn process_args(
        &self,
        ty_registry: &TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        let arg = expect_one_arg!("op-convert-to-bool", args);

        if !arg.is_scalar() {
            return Err(format!(
                "convert-to-bool operations expected a scalar (`u32`, `i32`, `f32`, `bool`), found \
                `{}`",
                arg.to_string(ty_registry)
            ));
        };

        Ok(Some(TY_BOOL))
    }
}
