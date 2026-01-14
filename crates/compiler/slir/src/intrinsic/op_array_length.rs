use serde::{Deserialize, Serialize};

use crate::intrinsic::{Intrinsic, expect_one_arg};
use crate::ty::{TY_U32, Type, TypeKind, TypeRegistry};

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpArrayLength;

impl Intrinsic for OpArrayLength {
    fn process_args(
        &self,
        ty_registry: &TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        let arg = expect_one_arg!("op-array-length", args);

        let TypeKind::Ptr(pointee) = *ty_registry.kind(arg) else {
            return Err(format!(
                "the `array-length` operation expects a pointer argument, found `{}`",
                arg.to_string(ty_registry)
            ));
        };

        if !matches!(
            *ty_registry.kind(pointee),
            TypeKind::Array { .. } | TypeKind::Slice { .. }
        ) {
            return Err(format!(
                "the `array-length` operation expects a pointer to an array or a slice, found `{}`",
                arg.to_string(ty_registry)
            ));
        };

        Ok(Some(TY_U32))
    }
}
