use serde::{Deserialize, Serialize};

use crate::intrinsic::{Intrinsic, expect_three_args};
use crate::ty::{ScalarKind, Type, TypeKind, TypeRegistry};

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpClamp;

impl Intrinsic for OpClamp {
    fn process_args(
        &self,
        ty_registry: &TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        let (value, min, max) = expect_three_args!("clamp", args);

        if value != min || value != max {
            return Err(format!(
                "intrinsic `clamp` expects all arguments to have the same type, found `{}`, `{}` \
                and `{}`",
                value.to_string(ty_registry),
                min.to_string(ty_registry),
                max.to_string(ty_registry)
            ));
        }

        match &*ty_registry.kind(value) {
            TypeKind::Scalar(ScalarKind::F32)
            | TypeKind::Scalar(ScalarKind::I32)
            | TypeKind::Scalar(ScalarKind::U32) => Ok(Some(value)),
            _ => Err(format!(
                "intrinsic `clamp` expects arguments to be of type `f32`, `i32` or `u32`, found \
                `{}`",
                value.to_string(ty_registry)
            )),
        }
    }

    fn affects_state(&self) -> bool {
        false
    }
}
