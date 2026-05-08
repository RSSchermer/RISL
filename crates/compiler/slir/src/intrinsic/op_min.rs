use serde::{Deserialize, Serialize};

use crate::intrinsic::{Intrinsic, expect_two_args};
use crate::ty::{Type, TypeKind, TypeRegistry};

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpMin;

impl Intrinsic for OpMin {
    fn process_args(
        &self,
        ty_registry: &TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        let (lhs, rhs) = expect_two_args!("min", args);

        if lhs != rhs {
            return Err(format!(
                "intrinsic `min` expects both arguments to have the same type, found `{}` and `{}`",
                lhs.to_string(ty_registry),
                rhs.to_string(ty_registry)
            ));
        }

        match &*ty_registry.kind(lhs) {
            TypeKind::Scalar(kind) if kind.is_numeric() => Ok(Some(lhs)),
            _ => Err(format!(
                "intrinsic `min` expects numeric scalar arguments, found `{}`",
                lhs.to_string(ty_registry)
            )),
        }
    }

    fn affects_state(&self) -> bool {
        false
    }
}
