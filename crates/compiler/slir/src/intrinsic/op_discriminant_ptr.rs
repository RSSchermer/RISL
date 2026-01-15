use serde::{Deserialize, Serialize};

use crate::intrinsic::{Intrinsic, expect_one_arg};
use crate::ty::{TY_PTR_U32, Type, TypeKind};

/// Operation that takes a pointer to an enum and returns a pointer to its discriminant.
///
/// An enum value's discriminant dynamically represents the enum value's active variant.
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpDiscriminantPtr;

impl Intrinsic for OpDiscriminantPtr {
    fn process_args(
        &self,
        ty_registry: &crate::ty::TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        let arg = expect_one_arg!("op-discriminant-ptr", args);

        let TypeKind::Ptr(pointee_ty) = *ty_registry.kind(arg) else {
            return Err(format!(
                "discriminant-ptr operation expected a pointer argument, found `{}`",
                arg.to_string(ty_registry)
            ));
        };
        let TypeKind::Enum(_) = &*ty_registry.kind(pointee_ty) else {
            return Err(format!(
                "discriminant-ptr operation expected a pointer to an `enum` type, found `{}`",
                arg.to_string(ty_registry)
            ));
        };

        Ok(Some(TY_PTR_U32))
    }

    fn affects_state(&self) -> bool {
        false
    }
}
