use serde::{Deserialize, Serialize};

use crate::intrinsic::{Intrinsic, expect_two_args};
use crate::ty::{TY_U32, Type, TypeKind};

/// Operation that takes a pointer to an enum and returns its discriminant.
///
/// An enum value's discriminant dynamically represents the enum value's active variant.
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpSetDiscriminant;

impl Intrinsic for OpSetDiscriminant {
    fn process_args(
        &self,
        ty_registry: &crate::ty::TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        let (arg_0, arg_1) = expect_two_args!("op-set-discriminant", args);

        let TypeKind::Ptr(pointee_ty) = *ty_registry.kind(arg_0) else {
            return Err(format!(
                "set-discriminant operation expects its first argument to be a pointer to an enum, \
                found `{}`",
                arg_0.to_string(ty_registry)
            ));
        };
        let TypeKind::Enum(_) = &*ty_registry.kind(pointee_ty) else {
            return Err(format!(
                "set-discriminant operation expects its first argument to be a pointer to an enum, \
                found `{}`",
                arg_0.to_string(ty_registry)
            ));
        };
        if arg_1 != TY_U32 {
            return Err(format!(
                "set-discriminant operation expects its second argument to be a `u32`, found `{}`",
                arg_1.to_string(ty_registry)
            ));
        }

        Ok(None)
    }
}
