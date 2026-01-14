use crate::intrinsic::{Intrinsic, expect_one_arg};
use crate::ty::{TY_U32, Type, TypeKind};

/// Operation that takes a pointer to an enum and returns its discriminant.
///
/// An enum value's discriminant dynamically represents the enum value's active variant.
pub struct OpGetDiscriminant;

impl Intrinsic for OpGetDiscriminant {
    fn process_args(
        &self,
        ty_registry: &crate::ty::TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        let arg = expect_one_arg!("op-get-discriminant", args);

        let TypeKind::Ptr(pointee_ty) = *ty_registry.kind(arg) else {
            return Err(format!(
                "get-discriminant operation expected a pointer argument, found `{}`",
                arg.to_string(ty_registry)
            ));
        };
        let TypeKind::Enum(_) = &*ty_registry.kind(pointee_ty) else {
            return Err(format!(
                "get-discriminant operation expected a pointer to an `enum` type, found `{}`",
                arg.to_string(ty_registry)
            ));
        };

        Ok(Some(TY_U32))
    }
}
