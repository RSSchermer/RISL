use serde::{Deserialize, Serialize};

use crate::intrinsic::{Intrinsic, expect_one_arg};
use crate::ty::{Type, TypeKind};

/// Operation that takes a pointer to an enum and returns its discriminant.
///
/// An enum value's discriminant dynamically represents the enum value's active variant.
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpSetDiscriminant {
    pub variant_index: u32,
}

impl Intrinsic for OpSetDiscriminant {
    fn process_args(
        &self,
        ty_registry: &crate::ty::TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        let arg = expect_one_arg!("op-set-discriminant", args);

        let TypeKind::Ptr(pointee_ty) = *ty_registry.kind(arg) else {
            return Err(format!(
                "set-discriminant operation expects its argument to be a pointer to an enum, found \
                `{}`",
                arg.to_string(ty_registry)
            ));
        };
        let TypeKind::Enum(_) = &*ty_registry.kind(pointee_ty) else {
            return Err(format!(
                "set-discriminant operation expects its argument to be a pointer to an enum, found \
                `{}`",
                arg.to_string(ty_registry)
            ));
        };

        Ok(None)
    }

    fn affects_state(&self) -> bool {
        true
    }
}
