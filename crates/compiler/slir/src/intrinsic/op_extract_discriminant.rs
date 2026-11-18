use serde::{Deserialize, Serialize};

use crate::intrinsic::{Intrinsic, expect_one_arg};
use crate::ty::{Type, TypeKind};

/// Extracts a enum value's logical discriminant.
///
/// This is a temporary node used during the [memory_transform], it should not exist in the RVSDG
/// after that.
///
/// [memory_transform]: crate::rvsdg::transform::memory_transform
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpExtractDiscriminant;

impl Intrinsic for OpExtractDiscriminant {
    fn process_args(
        &self,
        ty_registry: &crate::ty::TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        let arg = expect_one_arg!("op-extract-discriminant", args);

        let TypeKind::Enum(enum_ty) = &*ty_registry.kind(arg) else {
            return Err(format!(
                "extract-discriminant operation expected an `enum` type, found `{}`",
                arg.to_string(ty_registry)
            ));
        };

        Ok(Some(enum_ty.discriminant_ty.backend_ty()))
    }

    fn affects_state(&self) -> bool {
        false
    }
}
