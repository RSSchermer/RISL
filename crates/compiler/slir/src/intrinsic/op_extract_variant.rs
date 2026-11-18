use serde::{Deserialize, Serialize};

use crate::intrinsic::{Intrinsic, expect_one_arg};
use crate::ty::{Type, TypeKind};

/// Extracts a variant payload from an enum value.
///
/// This is a temporary node used during the [memory_transform], it should not exist in the RVSDG
/// after that.
///
/// [memory_transform]: crate::rvsdg::transform::memory_transform
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpExtractVariant {
    pub variant_index: u32,
}

impl Intrinsic for OpExtractVariant {
    fn process_args(
        &self,
        ty_registry: &crate::ty::TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        let arg = expect_one_arg!("op-extract-variant", args);

        let TypeKind::Enum(enum_ty) = &*ty_registry.kind(arg) else {
            return Err(format!(
                "extract-variant operation expected an `enum` type, found `{}`",
                arg.to_string(ty_registry)
            ));
        };

        let Some(variant) = enum_ty.variants.get(self.variant_index as usize) else {
            return Err(format!(
                "extract-variant operation wants to select variant `{}`, but enum type `{}` only \
                has {} variant(s)",
                self.variant_index,
                arg.to_string(ty_registry),
                enum_ty.variants.len(),
            ));
        };

        Ok(Some(variant.ty))
    }

    fn affects_state(&self) -> bool {
        false
    }
}
