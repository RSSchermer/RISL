use serde::{Deserialize, Serialize};

use crate::intrinsic::{Intrinsic, expect_one_arg};
use crate::ty::{Type, TypeKind};

/// Operation that takes a pointer to an enum and sets its discriminant according to the
/// `variant_index`.
///
/// *IMPORTANT*: this operation is parameterized by the variant-index, not the discriminant! While
/// the variant-index and the discriminant may be the same, they ofter are not. For example, in
/// Rust users can associate explicit discriminant values with enum variants. The discriminant
/// can be resolved from the variant index using the [Enum::variants](crate::ty::Enum::variants)
/// list, where the variant's position in the list maps to its
/// [EnumVariant::discriminant](crate::ty::EnumVariant::discriminant). Transformations need to be
/// mindful of the discintinction between variant-index and discriminant and ensure they use the
/// appropriate value.
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
