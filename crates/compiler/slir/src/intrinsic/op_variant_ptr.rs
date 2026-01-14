use crate::intrinsic::{Intrinsic, expect_one_arg};
use crate::ty::{Type, TypeKind};

pub struct OpVariantPtr {
    pub variant_index: u32,
}

impl Intrinsic for OpVariantPtr {
    fn process_args(
        &self,
        ty_registry: &crate::ty::TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        let arg = expect_one_arg!("op-variant-ptr", args);

        let TypeKind::Ptr(pointee_ty) = *ty_registry.kind(arg) else {
            return Err(format!(
                "variant-ptr operation expected a pointer argument, found `{}`",
                arg.to_string(ty_registry)
            ));
        };
        let TypeKind::Enum(enum_ty) = &*ty_registry.kind(pointee_ty) else {
            return Err(format!(
                "variant-ptr operation expected a pointer to an `enum` type, found `{}`",
                arg.to_string(ty_registry)
            ));
        };
        let Some(variant_ty) = enum_ty.variants.get(self.variant_index as usize) else {
            return Err(format!(
                "variant-ptr operation wants to select variant `{}`, but enum type `{}` only has \
                {} variant(s)",
                self.variant_index,
                pointee_ty.to_string(ty_registry),
                enum_ty.variants.len(),
            ));
        };
        let variant_ptr_ty = ty_registry.register(TypeKind::Ptr(*variant_ty));

        Ok(Some(variant_ptr_ty))
    }
}
