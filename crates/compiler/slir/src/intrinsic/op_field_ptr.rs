use serde::{Deserialize, Serialize};

use crate::intrinsic::{Intrinsic, expect_one_arg};
use crate::ty::{Type, TypeKind};

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpFieldPtr {
    pub field_index: u32,
}

impl Intrinsic for OpFieldPtr {
    fn process_args(
        &self,
        ty_registry: &crate::ty::TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        let arg = expect_one_arg!("op-field-ptr", args);

        let TypeKind::Ptr(pointee_ty) = *ty_registry.kind(arg) else {
            return Err(format!(
                "field-ptr operation expected a pointer argument, found `{}`",
                arg.to_string(ty_registry)
            ));
        };
        let TypeKind::Struct(struct_ty) = &*ty_registry.kind(pointee_ty) else {
            return Err(format!(
                "field-ptr operation expected a pointer to a `struct` type, found `{}`",
                arg.to_string(ty_registry)
            ));
        };
        let Some(field) = struct_ty.fields.get(self.field_index as usize) else {
            return Err(format!(
                "field-ptr operation wants to select field `{}`, but struct type `{}` only has {} \
                field(s)",
                self.field_index,
                pointee_ty.to_string(ty_registry),
                struct_ty.fields.len(),
            ));
        };
        let field_ptr_ty = ty_registry.register(TypeKind::Ptr(field.ty));

        Ok(Some(field_ptr_ty))
    }
}
