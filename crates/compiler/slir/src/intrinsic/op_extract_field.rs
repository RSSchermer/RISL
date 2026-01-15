use serde::{Deserialize, Serialize};

use crate::intrinsic::{Intrinsic, expect_one_arg};
use crate::ty::{Type, TypeKind};

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpExtractField {
    pub field_index: u32,
}

impl Intrinsic for OpExtractField {
    fn process_args(
        &self,
        ty_registry: &crate::ty::TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        let arg = expect_one_arg!("op-extract-field", args);

        let TypeKind::Struct(struct_ty) = &*ty_registry.kind(arg) else {
            return Err(format!(
                "extract-field operation expected a `struct` type, found `{}`",
                arg.to_string(ty_registry)
            ));
        };
        let Some(field) = struct_ty.fields.get(self.field_index as usize) else {
            return Err(format!(
                "extract-field operation wants to select field `{}`, but struct type `{}` only has \
                {} field(s)",
                self.field_index,
                arg.to_string(ty_registry),
                struct_ty.fields.len(),
            ));
        };

        Ok(Some(field.ty))
    }

    fn affects_state(&self) -> bool {
        false
    }
}
