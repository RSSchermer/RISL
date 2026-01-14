use serde::{Deserialize, Serialize};

use crate::intrinsic::{Intrinsic, expect_one_arg, expect_two_args};
use crate::ty::{TY_U32, Type, TypeKind};

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpElementPtr;

impl Intrinsic for OpElementPtr {
    fn process_args(
        &self,
        ty_registry: &crate::ty::TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        let (arg_0, arg_1) = expect_two_args!("op-element-ptr", args);

        let TypeKind::Ptr(pointee_ty) = *ty_registry.kind(arg_0) else {
            return Err(format!(
                "element-ptr operation expected the first argument to be a pointer, found `{}`",
                arg_0.to_string(ty_registry)
            ));
        };
        let element_ty = match &*ty_registry.kind(pointee_ty) {
            TypeKind::Array { element_ty, .. } => *element_ty,
            TypeKind::Slice { element_ty, .. } => *element_ty,
            TypeKind::Vector(v) => v.scalar.ty(),
            TypeKind::Matrix(m) => m.column_ty(),
            _ => {
                return Err(format!(
                    "element-ptr operation expected the first argument to be a pointer to an array, a \
                slice, a vector or a matrix; found `{}`",
                    arg_0.to_string(ty_registry)
                ));
            }
        };
        let element_ptr_ty = ty_registry.register(TypeKind::Ptr(element_ty));

        if arg_1 != TY_U32 {
            return Err(format!(
                "element-ptr operation expected the second argument to be a `u32`, found `{}`",
                arg_1.to_string(ty_registry)
            ));
        }

        Ok(Some(element_ptr_ty))
    }
}
