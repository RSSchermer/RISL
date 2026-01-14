use serde::{Deserialize, Serialize};

use crate::intrinsic::{Intrinsic, expect_one_arg, expect_two_args};
use crate::ty::{TY_U32, Type, TypeKind};

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpOffsetSlice;

impl Intrinsic for OpOffsetSlice {
    fn process_args(
        &self,
        ty_registry: &crate::ty::TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        let (arg_0, arg_1) = expect_two_args!("op-offset-slice", args);

        let TypeKind::Ptr(pointee_ty) = *ty_registry.kind(arg_0) else {
            return Err(format!(
                "offset-slice operation expected the first argument to be a pointer, found `{}`",
                arg_0.to_string(ty_registry)
            ));
        };
        let (element_ty, stride) = match &*ty_registry.kind(pointee_ty) {
            TypeKind::Array {
                element_ty, stride, ..
            }
            | TypeKind::Slice {
                element_ty, stride, ..
            } => (*element_ty, *stride),
            _ => {
                return Err(format!(
                    "offset-slice operation expected the first argument to be a pointer to an array or \
                a slice, found `{}`",
                    arg_0.to_string(ty_registry)
                ));
            }
        };
        let slice_ty = ty_registry.register(TypeKind::Slice { element_ty, stride });
        let slice_ptr_ty = ty_registry.register(TypeKind::Ptr(slice_ty));

        if arg_1 != TY_U32 {
            return Err(format!(
                "offset-slice operation expected the second argument to be a `u32`, found `{}`",
                arg_1.to_string(ty_registry)
            ));
        }

        Ok(Some(slice_ptr_ty))
    }
}
