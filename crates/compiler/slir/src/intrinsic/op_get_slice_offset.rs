use crate::intrinsic::{Intrinsic, expect_one_arg, expect_two_args};
use crate::ty::{TY_U32, Type, TypeKind};

pub struct OpGetSliceOffset;

impl Intrinsic for OpGetSliceOffset {
    fn process_args(
        &self,
        ty_registry: &crate::ty::TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        let arg = expect_one_arg!("op-get-slice-offset", args);

        let TypeKind::Ptr(pointee_ty) = *ty_registry.kind(arg) else {
            return Err(format!(
                "get-slice-offset operation expected the first argument to be a pointer, found \
                `{}`",
                arg.to_string(ty_registry)
            ));
        };

        match &*ty_registry.kind(pointee_ty) {
            TypeKind::Array { .. } | TypeKind::Slice { .. } => {}
            _ => {
                return Err(format!(
                    "get-slice-offset operation expected the first argument to be a pointer to an \
                    array or a slice, found `{}`",
                    arg.to_string(ty_registry)
                ));
            }
        };

        Ok(Some(TY_U32))
    }
}
