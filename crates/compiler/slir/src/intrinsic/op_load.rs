use crate::intrinsic::{Intrinsic, expect_one_arg};
use crate::ty::{Type, TypeKind};

pub struct OpLoad;

impl Intrinsic for OpLoad {
    fn process_args(
        &self,
        ty_registry: &crate::ty::TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        let arg = expect_one_arg!("op-load", args);

        let TypeKind::Ptr(pointee_ty) = *ty_registry.kind(arg) else {
            return Err(format!(
                "load operation expected a pointer argument, found `{}`",
                arg.to_string(ty_registry)
            ));
        };

        Ok(Some(pointee_ty))
    }
}
