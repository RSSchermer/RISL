use serde::{Deserialize, Serialize};

use crate::intrinsic::{Intrinsic, expect_two_args};
use crate::ty::{Type, TypeKind};

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpStore;

impl Intrinsic for OpStore {
    fn process_args(
        &self,
        ty_registry: &crate::ty::TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        let (arg_0, arg_1) = expect_two_args!("op-store", args);

        let TypeKind::Ptr(pointee_ty) = *ty_registry.kind(arg_0) else {
            return Err(format!(
                "store operation the first argument to be a pointer, found `{}`",
                arg_0.to_string(ty_registry)
            ));
        };

        if arg_1 != pointee_ty {
            return Err(format!(
                "store operation expected the second argument to be of type `{}`, but found type \
                `{}`",
                pointee_ty.to_string(ty_registry),
                arg_1.to_string(ty_registry),
            ));
        }

        Ok(None)
    }

    fn affects_state(&self) -> bool {
        true
    }
}
