use serde::{Deserialize, Serialize};

use crate::intrinsic::{Intrinsic, expect_three_args};
use crate::ty::{ScalarKind, Type, TypeKind, TypeRegistry};

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpFusedMulAdd;

impl Intrinsic for OpFusedMulAdd {
    fn process_args(
        &self,
        ty_registry: &TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        let (a, b, c) = expect_three_args!("fused_mul_add", args);

        if a != b || a != c {
            return Err(format!(
                "intrinsic `fused_mul_add` expects all arguments to have the same type, found \
                `{}`, `{}` and `{}`",
                a.to_string(ty_registry),
                b.to_string(ty_registry),
                c.to_string(ty_registry)
            ));
        }

        match &*ty_registry.kind(a) {
            TypeKind::Scalar(ScalarKind::F32) => Ok(Some(a)),
            _ => Err(format!(
                "intrinsic `fused_mul_add` expects arguments to be of type `f32`, found `{}`",
                a.to_string(ty_registry)
            )),
        }
    }

    fn affects_state(&self) -> bool {
        false
    }
}
