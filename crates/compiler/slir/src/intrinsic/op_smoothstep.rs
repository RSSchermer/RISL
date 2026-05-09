use serde::{Deserialize, Serialize};

use crate::intrinsic::{Intrinsic, expect_three_args};
use crate::ty::{ScalarKind, Type, TypeKind, TypeRegistry};

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpSmoothStep;

impl Intrinsic for OpSmoothStep {
    fn process_args(
        &self,
        ty_registry: &TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        let (edge0, edge1, x) = expect_three_args!("smoothstep", args);

        match (
            &*ty_registry.kind(edge0),
            &*ty_registry.kind(edge1),
            &*ty_registry.kind(x),
        ) {
            (
                TypeKind::Scalar(ScalarKind::F32),
                TypeKind::Scalar(ScalarKind::F32),
                TypeKind::Scalar(ScalarKind::F32),
            ) => Ok(Some(x)),
            _ => Err(format!(
                "intrinsic `smoothstep` expects three floating-point scalar arguments, found \
                `{}`, `{}` and `{}`",
                edge0.to_string(ty_registry),
                edge1.to_string(ty_registry),
                x.to_string(ty_registry)
            )),
        }
    }

    fn affects_state(&self) -> bool {
        false
    }
}
