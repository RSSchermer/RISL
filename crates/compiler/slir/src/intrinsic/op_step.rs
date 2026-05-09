use serde::{Deserialize, Serialize};

use crate::intrinsic::{Intrinsic, expect_two_args};
use crate::ty::{ScalarKind, Type, TypeKind, TypeRegistry};

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpStep;

impl Intrinsic for OpStep {
    fn process_args(
        &self,
        ty_registry: &TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        let (edge, x) = expect_two_args!("step", args);

        match (&*ty_registry.kind(edge), &*ty_registry.kind(x)) {
            (TypeKind::Scalar(ScalarKind::F32), TypeKind::Scalar(ScalarKind::F32)) => Ok(Some(x)),
            _ => Err(format!(
                "intrinsic `step` expects two floating-point scalar arguments, found `{}` and `{}`",
                edge.to_string(ty_registry),
                x.to_string(ty_registry)
            )),
        }
    }

    fn affects_state(&self) -> bool {
        false
    }
}
