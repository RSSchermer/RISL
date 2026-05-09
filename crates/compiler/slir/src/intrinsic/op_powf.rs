use serde::{Deserialize, Serialize};

use crate::intrinsic::{Intrinsic, expect_two_args};
use crate::ty::{ScalarKind, Type, TypeKind, TypeRegistry};

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpPowf;

impl Intrinsic for OpPowf {
    fn process_args(
        &self,
        ty_registry: &TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        let (base, exp) = expect_two_args!("powf", args);

        match (&*ty_registry.kind(base), &*ty_registry.kind(exp)) {
            (TypeKind::Scalar(ScalarKind::F32), TypeKind::Scalar(ScalarKind::F32)) => {
                Ok(Some(base))
            }
            _ => Err(format!(
                "intrinsic `powf` expects two floating-point scalar arguments, found `{}` and `{}`",
                base.to_string(ty_registry),
                exp.to_string(ty_registry)
            )),
        }
    }

    fn affects_state(&self) -> bool {
        false
    }
}
