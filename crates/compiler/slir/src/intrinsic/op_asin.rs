use serde::{Deserialize, Serialize};

use crate::intrinsic::{Intrinsic, expect_one_arg};
use crate::ty::{ScalarKind, Type, TypeKind, TypeRegistry};

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpAsin;

impl Intrinsic for OpAsin {
    fn process_args(
        &self,
        ty_registry: &TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        let arg = expect_one_arg!("asin", args);

        match &*ty_registry.kind(arg) {
            TypeKind::Scalar(ScalarKind::F32) => Ok(Some(arg)),
            _ => Err(format!(
                "intrinsic `asin` expects a floating-point scalar argument, found `{}`",
                arg.to_string(ty_registry)
            )),
        }
    }

    fn affects_state(&self) -> bool {
        false
    }
}
