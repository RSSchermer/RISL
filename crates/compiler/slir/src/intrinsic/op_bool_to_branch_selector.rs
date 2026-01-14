use serde::{Deserialize, Serialize};

use crate::intrinsic::{Intrinsic, expect_one_arg};
use crate::ty::{TY_BOOL, TY_PREDICATE, Type, TypeRegistry};

/// Converts a boolean value into a branch selector predicate.
///
/// If the boolean value is `true`, then the predicate will select branch `0`. If the boolean value
/// is `false` then the predicate will select branch `1`.
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpBoolToBranchSelector;

impl Intrinsic for OpBoolToBranchSelector {
    fn process_args(
        &self,
        ty_registry: &TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        let arg = expect_one_arg!("op-bool-to-branch-selector", args);

        if arg != TY_BOOL {
            return Err(format!(
                "bool-to-branch-selector operation expects its argument to be a `bool`, found `{}`",
                arg.to_string(ty_registry)
            ));
        }

        Ok(Some(TY_PREDICATE))
    }
}
