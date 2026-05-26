use serde::{Deserialize, Serialize};

use crate::intrinsic::{Intrinsic, expect_one_arg};
use crate::ty::{Int, TY_PREDICATE, Type, TypeRegistry};

/// Converts an integer into a branch selector predicate by comparing it against a list of cases.
///
/// If it matches one case at index `n` in the [cases] list, then the predicate produced will select
/// branch `n`. If it matches multiple cases, then `n` will be the index of the first case matched
/// in list-order. If it matches none of the cases, then the predicate will select branch
/// [cases.len()].
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpCaseToBranchSelector {
    pub encoding: Int,
    pub cases: Vec<u128>,
}

impl Intrinsic for OpCaseToBranchSelector {
    fn process_args(
        &self,
        ty_registry: &TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        let arg = expect_one_arg!("op-case-to-branch-selector", args);

        if arg != self.encoding.backend_ty() {
            return Err(format!(
                "case-to-branch-selector operation expects its argument to be a `{}`, found `{}`",
                self.encoding.backend_ty().to_string(ty_registry),
                arg.to_string(ty_registry)
            ));
        }

        Ok(Some(TY_PREDICATE))
    }

    fn affects_state(&self) -> bool {
        false
    }
}
