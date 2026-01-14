use crate::intrinsic::{Intrinsic, expect_one_arg};
use crate::ty::{TY_PREDICATE, TY_U32, Type, TypeRegistry};

/// Converts a `u32` value into a branch selector predicate.
///
/// The branch selected is the `u32` value clamped to the range `0..branch_count`.
pub struct OpU32ToBranchSelector {
    pub branch_count: u32,
}

impl Intrinsic for OpU32ToBranchSelector {
    fn process_args(
        &self,
        ty_registry: &TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        let arg = expect_one_arg!("op-u32-to-branch-selector", args);

        if arg != TY_U32 {
            return Err(format!(
                "u32-to-branch-selector operation expects its argument to be a `u32`, found `{}`",
                arg.to_string(ty_registry)
            ));
        }

        Ok(Some(TY_PREDICATE))
    }
}
