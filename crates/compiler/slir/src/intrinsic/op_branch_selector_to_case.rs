use serde::{Deserialize, Serialize};

use crate::intrinsic::{Intrinsic, expect_one_arg};
use crate::ty::{TY_PREDICATE, TY_U32, Type, TypeRegistry};

/// Converts a branch selector predicate into a `u32` value selected from a list of cases.
///
/// The case selected is the case at the branch-index position in the [cases] list.
///
/// This is a helper construct used in the RVSDG for optimizing transformations. With code that uses
/// enum types, a fairly common pattern is to have 2 or more switch nodes, the first of which will
/// output the enum's discriminant value: one variant's discriminant per branch. This output value
/// is then converted back into a branch selector predicate with an [OpCaseToBranchSelector] node.
/// The resulting second branch selector predicate is then used by the latter switch nodes.
///
/// This switch-node-discriminant-output pattern may be extracted into a separate
/// [OpBranchSelectorToCase] node. We then look for [OpBranchSelectorToCase] -
/// [OpCaseToBranchSelector] pairs in the RVSDG (where the [OpBranchSelectorToCase]'s output is the
/// [OpCaseToBranchSelector]'s input); if the cases match, then this pair can simply be replaced by
/// the original branch selector predicate. This then subsequently shows that switch nodes that
/// originally seemingly used different branch selector predicates in actuality use the same branch
/// selector predicate (the latter switch nodes are shown to use the same predicate as the first
/// switch node). Switch nodes that use the same branch selector predicate are candidates for
/// switch-node-merging. Switch-node-merging is itself a small optimization, but more importantly,
/// it allows for further optimizations inside the now merged branch regions.
///
/// See the [pred_to_case_extraction][0], the [pred_to_case_to_pred_merging][1] and the
/// [switch_merging][2] RVSDG transforms.
///
/// [0]: crate::rvsdg::transform::pred_to_case_extraction
/// [1]: crate::rvsdg::transform::pred_to_case_to_pred_merging
/// [2]: crate::rvsdg::transform::switch_merging
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpBranchSelectorToCase {
    pub cases: Vec<u32>,
}

impl Intrinsic for OpBranchSelectorToCase {
    fn process_args(
        &self,
        ty_registry: &TypeRegistry,
        args: impl IntoIterator<Item = Type>,
    ) -> Result<Option<Type>, String> {
        let arg = expect_one_arg!("op-branch-selector-to-case", args);

        if arg != TY_PREDICATE {
            return Err(format!(
                "branch-selector-to-case operations expects a `predicate` argument, found `{}`",
                arg.to_string(ty_registry)
            ));
        }

        Ok(Some(TY_U32))
    }
}
