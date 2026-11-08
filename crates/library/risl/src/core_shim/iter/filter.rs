use std::iter::Filter;

use risl_macros::gpu;

// The pre-compiled core implementation adds a TrustedLen optimizer assertion that exposes an
// unsupported runtime UB check.
#[gpu]
#[cfg_attr(
    rislc,
    rislc::core_shim("<core::iter::Filter<I, P> as core::iter::Iterator>::count")
)]
pub fn filter_count<I, P>(iter: Filter<I, P>) -> usize
where
    I: Iterator,
    P: FnMut(&I::Item) -> bool,
{
    // The core implementation is branchless, using bool-to-usize conversion. However, replicating
    // it here requires unpacking the iter and predicate from the Filter struct, which is possible,
    // but requires a complicated intrinsic. The `fold` implementation does produce a branch, but is
    // much simpler to implement, so we'll accept the potential inefficiency for now.
    iter.fold(0, |count, _| count + 1)
}
