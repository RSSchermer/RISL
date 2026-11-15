use std::iter::FlatMap;
use std::num::NonZero;

use risl_macros::gpu;

// FlatMap's pre-compiled `advance_by` and `advance_back_by` methods inline NonZero::new, exposing
// an unsupported transmute, so these are copied from `core` to force the use of our shimmed
// NonZero::new implementation.

#[gpu]
#[cfg_attr(
    rislc,
    rislc::core_shim("<core::iter::FlatMap<I, U, F> as core::iter::Iterator>::advance_by")
)]
pub fn flat_map_advance_by<I, U, F>(
    iter: &mut FlatMap<I, U, F>,
    n: usize,
) -> Result<(), NonZero<usize>>
where
    I: Iterator,
    U: IntoIterator,
    F: FnMut(I::Item) -> U,
{
    let Some(n) = NonZero::new(n) else {
        return Ok(());
    };

    let res = iter.try_fold(n, |n, _| NonZero::new(n.get() - 1));

    match res {
        None => Ok(()),
        Some(n) => Err(n),
    }
}

#[gpu]
#[cfg_attr(
    rislc,
    rislc::core_shim(
        "<core::iter::FlatMap<I, U, F> as core::iter::DoubleEndedIterator>::advance_back_by"
    )
)]
pub fn flat_map_advance_back_by<I, U, F>(
    iter: &mut FlatMap<I, U, F>,
    n: usize,
) -> Result<(), NonZero<usize>>
where
    I: DoubleEndedIterator,
    U: IntoIterator,
    U::IntoIter: DoubleEndedIterator,
    F: FnMut(I::Item) -> U,
{
    let Some(n) = NonZero::new(n) else {
        return Ok(());
    };

    let res = iter.try_rfold(n, |n, _| NonZero::new(n.get() - 1));

    match res {
        None => Ok(()),
        Some(n) => Err(n),
    }
}
