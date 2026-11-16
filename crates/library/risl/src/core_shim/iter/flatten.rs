use std::iter::Flatten;
use std::num::NonZero;

use risl_macros::gpu;

// Flatten's pre-compiled `advance_by` and `advance_back_by` methods inline NonZero::new, exposing
// an unsupported transmute, so these are copied from `core` to force the use of our shimmed
// NonZero::new implementation.

#[gpu]
#[cfg_attr(
    rislc,
    rislc::core_shim("<core::iter::Flatten<I> as core::iter::Iterator>::advance_by")
)]
pub fn flatten_advance_by<I, U>(iter: &mut Flatten<I>, n: usize) -> Result<(), NonZero<usize>>
where
    I: Iterator<Item: IntoIterator<IntoIter = U, Item = U::Item>>,
    U: Iterator,
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
        "<core::iter::Flatten<I> as core::iter::DoubleEndedIterator>::advance_back_by"
    )
)]
pub fn flatten_advance_back_by<I, U>(iter: &mut Flatten<I>, n: usize) -> Result<(), NonZero<usize>>
where
    I: DoubleEndedIterator<Item: IntoIterator<IntoIter = U, Item = U::Item>>,
    U: DoubleEndedIterator,
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
