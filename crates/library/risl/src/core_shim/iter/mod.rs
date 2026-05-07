use std::num::NonZero;
use std::ops::Try;

use risl_macros::gpu;

/// Shim marker to help the RISL compiler recognize the [core::iter::IntoIterator] trait as a
/// GPU-compatible trait.
#[cfg_attr(rislc, rislc::core_shim("core::iter::IntoIterator"))]
#[gpu]
pub trait IntoIteratorShimMarker {}

/// Shim marker to help the RISL compiler recognize the [core::iter::Iterator] trait as a
/// GPU-compatible trait.
#[cfg_attr(rislc, rislc::core_shim("core::iter::Iterator"))]
#[gpu]
pub trait IteratorShimMarker {}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::iter::Iterator::size_hint"))]
pub fn iterator_size_hint<I>(_: &I) -> (usize, Option<usize>)
where
    I: Iterator + ?Sized,
{
    (0, None)
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::iter::Iterator::fold"))]
pub fn iterator_fold<I, B, F>(mut iter: I, init: B, mut f: F) -> B
where
    I: Iterator,
    F: FnMut(B, I::Item) -> B,
{
    let mut accum = init;

    while let Some(x) = Iterator::next(&mut iter) {
        accum = f(accum, x);
    }

    accum
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::iter::Iterator::try_fold"))]
pub fn iterator_try_fold<I, B, F, R>(iter: &mut I, init: B, mut f: F) -> R
where
    I: Iterator,
    F: FnMut(B, I::Item) -> R,
    R: Try<Output = B>,
{
    let mut accum = init;

    while let Some(x) = Iterator::next(iter) {
        accum = f(accum, x)?;
    }

    R::from_output(accum)
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::iter::Iterator::count"))]
pub fn iterator_count<I>(iter: I) -> usize
where
    I: Iterator,
{
    iterator_fold(iter, 0, |count, _| count + 1)
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::iter::Iterator::last"))]
pub fn iterator_last<I>(iter: I) -> Option<I::Item>
where
    I: Iterator,
{
    fn some<T>(_: Option<T>, x: T) -> Option<T> {
        Some(x)
    }

    iterator_fold(iter, None, some)
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::iter::Iterator::nth"))]
pub fn iterator_nth<I>(iter: &mut I, n: usize) -> Option<I::Item>
where
    I: Iterator + ?Sized,
{
    iterator_advance_by(iter, n).ok()?;
    iter.next()
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::iter::Iterator::advance_by"))]
pub fn iterator_advance_by<I>(iter: &mut I, n: usize) -> Result<(), NonZero<usize>>
where
    I: Iterator + ?Sized,
{
    /// Helper trait to specialize `advance_by` via `try_fold` for `Sized` iterators.
    trait SpecAdvanceBy {
        fn spec_advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>>;
    }

    impl<I: Iterator + ?Sized> SpecAdvanceBy for I {
        default fn spec_advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
            for i in 0..n {
                if self.next().is_none() {
                    // SAFETY: `i` is always less than `n`.
                    return Err(unsafe { NonZero::new_unchecked(n - i) });
                }
            }
            Ok(())
        }
    }

    impl<I: Iterator> SpecAdvanceBy for I {
        fn spec_advance_by(&mut self, n: usize) -> Result<(), NonZero<usize>> {
            let Some(n) = NonZero::new(n) else {
                return Ok(());
            };

            let res = self.try_fold(n, |n, _| NonZero::new(n.get() - 1));

            match res {
                None => Ok(()),
                Some(n) => Err(n),
            }
        }
    }

    iter.spec_advance_by(n)
}
