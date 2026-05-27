use std::num::NonZero;

use risl_macros::gpu;

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::iter::Iterator::count"))]
pub fn iterator_count<I>(iter: I) -> usize
where
    I: Iterator,
{
    iter.fold(0, |count, _| count + 1)
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
