use std::num::NonZero;

use risl_macros::gpu;

// The body of this shim is mostly identical to the body of the regular Rust `core` version, except
// in that the regular Rust `core` version decorates the fold closure with
// `#[rustc_inherit_overflow_checks]`. This causes rustc to insert an `Assert` terminator, which we
// don't support in RISL. The RISL pass already disables overflow checks through the session config,
// however, the presence of a `#[rustc_inherit_overflow_checks]` attribute overrides this and forces
// an overflow check regardless. Hence the need for this shim.
#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::iter::Iterator::count"))]
pub fn iterator_count<I>(iter: I) -> usize
where
    I: Iterator,
{
    iter.fold(0, |count, _| count + 1)
}

// The body of this shim is identical to the body of the regular Rust `core` version. We still have
// to shim this function, because of MIR optimizations performed by rustc when compiling core/std.
// In this particular case, it will inline NonZero::new, which exposes a transmute that rislc cannot
// handle. Though rislc itself disables certain MIR optimizations like inlining to avoid this
// problem, the `core` library comes precompiled. Hence the need for a shim, despite the identical
// implementation.
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
