use std::iter::Cycle;
use std::num::NonZero;

use risl_macros::gpu;

use crate::intrinsic;

// The pre-compiled core implementation of Cycle::advance_by will inline a NonZero::new
// implementation that is not compatible with RISL. This is effectively a copy of the core
// implementation, but since it does not come pre-compiled with the Rust distribution, it will
// use our shimmed version of NonZero::new.
#[gpu]
#[cfg_attr(
    rislc,
    rislc::core_shim("<core::iter::Cycle<I> as core::iter::Iterator>::advance_by")
)]
pub fn cycle_advance_by<I>(iter: &mut Cycle<I>, n: usize) -> Result<(), NonZero<usize>>
where
    I: Clone + Iterator,
{
    let mut n = match unsafe { intrinsic::cycle_iter_mut(iter) }.advance_by(n) {
        Ok(()) => return Ok(()),
        Err(rem) => rem.get(),
    };

    while n > 0 {
        let orig = unsafe { intrinsic::cycle_orig_ref(iter) }.clone();
        let current = unsafe { intrinsic::cycle_iter_mut(iter) };

        *current = orig;

        n = match current.advance_by(n) {
            Ok(()) => return Ok(()),
            e @ Err(rem) if rem.get() == n => return e,
            Err(rem) => rem.get(),
        };
    }

    NonZero::new(n).map_or(Ok(()), Err)
}
