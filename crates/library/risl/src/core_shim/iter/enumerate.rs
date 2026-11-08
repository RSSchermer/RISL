use std::iter::Enumerate;
use std::num::NonZero;

use risl_macros::gpu;

use crate::intrinsic;

// The pre-compiled core implementation of Enumerate::advance_by inlines NonZero::get, exposing a
// transmute that RISL cannot handle. Copy the core algorithm here so that NonZero::get uses our
// shimmed implementation.
#[gpu]
#[cfg_attr(
    rislc,
    rislc::core_shim("<core::iter::Enumerate<I> as core::iter::Iterator>::advance_by")
)]
pub fn enumerate_advance_by<I>(iter: &mut Enumerate<I>, n: usize) -> Result<(), NonZero<usize>>
where
    I: Iterator,
{
    let remaining = unsafe { intrinsic::enumerate_iter_mut(iter) }.advance_by(n);
    let advanced = match remaining {
        Ok(()) => n,
        Err(rem) => n - rem.get(),
    };

    *unsafe { intrinsic::enumerate_count_mut(iter) } += advanced;

    remaining
}
