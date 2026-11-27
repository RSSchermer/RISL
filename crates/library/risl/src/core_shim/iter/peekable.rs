use std::iter::Peekable;

use risl_macros::gpu;

use crate::intrinsic;

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::iter::Peekable::<I>::next_if"))]
pub fn peekable_next_if<I, F>(iter: &mut Peekable<I>, func: F) -> Option<I::Item>
where
    I: Iterator,
    F: FnOnce(&I::Item) -> bool,
{
    match iter.next() {
        Some(matched) if func(&matched) => Some(matched),
        other => {
            // The version in `core` asserts that the peeked item stores on the adapter is None
            // here. This seems to be a sanity check, as the call to `next` above should ensure
            // that this is always the case. Since it seems sound to remove this check, and since
            // RISL cannot panic, this shims copies the `core` implementation, minus that sanity
            // check.
            *unsafe { intrinsic::peekable_peeked_mut(iter) } = Some(other);

            None
        }
    }
}
