use std::iter::Chain;
use std::num::NonZero;

use risl_macros::gpu;

use crate::intrinsic;

// Precompiled core inlines NonZero::get/new and integer intrinsics into these methods.
// Keep the traversal and fusion behavior, but compile the bodies without that inlining.
#[gpu]
#[cfg_attr(
    rislc,
    rislc::core_shim("<core::iter::Chain<A, B> as core::iter::Iterator>::advance_by")
)]
pub fn chain_advance_by<A, B>(iter: &mut Chain<A, B>, mut n: usize) -> Result<(), NonZero<usize>>
where
    A: Iterator,
    B: Iterator<Item = A::Item>,
{
    let a = unsafe { intrinsic::chain_a_mut(iter) };

    if let Some(a_iter) = a {
        n = match a_iter.advance_by(n) {
            Ok(()) => return Ok(()),
            Err(k) => k.get(),
        };

        *a = None;
    }

    if let Some(b) = unsafe { intrinsic::chain_b_mut(iter) } {
        return b.advance_by(n);
    }

    NonZero::new(n).map_or(Ok(()), Err)
}

#[gpu]
#[cfg_attr(
    rislc,
    rislc::core_shim("<core::iter::Chain<A, B> as core::iter::Iterator>::nth")
)]
pub fn chain_nth<A, B>(iter: &mut Chain<A, B>, mut n: usize) -> Option<A::Item>
where
    A: Iterator,
    B: Iterator<Item = A::Item>,
{
    let a = unsafe { intrinsic::chain_a_mut(iter) };

    if let Some(a_iter) = a {
        n = match a_iter.advance_by(n) {
            Ok(()) => match a_iter.next() {
                None => 0,
                x => return x,
            },
            Err(k) => k.get(),
        };

        *a = None;
    }

    unsafe { intrinsic::chain_b_mut(iter) }.as_mut()?.nth(n)
}

#[gpu]
#[cfg_attr(
    rislc,
    rislc::core_shim(
        "<core::iter::Chain<A, B> as core::iter::DoubleEndedIterator>::advance_back_by"
    )
)]
pub fn chain_advance_back_by<A, B>(
    iter: &mut Chain<A, B>,
    mut n: usize,
) -> Result<(), NonZero<usize>>
where
    A: DoubleEndedIterator,
    B: DoubleEndedIterator<Item = A::Item>,
{
    let b = unsafe { intrinsic::chain_b_mut(iter) };

    if let Some(b_iter) = b {
        n = match b_iter.advance_back_by(n) {
            Ok(()) => return Ok(()),
            Err(k) => k.get(),
        };

        *b = None;
    }

    if let Some(a) = unsafe { intrinsic::chain_a_mut(iter) } {
        return a.advance_back_by(n);
    }

    NonZero::new(n).map_or(Ok(()), Err)
}

#[gpu]
#[cfg_attr(
    rislc,
    rislc::core_shim("<core::iter::Chain<A, B> as core::iter::DoubleEndedIterator>::nth_back")
)]
pub fn chain_nth_back<A, B>(iter: &mut Chain<A, B>, mut n: usize) -> Option<A::Item>
where
    A: DoubleEndedIterator,
    B: DoubleEndedIterator<Item = A::Item>,
{
    let b = unsafe { intrinsic::chain_b_mut(iter) };

    if let Some(b_iter) = b {
        n = match b_iter.advance_back_by(n) {
            Ok(()) => match b_iter.next_back() {
                None => 0,
                x => return x,
            },
            Err(k) => k.get(),
        };

        *b = None;
    }

    unsafe { intrinsic::chain_a_mut(iter) }
        .as_mut()?
        .nth_back(n)
}

#[gpu]
#[cfg_attr(
    rislc,
    rislc::core_shim("<core::iter::Chain<A, B> as core::iter::Iterator>::size_hint")
)]
pub fn chain_size_hint<A, B>(iter: &Chain<A, B>) -> (usize, Option<usize>)
where
    A: Iterator,
    B: Iterator<Item = A::Item>,
{
    let a = unsafe { intrinsic::chain_a_ref(iter) };
    let b = unsafe { intrinsic::chain_b_ref(iter) };

    match (a, b) {
        (Some(a), Some(b)) => {
            let (a_lower, a_upper) = a.size_hint();
            let (b_lower, b_upper) = b.size_hint();

            let lower = if b_lower > usize::MAX - a_lower {
                usize::MAX
            } else {
                a_lower + b_lower
            };

            let upper = match (a_upper, b_upper) {
                (Some(x), Some(y)) => {
                    if y > usize::MAX - x {
                        None
                    } else {
                        Some(x + y)
                    }
                }
                _ => None,
            };

            (lower, upper)
        }
        (Some(a), None) => a.size_hint(),
        (None, Some(b)) => b.size_hint(),
        (None, None) => (0, Some(0)),
    }
}
