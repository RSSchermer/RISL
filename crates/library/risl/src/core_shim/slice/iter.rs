use core::slice::{Iter, IterMut};
use std::num::NonZero;

use risl_macros::gpu;

use crate::intrinsic;

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::slice::<impl [T]>::iter"))]
pub fn slice_iter<T>(slice: &[T]) -> Iter<'_, T> {
    unsafe { intrinsic::slice_iter_new(slice) }
}

// Note: the `T: 'a` bound is required to force the same signature shape as `impl IntoIterator`
#[gpu]
#[cfg_attr(
    rislc,
    rislc::core_shim("core::slice::iter::<impl core::iter::IntoIterator for &'a [T]>::into_iter")
)]
pub fn slice_into_iter<'a, T>(slice: &'a [T]) -> Iter<'a, T>
where
    T: 'a,
{
    unsafe { intrinsic::slice_iter_new(slice) }
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::slice::<impl [T]>::iter_mut"))]
pub fn slice_iter_mut<T>(slice: &mut [T]) -> IterMut<'_, T> {
    unsafe { intrinsic::slice_iter_mut_new(slice) }
}

// Note: the `T: 'a` bound is required to force the same signature shape as `impl IntoIterator`
#[gpu]
#[cfg_attr(
    rislc,
    rislc::core_shim(
        "core::slice::iter::<impl core::iter::IntoIterator for &'a mut [T]>::into_iter"
    )
)]
pub fn slice_into_iter_mut<'a, T>(slice: &'a mut [T]) -> IterMut<'a, T>
where
    T: 'a,
{
    unsafe { intrinsic::slice_iter_mut_new(slice) }
}

fn slice_iter_len<T>(iter: &Iter<'_, T>) -> usize {
    unsafe {
        let start = intrinsic::slice_iter_start(iter);
        let end = intrinsic::slice_iter_end(iter);

        end - start
    }
}

fn slice_iter_mut_len<T>(iter: &IterMut<'_, T>) -> usize {
    unsafe {
        let start = intrinsic::slice_iter_mut_start(iter);
        let end = intrinsic::slice_iter_mut_end(iter);

        end - start
    }
}

#[gpu]
#[cfg_attr(
    rislc,
    rislc::core_shim("<core::slice::Iter<'_, T> as core::clone::Clone>::clone")
)]
pub fn slice_iter_iterator_clone<'a, T>(iter: &Iter<'a, T>) -> Iter<'a, T>
where
    T: 'a,
{
    unsafe { intrinsic::slice_iter_clone(iter) }
}

#[gpu]
#[cfg_attr(
    rislc,
    rislc::core_shim("<core::slice::Iter<'a, T> as core::iter::Iterator>::next")
)]
pub fn slice_iter_iterator_next<'a, T>(iter: &mut Iter<'a, T>) -> Option<&'a T>
where
    T: 'a,
{
    unsafe {
        let start = intrinsic::slice_iter_start(iter);
        let end = intrinsic::slice_iter_end(iter);

        if start < end {
            let result = intrinsic::slice_iter_get_unchecked(iter, start);

            intrinsic::slice_iter_set_start(iter, start + 1);

            Some(result)
        } else {
            None
        }
    }
}

#[gpu]
#[cfg_attr(
    rislc,
    rislc::core_shim("<core::slice::IterMut<'a, T> as core::iter::Iterator>::next")
)]
pub fn slice_iter_mut_iterator_next<'a, T>(iter: &mut IterMut<'a, T>) -> Option<&'a mut T>
where
    T: 'a,
{
    unsafe {
        let start = intrinsic::slice_iter_mut_start(iter);
        let end = intrinsic::slice_iter_mut_end(iter);

        if start < end {
            let result = intrinsic::slice_iter_mut_get_unchecked(iter, start);

            intrinsic::slice_iter_mut_set_start(iter, start + 1);

            Some(result)
        } else {
            None
        }
    }
}

#[gpu]
#[cfg_attr(
    rislc,
    rislc::core_shim("<core::slice::Iter<'a, T> as core::iter::Iterator>::size_hint")
)]
pub fn slice_iter_iterator_size_hint<'a, T>(iter: &Iter<'a, T>) -> (usize, Option<usize>)
where
    T: 'a,
{
    let exact = slice_iter_len(iter);

    (exact, Some(exact))
}

#[gpu]
#[cfg_attr(
    rislc,
    rislc::core_shim("<core::slice::IterMut<'a, T> as core::iter::Iterator>::size_hint")
)]
pub fn slice_iter_iterator_mut_size_hint<'a, T>(iter: &IterMut<'a, T>) -> (usize, Option<usize>)
where
    T: 'a,
{
    let exact = slice_iter_mut_len(iter);

    (exact, Some(exact))
}

#[gpu]
#[cfg_attr(
    rislc,
    rislc::core_shim("<core::slice::Iter<'a, T> as core::iter::Iterator>::count")
)]
pub fn slice_iter_iterator_count<'a, T>(iter: Iter<'a, T>) -> usize
where
    T: 'a,
{
    slice_iter_len(&iter)
}

#[gpu]
#[cfg_attr(
    rislc,
    rislc::core_shim("<core::slice::IterMut<'a, T> as core::iter::Iterator>::count")
)]
pub fn slice_iter_mut_iterator_count<'a, T>(iter: IterMut<'a, T>) -> usize
where
    T: 'a,
{
    slice_iter_mut_len(&iter)
}

#[gpu]
#[cfg_attr(
    rislc,
    rislc::core_shim("<core::slice::Iter<'a, T> as core::iter::Iterator>::nth")
)]
pub fn slice_iter_iterator_nth<'a, T>(iter: &mut Iter<'a, T>, n: usize) -> Option<&'a T>
where
    T: 'a,
{
    unsafe {
        let start = intrinsic::slice_iter_start(iter);
        let end = intrinsic::slice_iter_end(iter);

        let index = start + n;

        if index < end {
            let result = intrinsic::slice_iter_get_unchecked(iter, index);

            intrinsic::slice_iter_set_start(iter, index + 1);

            Some(result)
        } else {
            intrinsic::slice_iter_set_start(iter, end);

            None
        }
    }
}

#[gpu]
#[cfg_attr(
    rislc,
    rislc::core_shim("<core::slice::IterMut<'a, T> as core::iter::Iterator>::nth")
)]
pub fn slice_iter_mut_iterator_nth<'a, T>(iter: &mut IterMut<'a, T>, n: usize) -> Option<&'a mut T>
where
    T: 'a,
{
    unsafe {
        let start = intrinsic::slice_iter_mut_start(iter);
        let end = intrinsic::slice_iter_mut_end(iter);

        let index = start + n;

        if index < end {
            let result = intrinsic::slice_iter_mut_get_unchecked(iter, index);

            intrinsic::slice_iter_mut_set_start(iter, index + 1);

            Some(result)
        } else {
            intrinsic::slice_iter_mut_set_start(iter, end);

            None
        }
    }
}

#[gpu]
#[cfg_attr(
    rislc,
    rislc::core_shim("<core::slice::Iter<'a, T> as core::iter::Iterator>::advance_by")
)]
pub fn slice_iter_iterator_advance_by<'a, T>(
    iter: &mut Iter<'a, T>,
    n: usize,
) -> Result<(), NonZero<usize>>
where
    T: 'a,
{
    let len = slice_iter_len(iter);
    let advance = usize::min(n, len);
    let rem = n - advance;

    unsafe {
        let start = intrinsic::slice_iter_start(iter);
        intrinsic::slice_iter_set_start(iter, start + advance);
    }

    if rem == 0 {
        Ok(())
    } else {
        // SAFETY: the condition guards against zero values
        unsafe { Err(NonZero::new_unchecked(rem)) }
    }
}

#[gpu]
#[cfg_attr(
    rislc,
    rislc::core_shim("<core::slice::IterMut<'a, T> as core::iter::Iterator>::advance_by")
)]
pub fn slice_iter_mut_iterator_advance_by<'a, T>(
    iter: &mut IterMut<'a, T>,
    n: usize,
) -> Result<(), NonZero<usize>>
where
    T: 'a,
{
    let len = slice_iter_mut_len(iter);
    let advance = usize::min(n, len);
    let rem = n - advance;

    unsafe {
        let start = intrinsic::slice_iter_mut_start(iter);
        intrinsic::slice_iter_mut_set_start(iter, start + advance);
    }

    if rem == 0 {
        Ok(())
    } else {
        // SAFETY: the condition guards against zero values
        unsafe { Err(NonZero::new_unchecked(rem)) }
    }
}

#[gpu]
#[cfg_attr(
    rislc,
    rislc::core_shim("<core::slice::Iter<'a, T> as core::iter::DoubleEndedIterator>::next_back")
)]
pub fn slice_iter_iterator_next_back<'a, T>(iter: &mut Iter<'a, T>) -> Option<&'a T>
where
    T: 'a,
{
    unsafe {
        let start = intrinsic::slice_iter_start(iter);
        let end = intrinsic::slice_iter_end(iter);

        if start < end {
            let index = end - 1;
            let result = intrinsic::slice_iter_get_unchecked(iter, index);

            intrinsic::slice_iter_set_end(iter, index);

            Some(result)
        } else {
            None
        }
    }
}

#[gpu]
#[cfg_attr(
    rislc,
    rislc::core_shim(
        "<core::slice::IterMut<'a, T> as core::iter::DoubleEndedIterator>::next_back"
    )
)]
pub fn slice_iter_mut_iterator_next_back<'a, T>(iter: &mut IterMut<'a, T>) -> Option<&'a mut T>
where
    T: 'a,
{
    unsafe {
        let start = intrinsic::slice_iter_mut_start(iter);
        let end = intrinsic::slice_iter_mut_end(iter);

        if start < end {
            let index = end - 1;
            let result = intrinsic::slice_iter_mut_get_unchecked(iter, index);

            intrinsic::slice_iter_mut_set_end(iter, index);

            Some(result)
        } else {
            None
        }
    }
}
