use core::slice::{Iter, IterMut};

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
