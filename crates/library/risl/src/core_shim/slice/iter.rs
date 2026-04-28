use std::mem;

use risl_macros::gpu;

use crate::core_shim;

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::slice::<impl [T]>::iter"))]
pub fn slice_iter<T>(slice: &[T]) -> Iter<'_, T> {
    Iter {
        slice,
        start: 0,
        end: slice.len(),
    }
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
    slice_iter(slice)
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::slice::<impl [T]>::iter_mut"))]
pub fn slice_iter_mut<T>(slice: &mut [T]) -> IterMut<'_, T> {
    IterMut {
        start: 0,
        end: slice.len(),
        slice,
    }
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
    slice_iter_mut(slice)
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::slice::Iter"))]
pub struct Iter<'a, T> {
    slice: &'a [T],
    start: usize,
    end: usize,
}

#[gpu]
impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[cfg_attr(
        rislc,
        rislc::core_shim("<core::slice::Iter<'a, T> as core::iter::Iterator>::next")
    )]
    fn next(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            let result = unsafe { self.slice.get_unchecked(self.start) };

            self.start += 1;

            Some(result)
        } else {
            None
        }
    }
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::slice::IterMut"))]
pub struct IterMut<'a, T>
where
    T: 'a,
{
    slice: &'a mut [T],
    start: usize,
    end: usize,
}

#[gpu]
impl<'a, T> Iterator for IterMut<'a, T>
where
    T: 'a,
{
    type Item = &'a mut T;

    #[cfg_attr(
        rislc,
        rislc::core_shim("<core::slice::IterMut<'a, T> as core::iter::Iterator>::next")
    )]
    fn next(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            let result = unsafe { self.slice.get_unchecked_mut(self.start) };

            self.start += 1;

            Some(unsafe { mem::transmute(result) })
        } else {
            None
        }
    }
}
