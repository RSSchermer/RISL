use std::ops;

use risl_macros::gpu;

use crate::intrinsic;

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::slice::<impl [T]>::len"))]
pub fn slice_len<T>(slice: &[T]) -> usize {
    unsafe { intrinsic::slice_len(slice) }
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::slice::<impl [T]>::is_empty"))]
pub fn slice_is_empty<T>(slice: &[T]) -> bool {
    slice_len(slice) == 0
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::slice::<impl [T]>::get"))]
pub fn slice_get<T, I>(slice: &[T], index: I) -> Option<&<I as SliceIndex<[T]>>::Output>
where
    I: SliceIndex<[T]>,
{
    index.get(slice)
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::slice::<impl [T]>::get_mut"))]
pub fn slice_get_mut<T, I>(slice: &[T], index: I) -> Option<&<I as SliceIndex<[T]>>::Output>
where
    I: SliceIndex<[T]>,
{
    index.get(slice)
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::slice::<impl [T]>::get_unchecked"))]
pub unsafe fn slice_get_unchecked<T, I>(slice: &[T], index: I) -> &<I as SliceIndex<[T]>>::Output
where
    I: SliceIndex<[T]>,
{
    unsafe { index.get_unchecked(slice) }
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::slice::<impl [T]>::get_unchecked_mut"))]
pub unsafe fn slice_get_unchecked_mut<T, I>(
    slice: &mut [T],
    index: I,
) -> &mut <I as SliceIndex<[T]>>::Output
where
    I: SliceIndex<[T]>,
{
    unsafe { index.get_unchecked_mut(slice) }
}

#[gpu]
pub trait SliceIndex<T>
where
    T: ?Sized,
{
    type Output: ?Sized;

    fn get(self, slice: &T) -> Option<&Self::Output>;

    fn get_mut(self, slice: &mut T) -> Option<&mut Self::Output>;

    unsafe fn get_unchecked(self, slice: &T) -> &Self::Output;

    unsafe fn get_unchecked_mut(self, slice: &mut T) -> &mut Self::Output;
}

#[gpu]
impl<T> SliceIndex<[T]> for usize {
    type Output = T;

    #[cfg_attr(
        rislc,
        rislc::core_shim("<usize as core::slice::SliceIndex<[T]>>::get")
    )]
    fn get(self, slice: &[T]) -> Option<&Self::Output> {
        if self < slice.len() {
            unsafe { Some(intrinsic::slice_element_ref::<T>(slice, self)) }
        } else {
            None
        }
    }

    #[cfg_attr(
        rislc,
        rislc::core_shim("<usize as core::slice::SliceIndex<[T]>>::get_mut")
    )]
    fn get_mut(self, slice: &mut [T]) -> Option<&mut Self::Output> {
        if self < slice.len() {
            unsafe { Some(intrinsic::slice_element_mut::<T>(slice, self)) }
        } else {
            None
        }
    }

    #[cfg_attr(
        rislc,
        rislc::core_shim("<usize as core::slice::SliceIndex<[T]>>::get_unchecked")
    )]
    unsafe fn get_unchecked(self, slice: &[T]) -> &Self::Output {
        unsafe { intrinsic::slice_element_ref::<T>(slice, self) }
    }

    #[cfg_attr(
        rislc,
        rislc::core_shim("<usize as core::slice::SliceIndex<[T]>>::get_unchecked_mut")
    )]
    unsafe fn get_unchecked_mut(self, slice: &mut [T]) -> &mut Self::Output {
        unsafe { intrinsic::slice_element_mut::<T>(slice, self) }
    }
}

#[gpu]
impl<T> SliceIndex<[T]> for ops::Range<usize> {
    type Output = [T];

    #[cfg_attr(
        rislc,
        rislc::core_shim("<core::ops::Range<usize> as core::slice::SliceIndex<[T]>>::get")
    )]
    fn get(self, slice: &[T]) -> Option<&Self::Output> {
        let in_bounds = self.start <= self.end && self.end <= slice.len();

        if in_bounds {
            unsafe { Some(intrinsic::slice_range::<T>(slice, self.start, self.end)) }
        } else {
            None
        }
    }

    #[cfg_attr(
        rislc,
        rislc::core_shim("<core::ops::Range<usize> as core::slice::SliceIndex<[T]>>::get_mut")
    )]
    fn get_mut(self, slice: &mut [T]) -> Option<&mut Self::Output> {
        let in_bounds = self.start <= self.end && self.end <= slice.len();

        if in_bounds {
            unsafe { Some(intrinsic::slice_range_mut::<T>(slice, self.start, self.end)) }
        } else {
            None
        }
    }

    #[cfg_attr(
        rislc,
        rislc::core_shim(
            "<core::ops::Range<usize> as core::slice::SliceIndex<[T]>>::get_unchecked"
        )
    )]
    unsafe fn get_unchecked(self, slice: &[T]) -> &Self::Output {
        unsafe { intrinsic::slice_range::<T>(slice, self.start, self.end) }
    }

    #[cfg_attr(
        rislc,
        rislc::core_shim(
            "<core::ops::Range<usize> as core::slice::SliceIndex<[T]>>::get_unchecked_mut"
        )
    )]
    unsafe fn get_unchecked_mut(self, slice: &mut [T]) -> &mut Self::Output {
        unsafe { intrinsic::slice_range_mut::<T>(slice, self.start, self.end) }
    }
}
