use std::num::{NonZero, ZeroablePrimitive};

use risl_macros::gpu;

use crate::intrinsic;

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::num::NonZero::<T>::new"))]
pub fn non_zero_new<T: ZeroablePrimitive>(n: T) -> Option<NonZero<T>> {
    unsafe { intrinsic::non_zero_new(n) }
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::num::NonZero::<T>::new_unchecked"))]
pub unsafe fn non_zero_new_unchecked<T: ZeroablePrimitive>(n: T) -> NonZero<T> {
    unsafe { intrinsic::non_zero_new_unchecked(n) }
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::num::NonZero::<T>::get"))]
pub fn non_zero_get<T: ZeroablePrimitive>(this: NonZero<T>) -> T {
    unsafe { intrinsic::non_zero_get(this) }
}
