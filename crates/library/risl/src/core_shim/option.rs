use risl_macros::gpu;

use crate::intrinsic;

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::option::Option::<T>::unwrap_unchecked"))]
pub unsafe fn option_unwrap_unchecked<T>(this: Option<T>) -> T {
    unsafe { intrinsic::option_unwrap_unchecked(this) }
}

// The pre-compiled Option::insert inlines unwrap_unchecked and its unsupported UB-check operation.
#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::option::Option::<T>::insert"))]
pub fn option_insert<T>(this: &mut Option<T>, value: T) -> &mut T {
    *this = Some(value);

    // SAFETY: the assignment above just filled the option.
    unsafe { this.as_mut().unwrap_unchecked() }
}
