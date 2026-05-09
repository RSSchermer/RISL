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

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::f32::<impl f32>::min"))]
pub fn f32_min(this: f32, other: f32) -> f32 {
    unsafe { intrinsic::min_f32(this, other) }
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::f32::<impl f32>::max"))]
pub fn f32_max(this: f32, other: f32) -> f32 {
    unsafe { intrinsic::max_f32(this, other) }
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::f32::<impl f32>::round"))]
pub fn f32_round(this: f32) -> f32 {
    unsafe { intrinsic::round_f32(this) }
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::f32::<impl f32>::floor"))]
pub fn f32_floor(this: f32) -> f32 {
    unsafe { intrinsic::floor_f32(this) }
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::f32::<impl f32>::ceil"))]
pub fn f32_ceil(this: f32) -> f32 {
    unsafe { intrinsic::ceil_f32(this) }
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::f32::<impl f32>::fract"))]
pub fn f32_fract(this: f32) -> f32 {
    // Note: though we have a SLIR `fract` intrinsic, we don't use it to implement this function,
    // as the WGSL-specified behavior for `fract` differs from the Rust behavior:
    //
    // WGSL: this - floor(this) (Example: -1.5 - (floor(-1.5)) = -1.5 + 2.0 = 0.5)
    // Rust: this - trunc(this) (Example: -1.5 - (trunc(-1.5)) = -1.5 + 1.0 = -0.5)
    //
    // We want this function to match the Rust behavior.
    this - f32_trunc(this)
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::f32::<impl f32>::trunc"))]
pub fn f32_trunc(this: f32) -> f32 {
    unsafe { intrinsic::trunc_f32(this) }
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::f32::<impl f32>::sqrt"))]
pub fn f32_sqrt(this: f32) -> f32 {
    unsafe { intrinsic::sqrt_f32(this) }
}
