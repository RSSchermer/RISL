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
#[cfg_attr(rislc, rislc::core_shim("core::f32::<impl f32>::mul_add"))]
pub fn f32_mul_add(this: f32, a: f32, b: f32) -> f32 {
    unsafe { intrinsic::mul_add_f32(this, a, b) }
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::f32::<impl f32>::round_ties_even"))]
pub fn f32_round_ties_even(this: f32) -> f32 {
    unsafe { intrinsic::round_ties_even_f32(this) }
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
#[cfg_attr(rislc, rislc::core_shim("core::f32::<impl f32>::clamp"))]
pub fn f32_clamp(this: f32, min: f32, max: f32) -> f32 {
    unsafe { intrinsic::clamp_f32(this, min, max) }
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

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::f32::<impl f32>::exp"))]
pub fn f32_exp(this: f32) -> f32 {
    unsafe { intrinsic::exp_f32(this) }
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::f32::<impl f32>::exp2"))]
pub fn f32_exp2(this: f32) -> f32 {
    unsafe { intrinsic::exp2_f32(this) }
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::f32::<impl f32>::ln"))]
pub fn f32_ln(this: f32) -> f32 {
    unsafe { intrinsic::ln_f32(this) }
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::f32::<impl f32>::log2"))]
pub fn f32_log2(this: f32) -> f32 {
    unsafe { intrinsic::log2_f32(this) }
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::f32::<impl f32>::cos"))]
pub fn f32_cos(this: f32) -> f32 {
    unsafe { intrinsic::cos_f32(this) }
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::f32::<impl f32>::acos"))]
pub fn f32_acos(this: f32) -> f32 {
    unsafe { intrinsic::acos_f32(this) }
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::f32::<impl f32>::cosh"))]
pub fn f32_cosh(this: f32) -> f32 {
    unsafe { intrinsic::cosh_f32(this) }
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::f32::<impl f32>::acosh"))]
pub fn f32_acosh(this: f32) -> f32 {
    unsafe { intrinsic::acosh_f32(this) }
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::f32::<impl f32>::sin"))]
pub fn f32_sin(this: f32) -> f32 {
    unsafe { intrinsic::sin_f32(this) }
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::f32::<impl f32>::asin"))]
pub fn f32_asin(this: f32) -> f32 {
    unsafe { intrinsic::asin_f32(this) }
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::f32::<impl f32>::sinh"))]
pub fn f32_sinh(this: f32) -> f32 {
    unsafe { intrinsic::sinh_f32(this) }
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::f32::<impl f32>::asinh"))]
pub fn f32_asinh(this: f32) -> f32 {
    unsafe { intrinsic::asinh_f32(this) }
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::f32::<impl f32>::tan"))]
pub fn f32_tan(this: f32) -> f32 {
    unsafe { intrinsic::tan_f32(this) }
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::f32::<impl f32>::atan"))]
pub fn f32_atan(this: f32) -> f32 {
    unsafe { intrinsic::atan_f32(this) }
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::f32::<impl f32>::tanh"))]
pub fn f32_tanh(this: f32) -> f32 {
    unsafe { intrinsic::tanh_f32(this) }
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::f32::<impl f32>::atanh"))]
pub fn f32_atanh(this: f32) -> f32 {
    unsafe { intrinsic::atanh_f32(this) }
}
