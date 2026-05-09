use risl_macros::gpu;

use crate::intrinsic;

#[gpu]
#[cfg_attr(
    rislc,
    rislc::core_shim("core::cmp::impls::<impl core::cmp::Ord for u32>::clamp")
)]
pub fn u32_clamp(this: u32, min: u32, max: u32) -> u32 {
    unsafe { intrinsic::clamp_u32(this, min, max) }
}

#[gpu]
#[cfg_attr(
    rislc,
    rislc::core_shim("core::cmp::impls::<impl core::cmp::Ord for i32>::clamp")
)]
pub fn i32_clamp(this: i32, min: i32, max: i32) -> i32 {
    unsafe { intrinsic::clamp_i32(this, min, max) }
}

#[gpu]
#[cfg_attr(
    rislc,
    rislc::core_shim("core::cmp::impls::<impl core::cmp::Ord for usize>::clamp")
)]
pub fn usize_clamp(this: usize, min: usize, max: usize) -> usize {
    unsafe { intrinsic::clamp_usize(this, min, max) }
}

#[gpu]
#[cfg_attr(
    rislc,
    rislc::core_shim("core::cmp::impls::<impl core::cmp::Ord for isize>::clamp")
)]
pub fn isize_clamp(this: isize, min: isize, max: isize) -> isize {
    unsafe { intrinsic::clamp_isize(this, min, max) }
}
