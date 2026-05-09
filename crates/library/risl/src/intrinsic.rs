//! Intrinsics for the RISL compiler.

use std::num::{NonZero, ZeroablePrimitive};

use crate::gpu;
use crate::mem::{Storage, StorageMut, Uniform, Workgroup};

macro_rules! gen_intrinsic {
    ($intrinsic:ident: $($decl_token:tt)*) => {
        #[cfg_attr(rislc, rislc::intrinsic($intrinsic))]
        #[gpu]
        pub unsafe $($decl_token)* {
             #[cfg(rislc)]
            core::intrinsics::abort();

            #[cfg(not(rislc))]
            panic!("RISL intrinsics cannot be called outside of a GPU context");
        }
    };
}

gen_intrinsic!(mem_resource_as_ref: fn uniform_as_ref<T>(uniform: &Uniform<T>) -> &T);
gen_intrinsic!(mem_resource_as_ref: fn storage_as_ref<T>(storage: &Storage<T>) -> &T where T: ?Sized);
gen_intrinsic!(mem_resource_as_ref: fn storage_mut_as_ref<T>(storage: &StorageMut<T>) -> &T where T: ?Sized);
gen_intrinsic!(mem_resource_as_ref: fn storage_mut_as_mut<T>(storage: &StorageMut<T>) -> &mut T where T: ?Sized);
gen_intrinsic!(mem_resource_as_ref: fn workgroup_as_ref<T>(workgroup: &Workgroup<T>) -> &T);
gen_intrinsic!(mem_resource_as_ref: fn workgroup_as_mut<T>(workgroup: &Workgroup<T>) -> &mut T);
gen_intrinsic!(slice_len: fn slice_len<T>(slice: &[T]) -> usize);
gen_intrinsic!(slice_element_ref: fn slice_element_ref<T>(slice: &[T], index: usize) -> &T);
gen_intrinsic!(slice_element_ref: fn slice_element_mut<T>(slice: &mut [T], index: usize) -> &mut T);
gen_intrinsic!(slice_range: fn slice_range<T>(slice: &[T], start: usize, end: usize) -> &[T]);
gen_intrinsic!(slice_range: fn slice_range_mut<T>(slice: &mut [T], start: usize, end: usize) -> &mut [T]);
gen_intrinsic!(non_zero_new: fn non_zero_new<T: ZeroablePrimitive>(n: T) -> Option<NonZero<T>>);
gen_intrinsic!(non_zero_new_unchecked: fn non_zero_new_unchecked<T: ZeroablePrimitive>(n: T) -> NonZero<T>);
gen_intrinsic!(non_zero_get: fn non_zero_get<T: ZeroablePrimitive>(n: NonZero<T>) -> T);
gen_intrinsic!(min_f32: fn min_f32(a: f32, b: f32) -> f32);
gen_intrinsic!(max_f32: fn max_f32(a: f32, b: f32) -> f32);
gen_intrinsic!(round_f32: fn round_f32(a: f32) -> f32);
gen_intrinsic!(floor_f32: fn floor_f32(a: f32) -> f32);
gen_intrinsic!(ceil_f32: fn ceil_f32(a: f32) -> f32);
gen_intrinsic!(fract_f32: fn fract_f32(a: f32) -> f32);
gen_intrinsic!(trunc_f32: fn trunc_f32(a: f32) -> f32);
gen_intrinsic!(sqrt_f32: fn sqrt_f32(a: f32) -> f32);
gen_intrinsic!(inverse_sqrt_f32: fn inverse_sqrt_f32(a: f32) -> f32);
gen_intrinsic!(exp_f32: fn exp_f32(a: f32) -> f32);
gen_intrinsic!(exp2_f32: fn exp2_f32(a: f32) -> f32);
gen_intrinsic!(ln_f32: fn ln_f32(a: f32) -> f32);
gen_intrinsic!(log2_f32: fn log2_f32(a: f32) -> f32);
