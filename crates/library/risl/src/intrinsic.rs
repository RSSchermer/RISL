//! Intrinsics for the RISL compiler.

use std::num::{NonZero, ZeroablePrimitive};

use crate::gpu;
use crate::mem::{Storage, StorageMut, Uniform, Workgroup};

macro_rules! gen_intrinsic {
    ($intrinsic:ident: $($decl_token:tt)*) => {
        #[cfg_attr(rislc, rislc::intrinsic($intrinsic))]
        #[gpu]
        #[allow(unused_variables)]
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

gen_intrinsic!(slice_iter_new: fn slice_iter_new<T>(slice: &[T]) -> core::slice::Iter<'_, T>);
gen_intrinsic!(slice_iter_get_unchecked: fn slice_iter_get_unchecked<'a, T>(slice: &core::slice::Iter<'a, T>, index: usize) -> &'a T);
gen_intrinsic!(slice_iter_start: fn slice_iter_start<T>(slice: &core::slice::Iter<'_, T>) -> usize);
gen_intrinsic!(slice_iter_set_start: fn slice_iter_set_start<T>(slice: &mut core::slice::Iter<'_, T>, start: usize));
gen_intrinsic!(slice_iter_end: fn slice_iter_end<T>(slice: &core::slice::Iter<'_, T>) -> usize);
gen_intrinsic!(slice_iter_set_end: fn slice_iter_set_end<T>(slice: &mut core::slice::Iter<'_, T>, end: usize));

gen_intrinsic!(slice_iter_new: fn slice_iter_mut_new<T>(slice: &mut [T]) -> core::slice::IterMut<'_, T>);
gen_intrinsic!(slice_iter_get_unchecked: fn slice_iter_mut_get_unchecked<'a, T>(slice: &core::slice::IterMut<'a, T>, index: usize) -> &'a mut T);
gen_intrinsic!(slice_iter_start: fn slice_iter_mut_start<T>(slice: &core::slice::IterMut<'_, T>) -> usize);
gen_intrinsic!(slice_iter_set_start: fn slice_iter_mut_set_start<T>(slice: &mut core::slice::IterMut<'_, T>, start: usize));
gen_intrinsic!(slice_iter_end: fn slice_iter_mut_end<T>(slice: &core::slice::IterMut<'_, T>) -> usize);
gen_intrinsic!(slice_iter_set_end: fn slice_iter_mut_set_end<T>(slice: &mut core::slice::IterMut<'_, T>, end: usize));

gen_intrinsic!(non_zero_new: fn non_zero_new<T: ZeroablePrimitive>(n: T) -> Option<NonZero<T>>);
gen_intrinsic!(non_zero_new_unchecked: fn non_zero_new_unchecked<T: ZeroablePrimitive>(n: T) -> NonZero<T>);
gen_intrinsic!(non_zero_get: fn non_zero_get<T: ZeroablePrimitive>(n: NonZero<T>) -> T);

gen_intrinsic!(min_f32: fn min_f32(a: f32, b: f32) -> f32);
gen_intrinsic!(max_f32: fn max_f32(a: f32, b: f32) -> f32);
gen_intrinsic!(round_ties_even_f32: fn round_ties_even_f32(a: f32) -> f32);
gen_intrinsic!(saturate_f32: fn saturate_f32(a: f32) -> f32);
gen_intrinsic!(floor_f32: fn floor_f32(a: f32) -> f32);
gen_intrinsic!(ceil_f32: fn ceil_f32(a: f32) -> f32);
gen_intrinsic!(clamp_f32: fn clamp_f32(a: f32, min: f32, max: f32) -> f32);
gen_intrinsic!(clamp_u32: fn clamp_u32(a: u32, min: u32, max: u32) -> u32);
gen_intrinsic!(clamp_i32: fn clamp_i32(a: i32, min: i32, max: i32) -> i32);
gen_intrinsic!(clamp_usize: fn clamp_usize(a: usize, min: usize, max: usize) -> usize);
gen_intrinsic!(clamp_isize: fn clamp_isize(a: isize, min: isize, max: isize) -> isize);
gen_intrinsic!(fract_f32: fn fract_f32(a: f32) -> f32);
gen_intrinsic!(mul_add_f32: fn mul_add_f32(a: f32, b: f32, c: f32) -> f32);
gen_intrinsic!(trunc_f32: fn trunc_f32(a: f32) -> f32);
gen_intrinsic!(sqrt_f32: fn sqrt_f32(a: f32) -> f32);
gen_intrinsic!(inverse_sqrt_f32: fn inverse_sqrt_f32(a: f32) -> f32);
gen_intrinsic!(exp_f32: fn exp_f32(a: f32) -> f32);
gen_intrinsic!(exp2_f32: fn exp2_f32(a: f32) -> f32);
gen_intrinsic!(ln_f32: fn ln_f32(a: f32) -> f32);
gen_intrinsic!(log2_f32: fn log2_f32(a: f32) -> f32);
gen_intrinsic!(powf_f32: fn powf_f32(a: f32, b: f32) -> f32);
gen_intrinsic!(step_f32: fn step_f32(edge: f32, x: f32) -> f32);
gen_intrinsic!(smoothstep_f32: fn smoothstep_f32(edge0: f32, edge1: f32, x: f32) -> f32);
gen_intrinsic!(cos_f32: fn cos_f32(a: f32) -> f32);
gen_intrinsic!(acos_f32: fn acos_f32(a: f32) -> f32);
gen_intrinsic!(cosh_f32: fn cosh_f32(a: f32) -> f32);
gen_intrinsic!(acosh_f32: fn acosh_f32(a: f32) -> f32);
gen_intrinsic!(sin_f32: fn sin_f32(a: f32) -> f32);
gen_intrinsic!(asin_f32: fn asin_f32(a: f32) -> f32);
gen_intrinsic!(sinh_f32: fn sinh_f32(a: f32) -> f32);
gen_intrinsic!(asinh_f32: fn asinh_f32(a: f32) -> f32);
gen_intrinsic!(tan_f32: fn tan_f32(a: f32) -> f32);
gen_intrinsic!(atan_f32: fn atan_f32(a: f32) -> f32);
gen_intrinsic!(tanh_f32: fn tanh_f32(a: f32) -> f32);
gen_intrinsic!(atanh_f32: fn atanh_f32(a: f32) -> f32);
gen_intrinsic!(to_radians_f32: fn to_radians_f32(a: f32) -> f32);
gen_intrinsic!(to_degrees_f32: fn to_degrees_f32(a: f32) -> f32);
