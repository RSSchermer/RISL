//! Intrinsics for the RISL compiler.

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
gen_intrinsic!(slice_element_ref: fn slice_element_ref<T>(slice: &[T], index: usize) -> &T);
gen_intrinsic!(slice_element_ref: fn slice_element_mut<T>(slice: &mut [T], index: usize) -> &mut T);
gen_intrinsic!(slice_range: fn slice_range<T>(slice: &[T], start: usize, end: usize) -> &[T]);
gen_intrinsic!(slice_range: fn slice_range_mut<T>(slice: &mut [T], start: usize, end: usize) -> &mut [T]);
