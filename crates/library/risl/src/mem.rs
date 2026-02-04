use std::marker;

use super::{gpu, intrinsic, resource};
use crate::prelude::Resource;

#[gpu]
#[cfg_attr(rislc, rislc::mem_resource_ty)]
pub struct Uniform<T> {
    _marker: marker::PhantomData<T>,
}

#[gpu]
impl<T> AsRef<T> for Uniform<T> {
    fn as_ref(&self) -> &T {
        unsafe { intrinsic::uniform_as_ref(self) }
    }
}

#[gpu]
impl<T> std::borrow::Borrow<T> for Uniform<T> {
    fn borrow(&self) -> &T {
        unsafe { intrinsic::uniform_as_ref(self) }
    }
}

#[gpu]
impl<T> std::ops::Deref for Uniform<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { intrinsic::uniform_as_ref(self) }
    }
}

impl<T> resource::seal::Sealed for Uniform<T> {}
impl<T> Resource for Uniform<T> {}

#[gpu]
#[cfg_attr(rislc, rislc::mem_resource_ty)]
pub struct Storage<T>
where
    T: ?Sized,
{
    _marker: marker::PhantomData<T>,
}

#[gpu]
impl<T> AsRef<T> for Storage<T>
where
    T: ?Sized,
{
    fn as_ref(&self) -> &T {
        unsafe { intrinsic::storage_as_ref(self) }
    }
}

#[gpu]
impl<T> std::borrow::Borrow<T> for Storage<T>
where
    T: ?Sized,
{
    fn borrow(&self) -> &T {
        unsafe { intrinsic::storage_as_ref(self) }
    }
}

#[gpu]
impl<T> std::ops::Deref for Storage<T>
where
    T: ?Sized,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { intrinsic::storage_as_ref(self) }
    }
}

impl<T> resource::seal::Sealed for Storage<T> where T: ?Sized {}
impl<T> Resource for Storage<T> where T: ?Sized {}

#[gpu]
#[cfg_attr(rislc, rislc::mem_resource_ty)]
pub struct StorageMut<T>
where
    T: ?Sized,
{
    _marker: marker::PhantomData<T>,
}

impl<T> StorageMut<T>
where
    T: ?Sized,
{
    #[gpu]
    pub unsafe fn as_ref_unchecked(&self) -> &T {
        unsafe { intrinsic::storage_mut_as_ref(self) }
    }

    #[gpu]
    pub unsafe fn as_mut_unchecked(&self) -> &mut T {
        unsafe { intrinsic::storage_mut_as_mut(self) }
    }
}

impl<T> resource::seal::Sealed for StorageMut<T> where T: ?Sized {}
impl<T> Resource for StorageMut<T> where T: ?Sized {}

#[gpu]
#[cfg_attr(rislc, rislc::mem_resource_ty)]
pub struct Workgroup<T> {
    _init: T,
}

impl<T> Workgroup<T> {
    #[gpu]
    pub unsafe fn as_ref_unchecked(&self) -> &T {
        unsafe { intrinsic::workgroup_as_ref(self) }
    }

    #[gpu]
    pub unsafe fn as_mut_unchecked(&self) -> &mut T {
        unsafe { intrinsic::workgroup_as_mut(self) }
    }
}
