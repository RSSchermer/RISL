use risl_macros::gpu;

#[gpu]
pub trait SliceIndex<T>
where
    T: ?Sized,
{
    type Output: ?Sized;

    fn get(self, slice: &T) -> Option<&Self::Output>;

    fn get_mut(self, slice: &mut T) -> Option<&mut Self::Output>;
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
impl<T> SliceIndex<[T]> for usize {
    type Output = [T];

    #[cfg_attr(
        rislc,
        rislc::core_shim("<usize as core::slice::SliceIndex<[T]>>::get")
    )]
    fn get(self, slice: &[T]) -> Option<&Self::Output> {
        todo!()
    }

    #[cfg_attr(
        rislc,
        rislc::core_shim("<usize as core::slice::SliceIndex<[T]>>::get_mut")
    )]
    fn get_mut(self, slice: &mut [T]) -> Option<&mut Self::Output> {
        todo!()
    }
}
