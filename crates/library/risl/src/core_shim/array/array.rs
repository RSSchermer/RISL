use risl_macros::gpu;

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::array::<impl [T; N]>::as_slice"))]
pub fn array_as_slice<T, const N: usize>(v: &[T; N]) -> &[T] {
    v
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::array::<impl [T; N]>::as_mut_slice"))]
pub fn array_as_mut_slice<T, const N: usize>(v: &mut [T; N]) -> &mut [T] {
    v
}

// Core's pre-compiled implementation inlines slice iterator construction and its raw-pointer cast.
#[gpu]
#[cfg_attr(
    rislc,
    rislc::core_shim("core::array::<impl core::iter::IntoIterator for &'a [T; N]>::into_iter")
)]
pub fn array_ref_into_iter<'a, T, const N: usize>(v: &'a [T; N]) -> std::slice::Iter<'a, T>
where
    T: 'a,
{
    v.as_slice().iter()
}

#[gpu]
#[cfg_attr(
    rislc,
    rislc::core_shim("core::array::<impl core::iter::IntoIterator for &'a mut [T; N]>::into_iter")
)]
pub fn array_ref_mut_into_iter<'a, T, const N: usize>(
    v: &'a mut [T; N],
) -> std::slice::IterMut<'a, T>
where
    T: 'a,
{
    v.as_mut_slice().iter_mut()
}
