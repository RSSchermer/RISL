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
