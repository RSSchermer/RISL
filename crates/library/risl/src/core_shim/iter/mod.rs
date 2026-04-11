use risl_macros::gpu;

#[cfg_attr(rislc, rislc::core_shim("core::iter::IntoIterator"))]
#[gpu]
pub trait IntoIterator {
    type Item;
    type IntoIter;

    fn into_iter(self) -> Self::IntoIter;
}
