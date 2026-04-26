use risl_macros::gpu;

#[cfg_attr(rislc, rislc::core_shim("core::iter::IntoIterator"))]
#[gpu]
pub trait IntoIterator {
    type Item;
    type IntoIter;

    fn into_iter(self) -> Self::IntoIter;
}

#[cfg_attr(rislc, rislc::core_shim("core::iter::Iterator"))]
#[gpu]
pub trait Iterator {
    type Item;

    fn next(&mut self) -> Option<Self::Item>;

    #[cfg_attr(rislc, rislc::core_shim("core::iter::Iterator::size_hint"))]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}
