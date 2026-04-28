use risl_macros::gpu;

/// Shim marker to help the RISL compiler recognize the [core::iter::IntoIterator] trait as a
/// GPU-compatible trait.
#[cfg_attr(rislc, rislc::core_shim("core::iter::IntoIterator"))]
#[gpu]
pub trait IntoIteratorShimMarker {}

/// Shim marker to help the RISL compiler recognize the [core::iter::Iterator] trait as a
/// GPU-compatible trait.
#[cfg_attr(rislc, rislc::core_shim("core::iter::Iterator"))]
#[gpu]
pub trait IteratorShimMarker {}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::iter::Iterator::size_hint"))]
pub fn iterator_size_hint<I>(_: &I) -> (usize, Option<usize>)
where
    I: Iterator + ?Sized,
{
    (0, None)
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::iter::Iterator::fold"))]
pub fn iterator_fold<I, B, F>(mut iter: I, init: B, mut f: F) -> B
where
    I: Iterator,
    F: FnMut(B, I::Item) -> B,
{
    let mut accum = init;

    while let Some(x) = Iterator::next(&mut iter) {
        accum = f(accum, x);
    }

    accum
}
