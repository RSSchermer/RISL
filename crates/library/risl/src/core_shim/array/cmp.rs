use risl_macros::gpu;

#[gpu]
impl<T, U, const N: usize> crate::core_shim::cmp::eq::PartialEq<[U; N]> for [T; N]
where
    T: PartialEq<U>,
{
    #[cfg_attr(
        rislc,
        rislc::core_shim(
            "core::array::equality::<impl core::cmp::PartialEq<[U; N]> for [T; N]>::eq"
        )
    )]
    fn eq(&self, other: &[U; N]) -> bool {
        self.as_slice() == other.as_slice()
    }

    #[cfg_attr(
        rislc,
        rislc::core_shim(
            "core::array::equality::<impl core::cmp::PartialEq<[U; N]> for [T; N]>::ne"
        )
    )]
    fn ne(&self, other: &[U; N]) -> bool {
        self.as_slice() != other.as_slice()
    }
}

#[gpu]
impl<T, U, const N: usize> crate::core_shim::cmp::eq::PartialEq<[U]> for [T; N]
where
    T: PartialEq<U>,
{
    #[cfg_attr(
        rislc,
        rislc::core_shim("core::array::equality::<impl core::cmp::PartialEq<[U]> for [T; N]>::eq")
    )]
    fn eq(&self, other: &[U]) -> bool {
        self.as_slice() == other
    }

    #[cfg_attr(
        rislc,
        rislc::core_shim("core::array::equality::<impl core::cmp::PartialEq<[U]> for [T; N]>::ne")
    )]
    fn ne(&self, other: &[U]) -> bool {
        self.as_slice() != other
    }
}

#[gpu]
impl<T, U, const N: usize> crate::core_shim::cmp::eq::PartialEq<[U; N]> for [T]
where
    T: PartialEq<U>,
{
    #[cfg_attr(
        rislc,
        rislc::core_shim("core::array::equality::<impl core::cmp::PartialEq<[U; N]> for [T]>::eq")
    )]
    fn eq(&self, other: &[U; N]) -> bool {
        self == other.as_slice()
    }

    #[cfg_attr(
        rislc,
        rislc::core_shim("core::array::equality::<impl core::cmp::PartialEq<[U; N]> for [T]>::ne")
    )]
    fn ne(&self, other: &[U; N]) -> bool {
        self != other.as_slice()
    }
}

#[gpu]
impl<T, U, const N: usize> crate::core_shim::cmp::eq::PartialEq<&[U]> for [T; N]
where
    T: PartialEq<U>,
{
    #[cfg_attr(
        rislc,
        rislc::core_shim(
            "core::array::equality::<impl core::cmp::PartialEq<&[U]> for [T; N]>::eq"
        )
    )]
    fn eq(&self, other: &&[U]) -> bool {
        self.as_slice() == *other
    }

    #[cfg_attr(
        rislc,
        rislc::core_shim(
            "core::array::equality::<impl core::cmp::PartialEq<&[U]> for [T; N]>::ne"
        )
    )]
    fn ne(&self, other: &&[U]) -> bool {
        self.as_slice() != *other
    }
}

#[gpu]
impl<T, U, const N: usize> crate::core_shim::cmp::eq::PartialEq<[U; N]> for &[T]
where
    T: PartialEq<U>,
{
    #[cfg_attr(
        rislc,
        rislc::core_shim(
            "core::array::equality::<impl core::cmp::PartialEq<[U; N]> for &[T]>::eq"
        )
    )]
    fn eq(&self, other: &[U; N]) -> bool {
        *self == other.as_slice()
    }

    #[cfg_attr(
        rislc,
        rislc::core_shim(
            "core::array::equality::<impl core::cmp::PartialEq<[U; N]> for &[T]>::ne"
        )
    )]
    fn ne(&self, other: &[U; N]) -> bool {
        *self != other.as_slice()
    }
}

#[gpu]
impl<T, U, const N: usize> crate::core_shim::cmp::eq::PartialEq<&mut [U]> for [T; N]
where
    T: PartialEq<U>,
{
    #[cfg_attr(
        rislc,
        rislc::core_shim(
            "core::array::equality::<impl core::cmp::PartialEq<&mut [U]> for [T; N]>::eq"
        )
    )]
    fn eq(&self, other: &&mut [U]) -> bool {
        self.as_slice() == *other
    }

    #[cfg_attr(
        rislc,
        rislc::core_shim(
            "core::array::equality::<impl core::cmp::PartialEq<&mut [U]> for [T; N]>::ne"
        )
    )]
    fn ne(&self, other: &&mut [U]) -> bool {
        self.as_slice() != *other
    }
}

#[gpu]
impl<T, U, const N: usize> crate::core_shim::cmp::eq::PartialEq<[U; N]> for &mut [T]
where
    T: PartialEq<U>,
{
    #[cfg_attr(
        rislc,
        rislc::core_shim(
            "core::array::equality::<impl core::cmp::PartialEq<[U; N]> for &mut [T]>::eq"
        )
    )]
    fn eq(&self, other: &[U; N]) -> bool {
        *self == other.as_slice()
    }

    #[cfg_attr(
        rislc,
        rislc::core_shim(
            "core::array::equality::<impl core::cmp::PartialEq<[U; N]> for &mut [T]>::ne"
        )
    )]
    fn ne(&self, other: &[U; N]) -> bool {
        *self != other.as_slice()
    }
}

#[gpu]
impl<T: Eq, const N: usize> crate::core_shim::cmp::eq::Eq for [T; N] {}
