use risl_macros::gpu;

#[gpu]
impl<T, U> crate::core_shim::cmp::eq::PartialEq<[U]> for [T]
where
    T: PartialEq<U>,
{
    #[cfg_attr(
        rislc,
        rislc::core_shim("core::slice::cmp::<impl core::cmp::PartialEq<[U]> for [T]>::eq")
    )]
    fn eq(&self, other: &[U]) -> bool {
        let len = self.len();

        if len == other.len() {
            let mut idx = 0;

            while idx < len {
                unsafe {
                    let a = self.get_unchecked(idx);
                    let b = other.get_unchecked(idx);

                    if PartialEq::ne(a, b) {
                        return false;
                    }
                }

                idx += 1;
            }

            true
        } else {
            false
        }
    }

    #[cfg_attr(
        rislc,
        rislc::core_shim("core::slice::cmp::<impl core::cmp::PartialEq<[U]> for [T]>::ne")
    )]
    fn ne(&self, other: &[U]) -> bool {
        !crate::core_shim::cmp::eq::PartialEq::eq(self, other)
    }
}

#[gpu]
impl<T: Eq> crate::core_shim::cmp::eq::Eq for [T] {}
