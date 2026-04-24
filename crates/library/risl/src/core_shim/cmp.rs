use risl_macros::gpu;

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::cmp::PartialEq"))]
pub trait PartialEq<Rhs = Self>
where
    Rhs: ?Sized,
{
    fn eq(&self, other: &Rhs) -> bool;

    fn ne(&self, other: &Rhs) -> bool {
        !self.eq(other)
    }
}

#[gpu]
#[cfg_attr(rislc, rislc::core_shim("core::cmp::Eq"))]
pub trait Eq: core::cmp::PartialEq {}
