use risl_macros::gpu;

use crate::intrinsic;

#[gpu]
pub trait F32Ext: Sized {
    fn inverse_sqrt(self) -> Self;
    fn saturate(self) -> Self;
}

#[gpu]
impl F32Ext for f32 {
    fn inverse_sqrt(self) -> Self {
        #[cfg(rislc)]
        unsafe {
            intrinsic::inverse_sqrt_f32(self)
        }

        #[cfg(not(rislc))]
        {
            1.0 / self.sqrt()
        }
    }

    fn saturate(self) -> Self {
        #[cfg(rislc)]
        unsafe {
            intrinsic::saturate_f32(self)
        }

        #[cfg(not(rislc))]
        {
            self.clamp(0.0, 1.0)
        }
    }
}
