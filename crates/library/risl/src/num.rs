use risl_macros::gpu;

use crate::intrinsic;

#[gpu]
pub trait F32Ext: Sized {
    fn inverse_sqrt(self) -> Self;
    fn saturate(self) -> Self;
    fn step(self, edge: f32) -> f32;
    fn smoothstep(self, edge0: f32, edge1: f32) -> f32;
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

    fn step(self, edge: f32) -> f32 {
        #[cfg(rislc)]
        unsafe {
            intrinsic::step_f32(edge, self)
        }

        #[cfg(not(rislc))]
        {
            if self < edge { 0.0 } else { 1.0 }
        }
    }

    fn smoothstep(self, edge0: f32, edge1: f32) -> f32 {
        #[cfg(rislc)]
        unsafe {
            intrinsic::smoothstep_f32(edge0, edge1, self)
        }

        #[cfg(not(rislc))]
        {
            let t = ((self - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);

            t * t * (3.0 - 2.0 * t)
        }
    }
}
