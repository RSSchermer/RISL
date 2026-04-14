#![allow(non_camel_case_types)]

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use risl_macros::impl_mat_mul;

use super::gpu;

#[gpu]
#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(8))]
#[cfg_attr(rislc, rislc::primitive(vec2_f32))]
pub struct vec2_f32(pub f32, pub f32);

#[gpu]
#[derive(Clone, Copy, Eq, Debug, Default)]
#[repr(C, align(8))]
#[cfg_attr(rislc, rislc::primitive(vec2_u32))]
pub struct vec2_u32(pub u32, pub u32);

#[gpu]
#[derive(Clone, Copy, Eq, Debug, Default)]
#[repr(C, align(8))]
#[cfg_attr(rislc, rislc::primitive(vec2_i32))]
pub struct vec2_i32(pub i32, pub i32);

#[gpu]
#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(16))]
#[cfg_attr(rislc, rislc::primitive(vec3_f32))]
pub struct vec3_f32(pub f32, pub f32, pub f32);

#[gpu]
#[derive(Clone, Copy, Eq, Debug, Default)]
#[repr(C, align(16))]
#[cfg_attr(rislc, rislc::primitive(vec3_u32))]
pub struct vec3_u32(pub u32, pub u32, pub u32);

#[gpu]
#[derive(Clone, Copy, Eq, Debug, Default)]
#[repr(C, align(16))]
#[cfg_attr(rislc, rislc::primitive(vec3_i32))]
pub struct vec3_i32(pub i32, pub i32, pub i32);

#[gpu]
#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(16))]
#[cfg_attr(rislc, rislc::primitive(vec4_f32))]
pub struct vec4_f32(pub f32, pub f32, pub f32, pub f32);

#[gpu]
#[derive(Clone, Copy, Eq, Debug, Default)]
#[repr(C, align(16))]
#[cfg_attr(rislc, rislc::primitive(vec4_u32))]
pub struct vec4_u32(pub u32, pub u32, pub u32, pub u32);

#[gpu]
#[derive(Clone, Copy, Eq, Debug, Default)]
#[repr(C, align(16))]
#[cfg_attr(rislc, rislc::primitive(vec4_i32))]
pub struct vec4_i32(pub i32, pub i32, pub i32, pub i32);

macro_rules! impl_vec_partial_eq {
    ($name:ident, $el_ty_:ident, ($($el_id:tt : $el_ty:ident),*)) => {
        #[gpu]
        impl PartialEq<$name> for $name {
            fn eq(&self, other: &$name) -> bool {
                $(self.$el_id == other.$el_id) && *
            }
        }
    }
}

impl_vec_partial_eq!(vec2_f32, f32, (0: f32, 1: f32));
impl_vec_partial_eq!(vec2_u32, u32, (0: u32, 1: u32));
impl_vec_partial_eq!(vec2_i32, i32, (0: i32, 1: i32));
impl_vec_partial_eq!(vec3_f32, f32, (0: f32, 1: f32, 2: f32));
impl_vec_partial_eq!(vec3_u32, u32, (0: u32, 1: u32, 2: u32));
impl_vec_partial_eq!(vec3_i32, i32, (0: i32, 1: i32, 2: i32));
impl_vec_partial_eq!(vec4_f32, f32, (0: f32, 1: f32, 2: f32, 3: f32));
impl_vec_partial_eq!(vec4_u32, u32, (0: u32, 1: u32, 2: u32, 3: u32));
impl_vec_partial_eq!(vec4_i32, i32, (0: i32, 1: i32, 2: i32, 3: i32));

macro_rules! impl_vec_arith {
    ($name:ident, $el_ty_:ident, ($($el_id:tt : $el_ty:ident),*)) => {
        #[gpu]
        impl Add<$name> for $name {
            type Output = $name;

            #[cfg_attr(rislc, rislc::intrinsic(add))]
            fn add(self, rhs: $name) -> Self::Output {
                $name($(self.$el_id + rhs.$el_id),*)
            }
        }

        #[gpu]
        impl AddAssign<$name> for $name {
            fn add_assign(&mut self, rhs: $name) {
                *self = self.add(rhs);
            }
        }

        #[gpu]
        impl Add<$el_ty_> for $name {
            type Output = $name;

            #[cfg_attr(rislc, rislc::intrinsic(add))]
            fn add(self, rhs: $el_ty_) -> Self::Output {
                $name($(self.$el_id + rhs),*)
            }
        }

        #[gpu]
        impl AddAssign<$el_ty_> for $name {
            fn add_assign(&mut self, rhs: $el_ty_) {
                *self = self.add(rhs);
            }
        }

        #[gpu]
        impl Add<$name> for $el_ty_ {
            type Output = $name;

            #[cfg_attr(rislc, rislc::intrinsic(add))]
            fn add(self, rhs: $name) -> Self::Output {
                $name($(self + rhs.$el_id),*)
            }
        }

        #[gpu]
        impl Sub<$name> for $name {
            type Output = $name;

            #[cfg_attr(rislc, rislc::intrinsic(sub))]
            fn sub(self, rhs: $name) -> Self::Output {
                $name($(self.$el_id - rhs.$el_id),*)
            }
        }

        #[gpu]
        impl SubAssign<$name> for $name {
            fn sub_assign(&mut self, rhs: $name) {
                *self = self.sub(rhs);
            }
        }

        #[gpu]
        impl Sub<$el_ty_> for $name {
            type Output = $name;

            #[cfg_attr(rislc, rislc::intrinsic(sub))]
            fn sub(self, rhs: $el_ty_) -> Self::Output {
                $name($(self.$el_id - rhs),*)
            }
        }

        #[gpu]
        impl SubAssign<$el_ty_> for $name {
            fn sub_assign(&mut self, rhs: $el_ty_) {
                *self = self.sub(rhs);
            }
        }

        #[gpu]
        impl Sub<$name> for $el_ty_ {
            type Output = $name;

            #[cfg_attr(rislc, rislc::intrinsic(sub))]
            fn sub(self, rhs: $name) -> Self::Output {
                $name($(self - rhs.$el_id),*)
            }
        }

        #[gpu]
        impl Mul<$name> for $name {
            type Output = $name;

            #[cfg_attr(rislc, rislc::intrinsic(mul))]
            fn mul(self, rhs: $name) -> Self::Output {
                $name($(self.$el_id * rhs.$el_id),*)
            }
        }

        #[gpu]
        impl MulAssign<$name> for $name {
            fn mul_assign(&mut self, rhs: $name) {
                *self = self.mul(rhs);
            }
        }

        #[gpu]
        impl Mul<$el_ty_> for $name {
            type Output = $name;

            #[cfg_attr(rislc, rislc::intrinsic(mul))]
            fn mul(self, rhs: $el_ty_) -> Self::Output {
                $name($(self.$el_id * rhs),*)
            }
        }

        #[gpu]
        impl MulAssign<$el_ty_> for $name {
            fn mul_assign(&mut self, rhs: $el_ty_) {
                *self = self.mul(rhs);
            }
        }

        #[gpu]
        impl Mul<$name> for $el_ty_ {
            type Output = $name;

            #[cfg_attr(rislc, rislc::intrinsic(mul))]
            fn mul(self, rhs: $name) -> Self::Output {
                $name($(self * rhs.$el_id),*)
            }
        }

        #[gpu]
        impl Div<$name> for $name {
            type Output = $name;

            #[cfg_attr(rislc, rislc::intrinsic(div))]
            fn div(self, rhs: $name) -> Self::Output {
                $name($(self.$el_id / rhs.$el_id),*)
            }
        }

        #[gpu]
        impl DivAssign<$name> for $name {
            fn div_assign(&mut self, rhs: $name) {
                *self = self.div(rhs);
            }
        }

        #[gpu]
        impl Div<$el_ty_> for $name {
            type Output = $name;

            #[cfg_attr(rislc, rislc::intrinsic(div))]
            fn div(self, rhs: $el_ty_) -> Self::Output {
                $name($(self.$el_id / rhs),*)
            }
        }

        #[gpu]
        impl DivAssign<$el_ty_> for $name {
            fn div_assign(&mut self, rhs: $el_ty_) {
                *self = self.div(rhs);
            }
        }

        #[gpu]
        impl Div<$name> for $el_ty_ {
            type Output = $name;

            #[cfg_attr(rislc, rislc::intrinsic(div))]
            fn div(self, rhs: $name) -> Self::Output {
                $name($(self / rhs.$el_id),*)
            }
        }
    };
}

impl_vec_arith!(vec2_f32, f32, (0: f32, 1: f32));
impl_vec_arith!(vec2_i32, i32, (0: i32, 1: i32));
impl_vec_arith!(vec2_u32, u32, (0: u32, 1: u32));
impl_vec_arith!(vec3_f32, f32, (0: f32, 1: f32, 2: f32));
impl_vec_arith!(vec3_i32, i32, (0: i32, 1: i32, 2: i32));
impl_vec_arith!(vec3_u32, u32, (0: u32, 1: u32, 2: u32));
impl_vec_arith!(vec4_f32, f32, (0: f32, 1: f32, 2: f32, 3: f32));
impl_vec_arith!(vec4_i32, i32, (0: i32, 1: i32, 2: i32, 3: i32));
impl_vec_arith!(vec4_u32, u32, (0: u32, 1: u32, 2: u32, 3: u32));

#[gpu]
#[derive(Clone, Copy, PartialEq, Debug, Default)]
#[cfg_attr(rislc, rislc::primitive(mat2x2_f32))]
pub struct mat2x2_f32(pub vec2_f32, pub vec2_f32);

#[gpu]
#[derive(Clone, Copy, PartialEq, Debug, Default)]
#[cfg_attr(rislc, rislc::primitive(mat2x3_f32))]
pub struct mat2x3_f32(pub vec3_f32, pub vec3_f32);

#[gpu]
#[derive(Clone, Copy, PartialEq, Debug, Default)]
#[cfg_attr(rislc, rislc::primitive(mat2x4_f32))]
pub struct mat2x4_f32(pub vec4_f32, pub vec4_f32);

#[gpu]
#[derive(Clone, Copy, PartialEq, Debug, Default)]
#[cfg_attr(rislc, rislc::primitive(mat3x2_f32))]
pub struct mat3x2_f32(pub vec2_f32, pub vec2_f32, pub vec2_f32);

#[gpu]
#[derive(Clone, Copy, PartialEq, Debug, Default)]
#[cfg_attr(rislc, rislc::primitive(mat3x3_f32))]
pub struct mat3x3_f32(pub vec3_f32, pub vec3_f32, pub vec3_f32);

#[gpu]
#[derive(Clone, Copy, PartialEq, Debug, Default)]
#[cfg_attr(rislc, rislc::primitive(mat3x4_f32))]
pub struct mat3x4_f32(pub vec4_f32, pub vec4_f32, pub vec4_f32);

#[gpu]
#[derive(Clone, Copy, PartialEq, Debug, Default)]
#[cfg_attr(rislc, rislc::primitive(mat4x2_f32))]
pub struct mat4x2_f32(pub vec2_f32, pub vec2_f32, pub vec2_f32, pub vec2_f32);

#[gpu]
#[derive(Clone, Copy, PartialEq, Debug, Default)]
#[cfg_attr(rislc, rislc::primitive(mat4x3_f32))]
pub struct mat4x3_f32(pub vec3_f32, pub vec3_f32, pub vec3_f32, pub vec3_f32);

#[gpu]
#[derive(Clone, Copy, PartialEq, Debug, Default)]
#[cfg_attr(rislc, rislc::primitive(mat4x4_f32))]
pub struct mat4x4_f32(pub vec4_f32, pub vec4_f32, pub vec4_f32, pub vec4_f32);

impl_mat_mul!(mat<2, 2> * mat<2, 2>);
impl_mat_mul!(mat<2, 3> * mat<2, 2>);
impl_mat_mul!(mat<2, 4> * mat<2, 2>);

impl_mat_mul!(mat<3, 2> * mat<2, 3>);
impl_mat_mul!(mat<3, 3> * mat<2, 3>);
impl_mat_mul!(mat<3, 4> * mat<2, 3>);

impl_mat_mul!(mat<4, 2> * mat<2, 4>);
impl_mat_mul!(mat<4, 3> * mat<2, 4>);
impl_mat_mul!(mat<4, 4> * mat<2, 4>);

impl_mat_mul!(mat<2, 2> * mat<3, 2>);
impl_mat_mul!(mat<2, 3> * mat<3, 2>);
impl_mat_mul!(mat<2, 4> * mat<3, 2>);

impl_mat_mul!(mat<3, 2> * mat<3, 3>);
impl_mat_mul!(mat<3, 3> * mat<3, 3>);
impl_mat_mul!(mat<3, 4> * mat<3, 3>);

impl_mat_mul!(mat<4, 2> * mat<3, 4>);
impl_mat_mul!(mat<4, 3> * mat<3, 4>);
impl_mat_mul!(mat<4, 4> * mat<3, 4>);

impl_mat_mul!(mat<2, 2> * mat<4, 2>);
impl_mat_mul!(mat<2, 3> * mat<4, 2>);
impl_mat_mul!(mat<2, 4> * mat<4, 2>);

impl_mat_mul!(mat<3, 2> * mat<4, 3>);
impl_mat_mul!(mat<3, 3> * mat<4, 3>);
impl_mat_mul!(mat<3, 4> * mat<4, 3>);

impl_mat_mul!(mat<4, 2> * mat<4, 4>);
impl_mat_mul!(mat<4, 3> * mat<4, 4>);
impl_mat_mul!(mat<4, 4> * mat<4, 4>);

impl_mat_mul!(mat<2, 2> * vec<2>);
impl_mat_mul!(mat<2, 3> * vec<2>);
impl_mat_mul!(mat<2, 4> * vec<2>);

impl_mat_mul!(mat<3, 2> * vec<3>);
impl_mat_mul!(mat<3, 3> * vec<3>);
impl_mat_mul!(mat<3, 4> * vec<3>);

impl_mat_mul!(mat<4, 2> * vec<4>);
impl_mat_mul!(mat<4, 3> * vec<4>);
impl_mat_mul!(mat<4, 4> * vec<4>);

impl_mat_mul!(vec<2> * mat<2, 2>);
impl_mat_mul!(vec<2> * mat<3, 2>);
impl_mat_mul!(vec<2> * mat<4, 2>);

impl_mat_mul!(vec<3> * mat<2, 3>);
impl_mat_mul!(vec<3> * mat<3, 3>);
impl_mat_mul!(vec<3> * mat<4, 3>);

impl_mat_mul!(vec<4> * mat<2, 4>);
impl_mat_mul!(vec<4> * mat<3, 4>);
impl_mat_mul!(vec<4> * mat<4, 4>);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mat_mat_mul() {
        let a = mat2x2_f32(vec2_f32(1.0, 2.0), vec2_f32(3.0, 4.0));
        let b = mat3x2_f32(vec2_f32(1.0, 2.0), vec2_f32(3.0, 4.0), vec2_f32(5.0, 6.0));

        let product = a * b;

        assert_eq!(
            product,
            mat3x2_f32(
                vec2_f32(7.0, 10.0),
                vec2_f32(15.0, 22.0),
                vec2_f32(23.0, 34.0)
            )
        );
    }

    #[test]
    fn test_mat_vec_mul() {
        let a = mat2x3_f32(vec3_f32(1.0, 2.0, 3.0), vec3_f32(4.0, 5.0, 6.0));
        let b = vec2_f32(1.0, 2.0);

        let product = a * b;

        assert_eq!(product, vec3_f32(9.0, 12.0, 15.0));
    }

    #[test]
    fn test_vec_mat_mul() {
        let a = vec2_f32(1.0, 2.0);
        let b = mat3x2_f32(vec2_f32(1.0, 2.0), vec2_f32(3.0, 4.0), vec2_f32(5.0, 6.0));

        let product = a * b;

        assert_eq!(product, vec3_f32(5.0, 11.0, 17.0));
    }
}
