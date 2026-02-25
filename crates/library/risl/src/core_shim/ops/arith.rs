use risl_macros::gpu;

#[cfg_attr(
    rislc,
    rislc::core_shim("<u32 as core::ops::AddAssign<u32>>::add_assign")
)]
#[gpu]
pub fn add_assign_u32(val: &mut u32, other: u32) {
    *val = *val + other;
}

#[cfg_attr(
    rislc,
    rislc::core_shim("<u32 as core::ops::AddAssign<&u32>>::add_assign")
)]
#[gpu]
pub fn add_assign_u32_ref(val: &mut u32, other: &u32) {
    *val = *val + *other;
}

// #[gpu]
// pub trait AddAssign<Rhs = Self> {
//     fn add_assign(&mut self, rhs: Rhs);
// }
//
// macro_rules! add_assign_impl {
//     ($t:ty => $shim:literal) => {
//         #[gpu]
//         impl AddAssign for $t {
//             #[cfg_attr(rislc, rislc::core_shim($shim))]
//             fn add_assign(&mut self, other: $t) { *self = *self + other }
//         }
//     }
// }
//
// macro_rules! add_assign_ref_impl {
//     ($t:ty => $shim:literal) => {
//         #[gpu]
//         impl AddAssign<&$t> for $t {
//             #[cfg_attr(rislc, rislc::core_shim($shim))]
//             fn add_assign(&mut self, other: &$t) { *self = *self + *other }
//         }
//     }
// }
//
// add_assign_impl!(u32 => "<u32 as core::ops::AddAssign<u32>>::add_assign");
// add_assign_impl!(i32 => "<i32 as core::ops::AddAssign<i32>>::add_assign");
// add_assign_impl!(usize => "<usize as core::ops::AddAssign<usize>>::add_assign");
//
// add_assign_ref_impl!(u32 => "<u32 as core::ops::AddAssign<&u32>>::add_assign");
// add_assign_ref_impl!(i32 => "<i32 as core::ops::AddAssign<&i32>>::add_assign");
// add_assign_ref_impl!(usize => "<usize as core::ops::AddAssign<&usize>>::add_assign");
