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
