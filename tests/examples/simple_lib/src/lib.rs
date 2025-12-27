use risl::prelude::*;

#[gpu]
pub fn or_3(a: u32, b: u32, c: u32) -> u32 {
    a | b | c
}

#[shader_module]
pub mod shader {
    use risl::prelude::*;

    #[resource(group = 0, binding = 0)]
    static VALUES: StorageMut<[u32]>;

    #[compute]
    fn main() {
        unsafe {
            if let Some(v) = VALUES.as_mut_unchecked().get_mut(0) {
                *v = *v + 1;
            }
        }
    }
}
