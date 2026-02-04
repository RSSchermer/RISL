#![feature(stmt_expr_attributes)]

use risl::prelude::*;

#[shader_module]
pub mod shader {
    use risl::prelude::*;

    #[resource(group = 0, binding = 0)]
    static VALUES: Uniform<[u32; 4]>;

    #[workgroup_shared]
    static VALUE: Workgroup<u32>;

    #[compute]
    fn entry_point_simple() {
        let value = if let Some(element) = VALUES.get(1..3) {
            1
        } else {
            0
        };

        unsafe {
            *VALUE.as_mut_unchecked() = value;
        }
    }

    // #[compute]
    // fn entry_point_local_range() {
    //     let data = [10, 20, 30, 40];
    //
    //     if let Some(slice) = data.get(1..3) {
    //         let value = if let Some(element) = slice.get(1) {
    //             *element
    //         } else {
    //             0
    //         };
    //
    //         unsafe {
    //             *VALUE.as_mut_unchecked() = value;
    //         }
    //     }
    // }
    //
    // #[compute]
    // fn entry_point_global_range() {
    //     if let Some(slice) = VALUES.get(1..3) {
    //         let value = if let Some(element) = slice.get(1) {
    //             *element
    //         } else {
    //             0
    //         };
    //
    //         unsafe {
    //             *VALUE.as_mut_unchecked() = value;
    //         }
    //     }
    // }
    //
    // #[compute]
    // fn entry_point_slice_destructuring() {
    //     if let [a, tail @ ..] = VALUES.as_ref() {
    //         let b = tail.get(0).copied().unwrap_or(10);
    //
    //         unsafe {
    //             *VALUE.as_mut_unchecked() = *a + b;
    //         }
    //     }
    // }
}

fn main() {}
