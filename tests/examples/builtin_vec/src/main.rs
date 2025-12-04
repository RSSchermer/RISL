#![feature(stmt_expr_attributes)]
use risl::prelude::*;
use risl::shader::shader_wgsl;

#[shader_module]
pub mod shader {
    use risl::prelude::*;

    #[workgroup_shared]
    static VALUE_0: Workgroup<vec2_f32>;

    #[compute]
    fn main(factor: f32) {
        unsafe {
            if *VALUE_0.as_ref_unchecked() == vec2_f32(0.0, 1.0) {
                let matrix = mat2x2_f32(vec2_f32(1.0 * factor, 0.0), *VALUE_0.as_ref_unchecked());

                let v = matrix * vec2_f32(1.0, 1.0);

                *VALUE_0.as_mut_unchecked() += v;
            }
        }
    }
}

// const WGSL: &'static str = shader_wgsl!(shader);

fn main() {
    // println!("{}", WGSL);
}
