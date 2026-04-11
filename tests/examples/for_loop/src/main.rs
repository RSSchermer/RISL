#![feature(stmt_expr_attributes)]
use risl::prelude::*;
use risl::shader::{shader_module_interface, shader_wgsl};
use risl::smi::ShaderModuleInterface;

#[shader_module]
pub mod shader {
    use risl::prelude::*;

    #[resource(group = 0, binding = 0)]
    static VALUES: Storage<[u32]>;

    #[resource(group = 0, binding = 1)]
    static RESULT: StorageMut<u32>;

    #[compute]
    fn main(factor: f32) {
        let mut sum = 0u32;

        for value in &*VALUES {
            sum += value;
        }

        unsafe {
            *RESULT.as_mut_unchecked() = sum;
        }
    }
}

const SHADER: &str = shader_wgsl!(shader);

const SMI: ShaderModuleInterface = shader_module_interface!(shader);

fn main() {
    println!("{}", SHADER);
    println!("{:#?}", &SMI);
}
