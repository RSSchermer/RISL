#![feature(stmt_expr_attributes)]
use risl::prelude::*;
use risl::shader::{shader_module_interface, shader_wgsl};
use risl::smi::ShaderModuleInterface;

#[shader_module]
pub mod shader {
    use risl::prelude::*;

    #[resource(group = 0, binding = 0)]
    static VALUES: Storage<[u32]>;

    #[compute]
    fn main(factor: f32) {
        let mut iter = VALUES.iter();
        let mut sum = 0u32;

        while let Some(value) = iter.next() {
            sum += value;
        }
    }
}

const SHADER: &str = shader_wgsl!(shader);

const SMI: ShaderModuleInterface = shader_module_interface!(shader);

fn main() {
    println!("{}", SHADER);
    println!("{:#?}", &SMI);
}
