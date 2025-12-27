#![feature(stmt_expr_attributes)]

use risl::shader::shader_wgsl;

const SHADER: &'static str = shader_wgsl!(simple_lib::shader);

fn main() {
    println!("{}", SHADER);
}
