use std::io::Write;
use crate::renderer::Renderer;
use anyhow::{Result, Context};

pub fn render_type_inspect<W: Write>(renderer: &Renderer, writer: &mut W, ty_id_str: &str) -> Result<()> {
    // Resolve type ID
    // types are just indices in TypeRegistry
    let ty_id: u32 = if ty_id_str.starts_with("struct(") || ty_id_str.starts_with("enum(") {
        let start = ty_id_str.find('(').unwrap() + 1;
        let end = ty_id_str.find(')').unwrap();
        ty_id_str[start..end].parse().context("Failed to parse type ID")?
    } else {
        ty_id_str.parse().context("Failed to parse type ID")?
    };

    let ty = slir::ty::Type::from_registration_id(ty_id as usize);
    writeln!(writer, "{}", renderer.format_type_detail(ty))?;
    Ok(())
}
