use slir::rvsdg::Rvsdg;
use std::io::Write;
use crate::renderer::Renderer;
use crate::id_resolution;
use anyhow::Result;

pub fn render_region_mode<W: Write>(_rvsdg: &Rvsdg, renderer: &Renderer, writer: &mut W, region_id_str: &str) -> Result<()> {
    let region_id = id_resolution::parse_region_id(region_id_str)?;
    renderer.write_region(writer, region_id, 0, 0)?;
    Ok(())
}
