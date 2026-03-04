use slir::rvsdg::Rvsdg;
use std::io::Write;
use crate::renderer::Renderer;
use crate::id_resolution;
use anyhow::Result;

pub fn render_node_mode<W: Write>(_rvsdg: &Rvsdg, renderer: &Renderer, writer: &mut W, node_id_str: &str) -> Result<()> {
    let node_id = id_resolution::parse_node_id(node_id_str)?;
    renderer.write_node(writer, node_id)?;
    writeln!(writer)?;
    // Also show nested regions
    let mut nested = String::new();
    renderer.render_nested_regions(node_id, &mut nested, 2, 0);
    if !nested.is_empty() {
        writeln!(writer, "  Nested Regions:")?;
        write!(writer, "{}", nested)?;
    }
    Ok(())
}
