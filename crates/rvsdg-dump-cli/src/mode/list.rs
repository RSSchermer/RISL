use slir::rvsdg::{Rvsdg, Connectivity};
use std::io::Write;
use crate::renderer::Renderer;
use anyhow::Result;

pub fn render_list<W: Write>(rvsdg: &Rvsdg, renderer: &Renderer, writer: &mut W) -> Result<()> {
    writeln!(writer, "Functions:")?;
    for (function, node) in rvsdg.registered_functions() {
        writeln!(writer, "  - {}: {}", function.name.as_str(), renderer.format_node_id(node))?;
    }

    writeln!(writer, "\nGlobal Bindings:")?;
    let global_region = rvsdg.global_region();
    for &node in rvsdg[global_region].nodes() {
        let node_data = &rvsdg[node];
        match node_data.kind() {
            slir::rvsdg::NodeKind::UniformBinding(_) => {
                writeln!(writer, "  - Uniform: {}", renderer.format_node_id(node))?;
            }
            slir::rvsdg::NodeKind::StorageBinding(_) => {
                writeln!(writer, "  - Storage: {}", renderer.format_node_id(node))?;
            }
            slir::rvsdg::NodeKind::WorkgroupBinding(_) => {
                writeln!(writer, "  - Workgroup: {}", renderer.format_node_id(node))?;
            }
            _ => {}
        }
    }

    writeln!(writer, "\nGlobal Constants:")?;
    for &node in rvsdg[global_region].nodes() {
        let node_data = &rvsdg[node];
        if let slir::rvsdg::NodeKind::Constant(c) = node_data.kind() {
            writeln!(writer, "  - {}: {} = {:?}", renderer.format_node_id(node), renderer.format_type(node_data.value_outputs()[0].ty), c.constant())?;
        }
    }

    writeln!(writer, "\nType Registry:")?;
    // TypeRegistry doesn't expose len, but we can't do much about it easily.
    // Let's just say it's available.
    writeln!(writer, "  - Type registry is present")?;
    Ok(())
}
