use slir::rvsdg::Rvsdg;
use std::io::Write;
use crate::renderer::Renderer;
use crate::id_resolution;
use anyhow::Result;

pub fn render_function<W: Write>(rvsdg: &Rvsdg, renderer: &Renderer, writer: &mut W, func_name: &str) -> Result<()> {
    let mut found = false;
    for (function, node) in rvsdg.registered_functions() {
        if function.name.as_str() == func_name {
            let node_data = &rvsdg[node];
            let f = node_data.expect_function();
            writeln!(writer, "Function: {}", func_name)?;
            renderer.write_region(writer, f.body_region(), 0, 0)?;
            found = true;
            break;
        }
    }
    if !found {
        // Try as ID
        if let Ok(node_id) = id_resolution::parse_node_id(func_name) {
            let node_data = &rvsdg[node_id];
            if node_data.is_function() {
                let f = node_data.expect_function();
                if let Some(function) = rvsdg.registered_functions().find(|(_, n)| *n == node_id).map(|(f, _)| f) {
                    writeln!(writer, "Function: {}", function.name.as_str())?;
                }
                renderer.write_region(writer, f.body_region(), 0, 0)?;
                found = true;
            }
        }
    }
    if !found {
        anyhow::bail!("Function '{}' not found", func_name);
    }
    Ok(())
}
