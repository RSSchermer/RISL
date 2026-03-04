use slir::rvsdg::{Rvsdg, Connectivity};
use std::io::Write;
use crate::renderer::Renderer;
use crate::id_resolution;
use anyhow::Result;

pub fn render_trace_state<W: Write>(rvsdg: &Rvsdg, renderer: &Renderer, writer: &mut W, region_id_str: &str) -> Result<()> {
    let region_id = id_resolution::parse_region_id(region_id_str)?;
    let region_data = &rvsdg[region_id];
    writeln!(writer, "State Trace for {}:", region_id_str)?;
    
    writeln!(writer, "  [State Arg]")?;
    
    let mut current_origin = slir::rvsdg::StateOrigin::Argument;
    
    loop {
        let mut found_next = false;
        for &node in region_data.nodes() {
            let node_data = &rvsdg[node];
            if let Some(state) = node_data.state() {
                if state.origin == current_origin {
                    writeln!(writer, "    -> {} ({})", renderer.format_node_id(node), renderer.render_node(node))?;
                    current_origin = slir::rvsdg::StateOrigin::Node(node);
                    found_next = true;
                    break;
                }
            }
        }
        
        if !found_next {
            // Check if it reaches state_result
            if *region_data.state_result() == current_origin {
                writeln!(writer, "    -> [State Result]")?;
            } else {
                writeln!(writer, "    !! State chain break detected !!")?;
            }
            break;
        }
        
        // To prevent infinite loop in malformed graphs
        if current_origin == slir::rvsdg::StateOrigin::Argument { break; }
    }
    Ok(())
}
