use slir::rvsdg::{Rvsdg, Connectivity};
use std::io::Write;
use crate::renderer::Renderer;
use crate::id_resolution;
use anyhow::Result;

pub fn render_trace_value<W: Write>(rvsdg: &Rvsdg, renderer: &Renderer, writer: &mut W, value_id_str: &str) -> Result<()> {
    let parsed = id_resolution::parse_value_id(value_id_str)?;
    writeln!(writer, "Trace for: {}", value_id_str)?;
    
    match parsed {
        id_resolution::ParsedValueId::NodeOutput(node, output) => {
            writeln!(writer, "  Forward Trace (Consumers):")?;
            // Find consumers in the same region
            let node_data = &rvsdg[node];
            let region = node_data.region();
            for &consumer_node in rvsdg[region].nodes() {
                let consumer_data = &rvsdg[consumer_node];
                if let Some((input_idx, _input)) = consumer_data.value_inputs().iter().enumerate().find(|(_, i)| i.origin == (slir::rvsdg::ValueOrigin::Output { producer: node, output })) {
                    writeln!(writer, "    - Consumer: {}i{} ({})", 
                        renderer.format_node_id(consumer_node), 
                        input_idx, 
                        renderer.render_node(consumer_node))?;
                }
            }
            // Check region results
            for (res_idx, res) in rvsdg[region].value_results().iter().enumerate() {
                if res.origin == (slir::rvsdg::ValueOrigin::Output { producer: node, output }) {
                    writeln!(writer, "    - Consumer: Region Result r{}", res_idx)?;
                }
            }
        }
        id_resolution::ParsedValueId::NodeInput(node, input_idx) => {
            writeln!(writer, "  Backward Trace (Producer):")?;
            let node_data = &rvsdg[node];
            let input = &node_data.value_inputs()[input_idx as usize];
            writeln!(writer, "    - Origin: {}", renderer.format_value_origin(input.origin))?;
            match input.origin {
                slir::rvsdg::ValueOrigin::Output { producer, .. } => {
                    writeln!(writer, "      Producer: {}", renderer.render_node(producer))?;
                }
                slir::rvsdg::ValueOrigin::Argument(arg_idx) => {
                    writeln!(writer, "      Region Argument: a{}", arg_idx)?;
                }
            }
        }
        id_resolution::ParsedValueId::RegionArgument(region, arg_idx) => {
            writeln!(writer, "  Forward Trace (Consumers):")?;
            for &consumer_node in rvsdg[region].nodes() {
                let consumer_data = &rvsdg[consumer_node];
                for (input_idx, input) in consumer_data.value_inputs().iter().enumerate() {
                    if input.origin == (slir::rvsdg::ValueOrigin::Argument(arg_idx)) {
                        writeln!(writer, "    - Consumer: {}i{} ({})", 
                            renderer.format_node_id(consumer_node), 
                            input_idx, 
                            renderer.render_node(consumer_node))?;
                    }
                }
            }
            // Check region results
            for (res_idx, res) in rvsdg[region].value_results().iter().enumerate() {
                if res.origin == (slir::rvsdg::ValueOrigin::Argument(arg_idx)) {
                    writeln!(writer, "    - Consumer: Region Result r{}", res_idx)?;
                }
            }
        }
        id_resolution::ParsedValueId::RegionResult(region, res_idx) => {
            writeln!(writer, "  Backward Trace (Producer):")?;
            let res = &rvsdg[region].value_results()[res_idx as usize];
            writeln!(writer, "    - Origin: {}", renderer.format_value_origin(res.origin))?;
            match res.origin {
                slir::rvsdg::ValueOrigin::Output { producer, .. } => {
                    writeln!(writer, "      Producer: {}", renderer.render_node(producer))?;
                }
                slir::rvsdg::ValueOrigin::Argument(arg_idx) => {
                    writeln!(writer, "      Region Argument: a{}", arg_idx)?;
                }
            }
        }
    }
    Ok(())
}
