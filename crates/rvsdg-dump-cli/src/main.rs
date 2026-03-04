mod id_resolution;
mod renderer;
mod mode;

use std::path::PathBuf;
use clap::Parser;
use anyhow::{Context, Result};
use slir::rvsdg::Rvsdg;
use std::fs::File;
use std::io::{BufReader, Write};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the RVSDG dump file
    path: PathBuf,

    /// Mode: Discovery (high-level inventory)
    #[arg(long)]
    list: bool,

    /// Mode: Function Analysis (analyze top-level logic of a specific function)
    #[arg(long, value_name = "NAME|ID")]
    function: Option<String>,

    /// Mode: Region Inspection (zoom in on a specific nested region)
    #[arg(long, value_name = "ID")]
    region: Option<String>,

    /// Mode: Value Tracing (trace provenance and usage of a specific value)
    #[arg(long, value_name = "ID")]
    trace_value: Option<String>,

    /// Mode: State Tracing (visualize linear chain of side-effecting operations)
    #[arg(long, value_name = "REGION_ID")]
    trace_state: Option<String>,

    /// Mode: Type Inspection (inspect declaration of a specific registered type)
    #[arg(long, value_name = "ID")]
    #[clap(name = "type")] // Rename it to "type" for CLI usage if needed, but it's --type anyway.
    type_inspect: Option<String>,

    /// Mode: Node Inspection (inspect a single node in isolation)
    #[arg(long, value_name = "ID")]
    node: Option<String>,

    /// Smart Inlining: Maximum number of nodes a nested region may have to be inlined
    #[arg(long, default_value_t = 5)]
    inline_max_node_count: u32,

    /// Smart Inlining: Maximum nesting level for inlining
    #[arg(long, default_value_t = 1)]
    inline_max_nesting_level: u32,

    /// Disable ANSI-colored output
    #[arg(long)]
    no_color: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Load RVSDG from file
    let rvsdg = load_rvsdg(&args.path)?;

    let renderer = renderer::Renderer::new(
        &rvsdg,
        args.inline_max_node_count,
        args.inline_max_nesting_level,
        args.no_color,
    );

    let mut stdout = std::io::stdout();

    if args.list {
        mode::render_list(&rvsdg, &renderer, &mut stdout)?;
    } else if let Some(ref func_name) = args.function {
        mode::render_function(&rvsdg, &renderer, &mut stdout, func_name)?;
    } else if let Some(ref region_id_str) = args.region {
        mode::render_region_mode(&rvsdg, &renderer, &mut stdout, region_id_str)?;
    } else if let Some(ref ty_id_str) = args.type_inspect {
        mode::render_type_inspect(&renderer, &mut stdout, ty_id_str)?;
    } else if let Some(ref value_id_str) = args.trace_value {
        mode::render_trace_value(&rvsdg, &renderer, &mut stdout, value_id_str)?;
    } else if let Some(ref region_id_str) = args.trace_state {
        mode::render_trace_state(&rvsdg, &renderer, &mut stdout, region_id_str)?;
    } else if let Some(ref node_id_str) = args.node {
        mode::render_node_mode(&rvsdg, &renderer, &mut stdout, node_id_str)?;
    } else {
        writeln!(stdout, "No mode selected. Use --help for usage information.")?;
    }

    Ok(())
}

fn load_rvsdg(path: &PathBuf) -> Result<Rvsdg> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open RVSDG dump file: {:?}", path))?;
    let mut reader = BufReader::new(file);

    // Using bincode 2 with serde.
    let config = bincode::config::standard();
    let rvsdg: Rvsdg = bincode::serde::decode_from_reader(&mut reader, config)
        .with_context(|| format!("Failed to deserialize RVSDG from {:?}", path))?;

    Ok(rvsdg)
}
