use slir::rvsdg::{Connectivity, Node, NodeKind, Region, Rvsdg, StateOrigin, ValueOrigin, ValueUser, StateUser};
use slir::ty::Type;
use std::collections::BTreeMap;
use slir::rvsdg::analyse::region_stratification::RegionStratifier;
use slotmap::Key;

pub struct Renderer<'a> {
    rvsdg: &'a Rvsdg,
    inline_max_node_count: u32,
    #[allow(dead_code)]
    inline_max_nesting_level: u32,
    #[allow(dead_code)]
    no_color: bool,
}

impl<'a> Renderer<'a> {
    pub fn new(rvsdg: &'a Rvsdg, inline_max_node_count: u32, inline_max_nesting_level: u32, no_color: bool) -> Self {
        Self {
            rvsdg,
            inline_max_node_count,
            inline_max_nesting_level,
            no_color,
        }
    }

    pub fn format_node_id(&self, node: Node) -> String {
        let (index, version) = self.unpack_key_data(node.data());
        format!("Node({}v{})", index, version)
    }

    pub fn format_region_id(&self, region: Region) -> String {
        let (index, version) = self.unpack_key_data(region.data());
        format!("Region({}v{})", index, version)
    }

    fn unpack_key_data(&self, data: slotmap::KeyData) -> (u32, u32) {
        let ffi = data.as_ffi();
        let version = (ffi >> 32) as u32;
        let index = ffi as u32;
        (index, version)
    }

    pub fn format_type(&self, ty: Type) -> String {
        ty.to_string(self.rvsdg.ty())
    }

    pub fn format_type_detail(&self, ty: Type) -> String {
        let registry = self.rvsdg.ty();
        let kind = registry.kind(ty);
        match &*kind {
            slir::ty::TypeKind::Struct(s) => {
                let mut out = format!("{}:\n", ty.to_string(registry));
                for (i, field) in s.fields.iter().enumerate() {
                    out.push_str(&format!("  - field {}: {} (offset: {})\n", i, self.format_type(field.ty), field.offset));
                }
                out
            }
            slir::ty::TypeKind::Enum(e) => {
                let mut out = format!("{}:\n", ty.to_string(registry));
                for (i, variant_ty) in e.variants.iter().enumerate() {
                    out.push_str(&format!("  - variant {}: {}\n", i, self.format_type(*variant_ty)));
                }
                out
            }
            _ => format!("{}: {:?}", ty.to_string(registry), &*kind),
        }
    }

    pub fn format_value_origin(&self, origin: ValueOrigin) -> String {
        match origin {
            ValueOrigin::Argument(arg) => format!("a{}", arg),
            ValueOrigin::Output { producer, output } => {
                format!("{}e{}", self.format_node_id(producer), output)
            }
        }
    }

    pub fn format_value_user(&self, user: ValueUser) -> String {
        match user {
            ValueUser::Result(res) => format!("r{}", res),
            ValueUser::Input { consumer, input } => {
                format!("{}i{}", self.format_node_id(consumer), input)
            }
        }
    }

    pub fn render_region(&self, region: Region, indent: usize, nesting_level: u32) -> String {
        let mut output = Vec::new();
        self.write_region(&mut output, region, indent, nesting_level).unwrap();
        String::from_utf8(output).unwrap()
    }

    pub fn write_region<W: std::io::Write>(&self, writer: &mut W, region: Region, indent: usize, nesting_level: u32) -> std::io::Result<()> {
        let region_data = &self.rvsdg[region];
        let id_str = self.format_region_id(region);
        let indent_str = " ".repeat(indent);

        writeln!(writer, "{}Region({}):", indent_str, id_str)?;
        
        // Arguments
        let args: Vec<String> = region_data.value_arguments().iter().enumerate().map(|(i, arg)| {
            format!("{}a{}: {}", id_str, i, self.format_type(arg.ty))
        }).collect();
        
        let state_arg_str = match region_data.state_argument() {
            StateUser::Result => String::new(),
            StateUser::Node(_) => format!(", {}s: State", id_str),
        };

        write!(writer, "{}  Arguments: [{}", indent_str, args.join(", "))?;
        write!(writer, "{}", state_arg_str)?;
        writeln!(writer, "]")?;

        // Nodes (stratified)
        let mut strata: BTreeMap<usize, Vec<Node>> = BTreeMap::new();
        RegionStratifier::new().stratify(self.rvsdg, region, |node, stratum| {
            strata.entry(stratum).or_default().push(node);
        });

        for nodes in strata.values() {
            for &node in nodes {
                write!(writer, "{}  ", indent_str)?;
                self.write_node(writer, node)?;
                writeln!(writer)?;
                
                // Handle nested regions (Smart Inlining)
                self.write_nested_regions(writer, node, indent + 4, nesting_level)?;
            }
        }

        // Results
        let results: Vec<String> = region_data.value_results().iter().map(|res| {
            self.format_value_origin(res.origin)
        }).collect();
        
        let state_res_str = match region_data.state_result() {
            StateOrigin::Argument => format!(", {}s", id_str),
            StateOrigin::Node(_) => format!(", {}s", id_str),
        };

        write!(writer, "{}  Results: [{}", indent_str, results.join(", "))?;
        write!(writer, "{}", state_res_str)?;
        writeln!(writer, "]")?;

        Ok(())
    }

    pub fn render_nested_regions(&self, node: Node, output: &mut String, indent: usize, nesting_level: u32) {
        let mut buf = Vec::new();
        self.write_nested_regions(&mut buf, node, indent, nesting_level).unwrap();
        output.push_str(&String::from_utf8(buf).unwrap());
    }

    pub fn write_nested_regions<W: std::io::Write>(&self, writer: &mut W, node: Node, indent: usize, nesting_level: u32) -> std::io::Result<()> {
        let node_data = &self.rvsdg[node];
        
        match node_data.kind() {
            NodeKind::Switch(n) => {
                for &region in n.branches() {
                    self.write_nested_region_with_inlining(writer, region, indent, nesting_level)?;
                }
            }
            NodeKind::Loop(n) => {
                let region = n.loop_region();
                self.write_nested_region_with_inlining(writer, region, indent, nesting_level)?;
            }
            _ => {}
        };
        Ok(())
    }

    fn write_nested_region_with_inlining<W: std::io::Write>(&self, writer: &mut W, region: Region, indent: usize, nesting_level: u32) -> std::io::Result<()> {
        let region_data = &self.rvsdg[region];
        let node_count = region_data.nodes().len() as u32;
        
        if node_count <= self.inline_max_node_count && nesting_level < self.inline_max_nesting_level {
            self.write_region(writer, region, indent, nesting_level + 1)?;
        } else {
            let indent_str = " ".repeat(indent);
            writeln!(writer, "{}Region({}): {} child nodes", 
                indent_str, self.format_region_id(region), node_count)?;
        }
        Ok(())
    }

    pub fn render_node(&self, node: Node) -> String {
        let mut output = Vec::new();
        self.write_node(&mut output, node).unwrap();
        String::from_utf8(output).unwrap()
    }

    pub fn write_node<W: std::io::Write>(&self, writer: &mut W, node: Node) -> std::io::Result<()> {
        let node_data = &self.rvsdg[node];
        let id_str = self.format_node_id(node);
        
        let op_str = match node_data.kind() {
            NodeKind::Function(_) => format!("FunctionNode"),
            NodeKind::UniformBinding(_) => format!("UniformBinding"),
            NodeKind::StorageBinding(_) => format!("StorageBinding"),
            NodeKind::WorkgroupBinding(_) => format!("WorkgroupBinding"),
            NodeKind::Constant(_) => format!("Constant"),
            NodeKind::Simple(_n) => self.format_simple_node(node),
            NodeKind::Switch(_) => format!("Switch"),
            NodeKind::Loop(_) => format!("Loop"),
        };

        let inputs: Vec<String> = node_data.value_inputs().iter().map(|i| {
            self.format_value_origin(i.origin)
        }).collect();

        let outputs: Vec<String> = node_data.value_outputs().iter().enumerate().map(|(i, out)| {
            format!("{}e{} : {}", id_str, i, self.format_type(out.ty))
        }).collect();

        let state_str = if let Some(state) = node_data.state() {
            match state.origin {
                StateOrigin::Argument => format!(" (state: Arg)"),
                StateOrigin::Node(n) => format!(" (state: {})", self.format_node_id(n)),
            }
        } else {
            String::new()
        };

        write!(writer, "[{}] {}({}){} -> {}", id_str, op_str, inputs.join(", "), state_str, outputs.join(", "))
    }

    fn format_simple_node(&self, node: Node) -> String {
        let node_data = &self.rvsdg[node];
        if node_data.is_simple() {
            let simple = node_data.expect_simple();
            // In slir/src/rvsdg/rvsdg.rs, SimpleNode::intrinsic() returns &IntrinsicNode<T>
            // We can try to use Debug if it's available, or just look at the type name via metadata if possible.
            // Since we're in another crate, we might just have to stick with "Op".
            // However, we can use the Debug implementation of SimpleNode which might show the intrinsic.
            let debug_str = format!("{:?}", simple);
            if let Some(start) = debug_str.find("intrinsic: ") {
                let rest = &debug_str[start + 11..];
                if let Some(end) = rest.find(" {") {
                    return rest[..end].to_string();
                }
            }
            "Op".to_string() 
        } else {
            "Simple".to_string()
        }
    }
}
