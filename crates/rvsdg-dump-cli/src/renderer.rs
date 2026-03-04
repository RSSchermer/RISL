use std::collections::BTreeMap;

use slir::rvsdg::analyse::region_stratification::RegionStratifier;
use slir::rvsdg::{
    Connectivity, Node, NodeKind, Region, Rvsdg, StateOrigin, ValueOrigin, ValueUser,
};
use slir::ty::{Type, TypeKind};
use slotmap::{Key, KeyData};

pub struct Renderer<'a> {
    rvsdg: &'a Rvsdg,
    inline_max_node_count: u32,
    #[allow(dead_code)]
    inline_max_nesting_level: u32,
    #[allow(dead_code)]
    no_color: bool,
}

impl<'a> Renderer<'a> {
    pub fn new(
        rvsdg: &'a Rvsdg,
        inline_max_node_count: u32,
        inline_max_nesting_level: u32,
        no_color: bool,
    ) -> Self {
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

    fn unpack_key_data(&self, data: KeyData) -> (u32, u32) {
        let ffi = data.as_ffi();
        let version = (ffi >> 32) as u32;
        let index = ffi as u32;
        (index, version)
    }

    pub fn format_type(&self, ty: Type) -> String {
        let registry = self.rvsdg.ty();
        let kind = registry.kind(ty);
        match &*kind {
            TypeKind::Struct(_) => {
                format!("struct({})", ty.registration_id().unwrap())
            }
            TypeKind::Enum(_) => {
                format!("enum({})", ty.registration_id().unwrap())
            }
            _ => ty.to_string(registry),
        }
    }

    pub fn format_type_detail(&self, ty: Type) -> String {
        let registry = self.rvsdg.ty();
        let kind = registry.kind(ty);
        match &*kind {
            TypeKind::Struct(s) => {
                let mut out = format!("{}:\n", self.format_type(ty));
                for (i, field) in s.fields.iter().enumerate() {
                    out.push_str(&format!(
                        "  - field {}: {} (offset: {})\n",
                        i,
                        self.format_type(field.ty),
                        field.offset
                    ));
                }
                out
            }
            TypeKind::Enum(e) => {
                let mut out = format!("{}:\n", self.format_type(ty));
                for (i, variant_ty) in e.variants.iter().enumerate() {
                    out.push_str(&format!(
                        "  - variant {}: {}\n",
                        i,
                        self.format_type(*variant_ty)
                    ));
                }
                out
            }
            _ => format!("{}: {:?}", self.format_type(ty), &*kind),
        }
    }

    pub fn format_value_origin(&self, origin: ValueOrigin, region: Option<Region>) -> String {
        match origin {
            ValueOrigin::Argument(arg) => {
                if let Some(r) = region {
                    format!("{}a{}", self.format_region_id(r), arg)
                } else {
                    format!("a{}", arg)
                }
            }
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

    pub fn format_function_signature(&self, node: Node) -> String {
        let node_data = &self.rvsdg[node];
        let function_node = node_data.expect_function();
        let body_region = function_node.body_region();
        let body_data = &self.rvsdg[body_region];

        let args = body_data.value_arguments();
        let results = body_data.value_results();
        let dep_count = function_node.dependencies().len();

        let mut sig = String::from("(");
        for (i, arg) in args.iter().enumerate().skip(dep_count) {
            if i > dep_count {
                sig.push_str(", ");
            }
            sig.push_str(&format!(
                "arg{}: {}",
                i - dep_count,
                self.format_type(arg.ty)
            ));
        }
        sig.push_str(") -> ");

        if results.len() == 1 {
            sig.push_str(&self.format_type(results[0].ty));
        } else {
            sig.push('(');
            for (i, res) in results.iter().enumerate() {
                if i > 0 {
                    sig.push_str(", ");
                }
                sig.push_str(&self.format_type(res.ty));
            }
            sig.push(')');
        }

        sig
    }

    pub fn render_region(&self, region: Region, indent: usize, nesting_level: u32) -> String {
        let mut output = Vec::new();
        self.write_region(&mut output, region, indent, nesting_level)
            .unwrap();
        String::from_utf8(output).unwrap()
    }

    pub fn write_region<W: std::io::Write>(
        &self,
        writer: &mut W,
        region: Region,
        indent: usize,
        nesting_level: u32,
    ) -> std::io::Result<()> {
        let region_data = &self.rvsdg[region];
        let id_str = self.format_region_id(region);
        let indent_str = " ".repeat(indent);

        writeln!(writer, "{}Region({}):", indent_str, id_str)?;

        // Arguments
        let args: Vec<String> = region_data
            .value_arguments()
            .iter()
            .enumerate()
            .map(|(i, arg)| format!("{}a{}: {}", id_str, i, self.format_type(arg.ty)))
            .collect();

        write!(writer, "{}  Arguments: [{}", indent_str, args.join(", "))?;
        write!(writer, ", {}s: State", id_str)?;
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
        let results: Vec<String> = region_data
            .value_results()
            .iter()
            .map(|res| self.format_value_origin(res.origin, Some(region)))
            .collect();

        let state_res_str = match region_data.state_result() {
            StateOrigin::Argument => format!(", {}s", id_str),
            StateOrigin::Node(_) => format!(", {}s", id_str),
        };

        write!(writer, "{}  Results: [{}", indent_str, results.join(", "))?;
        write!(writer, "{}", state_res_str)?;
        writeln!(writer, "]")?;

        Ok(())
    }

    pub fn render_nested_regions(
        &self,
        node: Node,
        output: &mut String,
        indent: usize,
        nesting_level: u32,
    ) {
        let mut buf = Vec::new();
        self.write_nested_regions(&mut buf, node, indent, nesting_level)
            .unwrap();
        output.push_str(&String::from_utf8(buf).unwrap());
    }

    pub fn write_nested_regions<W: std::io::Write>(
        &self,
        writer: &mut W,
        node: Node,
        indent: usize,
        nesting_level: u32,
    ) -> std::io::Result<()> {
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

    fn write_nested_region_with_inlining<W: std::io::Write>(
        &self,
        writer: &mut W,
        region: Region,
        indent: usize,
        nesting_level: u32,
    ) -> std::io::Result<()> {
        let region_data = &self.rvsdg[region];
        let node_count = region_data.nodes().len() as u32;

        if node_count <= self.inline_max_node_count && nesting_level < self.inline_max_nesting_level
        {
            self.write_region(writer, region, indent, nesting_level + 1)?;
        } else {
            let indent_str = " ".repeat(indent);
            writeln!(
                writer,
                "{}Region({}): {} child nodes",
                indent_str,
                self.format_region_id(region),
                node_count
            )?;
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

        let inputs: Vec<String> = node_data
            .value_inputs()
            .iter()
            .map(|i| self.format_value_origin(i.origin, node_data.region().into()))
            .collect();

        let outputs: Vec<String> = node_data
            .value_outputs()
            .iter()
            .enumerate()
            .map(|(i, out)| format!("{}e{} : {}", id_str, i, self.format_type(out.ty)))
            .collect();

        let mut state_outputs = Vec::new();
        let state_str = if let Some(state) = node_data.state() {
            state_outputs.push(format!("{}s : State", id_str));
            match state.origin {
                StateOrigin::Argument => format!(" (state: Arg)"),
                StateOrigin::Node(n) => format!(" (state: {})", self.format_node_id(n)),
            }
        } else {
            String::new()
        };

        let mut all_outputs = outputs;
        all_outputs.extend(state_outputs);

        write!(
            writer,
            "[{}] {}({}){} -> {}",
            id_str,
            op_str,
            inputs.join(", "),
            state_str,
            all_outputs.join(", ")
        )
    }

    fn format_simple_node(&self, node: Node) -> String {
        let node_data = &self.rvsdg[node];
        if node_data.is_simple() {
            let simple = node_data.expect_simple();
            let debug_str = format!("{:?}", simple);
            // Example: IntrinsicNode { intrinsic: OpLoad, ... }
            // or IntrinsicNode { intrinsic: OpFieldPtr { field_index: 0 }, ... }
            if let Some(start) = debug_str.find("intrinsic: ") {
                let rest = &debug_str[start + 11..];
                if let Some(end) = rest.find(", value_inputs") {
                    let intrinsic_debug = rest[..end].to_string();
                    // Clean up: OpFieldPtr { field_index: 0 } -> OpFieldPtr(field_index: 0)
                    if let Some(brace_start) = intrinsic_debug.find(" { ") {
                        if let Some(brace_end) = intrinsic_debug.rfind(" }") {
                            let fields = &intrinsic_debug[brace_start + 3..brace_end];
                            let name = &intrinsic_debug[..brace_start];
                            return format!("{}({})", name, fields);
                        }
                    }
                    return intrinsic_debug;
                }
            }
            "Op".to_string()
        } else {
            "Simple".to_string()
        }
    }
}
