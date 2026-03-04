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

    pub fn format_value_user(&self, user: ValueUser, _region: Option<Region>) -> String {
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

    pub fn write_region<W: std::io::Write>(
        &self,
        writer: &mut W,
        region: Region,
        header_name: &str,
        indent: usize,
        nesting_level: u32,
    ) -> std::io::Result<()> {
        let region_data = &self.rvsdg[region];
        let id_str = self.format_region_id(region);
        let indent_str = " ".repeat(indent);

        writeln!(writer, "{}{} ({}):", indent_str, header_name, id_str)?;

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
                self.write_node(writer, node, indent + 4, nesting_level)?;
                writeln!(writer)?;
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

    fn write_nested_region_with_inlining<W: std::io::Write>(
        &self,
        writer: &mut W,
        region: Region,
        header_name: &str,
        indent: usize,
        nesting_level: u32,
    ) -> std::io::Result<()> {
        let region_data = &self.rvsdg[region];
        let node_count = region_data.nodes().len() as u32;

        if node_count <= self.inline_max_node_count && nesting_level < self.inline_max_nesting_level
        {
            write!(writer, "\n")?;
            self.write_region(writer, region, header_name, indent, nesting_level + 1)?;
        } else {
            let indent_str = " ".repeat(indent);
            write!(
                writer,
                "{}{} ({}): {} child nodes",
                indent_str,
                header_name,
                self.format_region_id(region),
                node_count
            )?;
        }
        Ok(())
    }

    pub fn write_node<W: std::io::Write>(
        &self,
        writer: &mut W,
        node: Node,
        indent: usize,
        nesting_level: u32,
    ) -> std::io::Result<()> {
        match self.rvsdg[node].kind() {
            NodeKind::Function(_) => self.write_node_common(writer, node, "FunctionNode"),
            NodeKind::UniformBinding(_) => self.write_node_common(writer, node, "UniformBinding"),
            NodeKind::StorageBinding(_) => self.write_node_common(writer, node, "StorageBinding"),
            NodeKind::WorkgroupBinding(_) => {
                self.write_node_common(writer, node, "WorkgroupBinding")
            }
            NodeKind::Constant(_) => self.write_node_common(writer, node, "Constant"),
            NodeKind::Simple(_) => self.write_simple_node(writer, node),
            NodeKind::Switch(_) => self.write_switch_node(writer, node, indent, nesting_level),
            NodeKind::Loop(_) => self.write_loop_node(writer, node, indent, nesting_level),
        }
    }

    fn write_simple_node<W: std::io::Write>(
        &self,
        writer: &mut W,
        node: Node,
    ) -> std::io::Result<()> {
        use slir::rvsdg::SimpleNode;
        let node_data = &self.rvsdg[node];
        let simple = node_data.expect_simple();
        match simple {
            SimpleNode::ConstU32(n) => {
                self.write_node_common(writer, node, &format!("ConstU32{{{}}}", n.value()))
            }
            SimpleNode::ConstI32(n) => {
                self.write_node_common(writer, node, &format!("ConstI32{{{}}}", n.value()))
            }
            SimpleNode::ConstF32(n) => {
                self.write_node_common(writer, node, &format!("ConstF32{{{}}}", n.value()))
            }
            SimpleNode::ConstBool(n) => {
                self.write_node_common(writer, node, &format!("ConstBool{{{}}}", n.value()))
            }
            SimpleNode::ConstPredicate(n) => {
                self.write_node_common(writer, node, &format!("ConstPredicate{{{}}}", n.value()))
            }
            SimpleNode::ConstPtr(n) => self.write_node_common(
                writer,
                node,
                &format!(
                    "ConstPtr{{pointee_ty: {}}}",
                    self.format_type(n.pointee_ty())
                ),
            ),
            SimpleNode::ConstFallback(_) => self.write_node_common(writer, node, "ConstFallback"),
            SimpleNode::OpAlloca(n) => self.write_node_common(
                writer,
                node,
                &format!("OpAlloca{{ty: {}}}", self.format_type(n.ty())),
            ),
            SimpleNode::OpLoad(_) => self.write_node_common(writer, node, "OpLoad"),
            SimpleNode::OpStore(_) => self.write_node_common(writer, node, "OpStore"),
            SimpleNode::OpExtractField(n) => self.write_node_common(
                writer,
                node,
                &format!("OpExtractField{{field_index: {}}}", n.field_index()),
            ),
            SimpleNode::OpExtractElement(_) => {
                self.write_node_common(writer, node, "OpExtractElement")
            }
            SimpleNode::OpFieldPtr(n) => self.write_node_common(
                writer,
                node,
                &format!("OpFieldPtr{{field_index: {}}}", n.field_index()),
            ),
            SimpleNode::OpElementPtr(_) => self.write_node_common(writer, node, "OpElementPtr"),
            SimpleNode::OpDiscriminantPtr(_) => {
                self.write_node_common(writer, node, "OpDiscriminantPtr")
            }
            SimpleNode::OpVariantPtr(n) => self.write_node_common(
                writer,
                node,
                &format!("OpVariantPtr{{variant_index: {}}}", n.variant_index()),
            ),
            SimpleNode::OpGetDiscriminant(_) => {
                self.write_node_common(writer, node, "OpGetDiscriminant")
            }
            SimpleNode::OpSetDiscriminant(n) => self.write_node_common(
                writer,
                node,
                &format!("OpSetDiscriminant{{variant_index: {}}}", n.variant_index()),
            ),
            SimpleNode::OpOffsetSlice(_) => self.write_node_common(writer, node, "OpOffsetSlice"),
            SimpleNode::OpGetSliceOffset(_) => {
                self.write_node_common(writer, node, "OpGetSliceOffset")
            }
            SimpleNode::OpUnary(n) => self.write_node_common(
                writer,
                node,
                &format!("OpUnary{{operator: {}}}", n.operator()),
            ),
            SimpleNode::OpBinary(n) => self.write_node_common(
                writer,
                node,
                &format!("OpBinary{{operator: {}}}", n.operator()),
            ),
            SimpleNode::OpVector(n) => {
                self.write_node_common(writer, node, &format!("OpVector{{ty: {}}}", n.ty()))
            }
            SimpleNode::OpMatrix(n) => {
                self.write_node_common(writer, node, &format!("OpMatrix{{ty: {}}}", n.ty()))
            }
            SimpleNode::OpCaseToBranchSelector(n) => self.write_node_common(
                writer,
                node,
                &format!("OpCaseToBranchSelector{{cases: {:?}}}", n.cases()),
            ),
            SimpleNode::OpBoolToBranchSelector(_) => {
                self.write_node_common(writer, node, "OpBoolToBranchSelector")
            }
            SimpleNode::OpU32ToBranchSelector(n) => self.write_node_common(
                writer,
                node,
                &format!(
                    "OpU32ToBranchSelector{{branch_count: {}}}",
                    n.branch_count()
                ),
            ),
            SimpleNode::OpBranchSelectorToCase(n) => self.write_node_common(
                writer,
                node,
                &format!("OpBranchSelectorToCase{{cases: {:?}}}", n.cases()),
            ),
            SimpleNode::OpConvertToU32(_) => self.write_node_common(writer, node, "OpConvertToU32"),
            SimpleNode::OpConvertToI32(_) => self.write_node_common(writer, node, "OpConvertToI32"),
            SimpleNode::OpConvertToF32(_) => self.write_node_common(writer, node, "OpConvertToF32"),
            SimpleNode::OpConvertToBool(_) => {
                self.write_node_common(writer, node, "OpConvertToBool")
            }
            SimpleNode::OpArrayLength(_) => self.write_node_common(writer, node, "OpArrayLength"),
            SimpleNode::OpCall(_) => self.write_node_common(writer, node, "OpCall"),
            SimpleNode::ValueProxy(_) => self.write_node_common(writer, node, "ValueProxy"),
            SimpleNode::Reaggregation(_) => self.write_node_common(writer, node, "Reaggregation"),
        }
    }

    fn write_switch_node<W: std::io::Write>(
        &self,
        writer: &mut W,
        node: Node,
        indent: usize,
        nesting_level: u32,
    ) -> std::io::Result<()> {
        self.write_node_common(writer, node, "Switch")?;

        let node_data = &self.rvsdg[node];
        if let NodeKind::Switch(n) = node_data.kind() {
            for (i, &region) in n.branches().iter().enumerate() {
                let header = format!("Branch {}", i);
                self.write_nested_region_with_inlining(
                    writer,
                    region,
                    &header,
                    indent,
                    nesting_level,
                )?;
            }
        }
        Ok(())
    }

    fn write_loop_node<W: std::io::Write>(
        &self,
        writer: &mut W,
        node: Node,
        indent: usize,
        nesting_level: u32,
    ) -> std::io::Result<()> {
        self.write_node_common(writer, node, "Loop")?;

        let node_data = &self.rvsdg[node];
        if let NodeKind::Loop(n) = node_data.kind() {
            let region = n.loop_region();
            self.write_nested_region_with_inlining(
                writer,
                region,
                "Loop Region",
                indent,
                nesting_level,
            )?;
        }
        Ok(())
    }

    fn write_node_common<W: std::io::Write>(
        &self,
        writer: &mut W,
        node: Node,
        op_str: &str,
    ) -> std::io::Result<()> {
        let node_data = &self.rvsdg[node];
        let id_str = self.format_node_id(node);

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
            "[{}] {}({}){}",
            id_str,
            op_str,
            inputs.join(", "),
            state_str
        )?;

        write!(writer, " -> {}", all_outputs.join(", "))?;

        Ok(())
    }
}
