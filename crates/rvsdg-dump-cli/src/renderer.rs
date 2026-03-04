use std::collections::BTreeMap;

use slir::rvsdg::analyse::region_stratification::RegionStratifier;
use slir::rvsdg::{
    Connectivity, Node, NodeKind, Region, Rvsdg, SimpleNode, StateOrigin, ValueOrigin, ValueUser,
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

    pub fn write_node_id<W: std::io::Write>(
        &self,
        writer: &mut W,
        node: Node,
    ) -> std::io::Result<()> {
        let (index, version) = self.unpack_key_data(node.data());
        write!(writer, "Node({}v{})", index, version)
    }

    pub fn write_region_id<W: std::io::Write>(
        &self,
        writer: &mut W,
        region: Region,
    ) -> std::io::Result<()> {
        let (index, version) = self.unpack_key_data(region.data());
        write!(writer, "Region({}v{})", index, version)
    }

    fn unpack_key_data(&self, data: KeyData) -> (u32, u32) {
        let ffi = data.as_ffi();
        let version = (ffi >> 32) as u32;
        let index = ffi as u32;
        (index, version)
    }

    pub fn write_type<W: std::io::Write>(&self, writer: &mut W, ty: Type) -> std::io::Result<()> {
        let registry = self.rvsdg.ty();
        let kind = registry.kind(ty);
        match &*kind {
            TypeKind::Struct(_) => {
                write!(writer, "struct({})", ty.registration_id().unwrap())
            }
            TypeKind::Enum(_) => {
                write!(writer, "enum({})", ty.registration_id().unwrap())
            }
            _ => write!(writer, "{}", ty.to_string(registry)),
        }
    }

    pub fn write_type_detail<W: std::io::Write>(
        &self,
        writer: &mut W,
        ty: Type,
    ) -> std::io::Result<()> {
        let registry = self.rvsdg.ty();
        let kind = registry.kind(ty);
        match &*kind {
            TypeKind::Struct(s) => {
                self.write_type(writer, ty)?;
                writeln!(writer, ":")?;
                for (i, field) in s.fields.iter().enumerate() {
                    write!(writer, "  - field {}: ", i)?;
                    self.write_type(writer, field.ty)?;
                    write!(writer, " (offset: {})", field.offset)?;
                    if i < s.fields.len() - 1 {
                        writeln!(writer)?;
                    }
                }
                Ok(())
            }
            TypeKind::Enum(e) => {
                self.write_type(writer, ty)?;
                writeln!(writer, ":")?;
                for (i, variant_ty) in e.variants.iter().enumerate() {
                    write!(writer, "  - variant {}: ", i)?;
                    self.write_type(writer, *variant_ty)?;
                    if i < e.variants.len() - 1 {
                        writeln!(writer)?;
                    }
                }
                Ok(())
            }
            _ => {
                self.write_type(writer, ty)?;
                write!(writer, ": {:?}", &*kind)
            }
        }
    }

    pub fn write_value_origin<W: std::io::Write>(
        &self,
        writer: &mut W,
        origin: ValueOrigin,
        region: Option<Region>,
    ) -> std::io::Result<()> {
        match origin {
            ValueOrigin::Argument(arg) => {
                if let Some(r) = region {
                    self.write_region_id(writer, r)?;
                    write!(writer, "a{}", arg)
                } else {
                    write!(writer, "a{}", arg)
                }
            }
            ValueOrigin::Output { producer, output } => {
                self.write_node_id(writer, producer)?;
                write!(writer, "e{}", output)
            }
        }
    }

    pub fn write_value_user<W: std::io::Write>(
        &self,
        writer: &mut W,
        user: ValueUser,
        _region: Option<Region>,
    ) -> std::io::Result<()> {
        match user {
            ValueUser::Result(res) => write!(writer, "r{}", res),
            ValueUser::Input { consumer, input } => {
                self.write_node_id(writer, consumer)?;
                write!(writer, "i{}", input)
            }
        }
    }

    pub fn write_function_signature<W: std::io::Write>(
        &self,
        writer: &mut W,
        node: Node,
    ) -> std::io::Result<()> {
        let node_data = &self.rvsdg[node];
        let function_node = node_data.expect_function();
        let body_region = function_node.body_region();
        let body_data = &self.rvsdg[body_region];

        let args = body_data.value_arguments();
        let results = body_data.value_results();
        let dep_count = function_node.dependencies().len();

        write!(writer, "(")?;
        for (i, arg) in args.iter().enumerate().skip(dep_count) {
            // TODO: I find the order here weird. Why not skip first and then enumerate?
            if i > dep_count {
                write!(writer, ", ")?;
            }
            write!(writer, "arg{}: ", i - dep_count)?;
            self.write_type(writer, arg.ty)?;
        }
        write!(writer, ") -> ")?;

        if results.len() == 1 {
            self.write_type(writer, results[0].ty)?;
        } else {
            write!(writer, "(")?;
            for (i, res) in results.iter().enumerate() {
                if i > 0 {
                    write!(writer, ", ")?;
                }
                self.write_type(writer, res.ty)?;
            }
            write!(writer, ")")?;
        }

        Ok(())
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
        let indent_str = " ".repeat(indent);

        write!(writer, "{}{} (", indent_str, header_name)?;
        self.write_region_id(writer, region)?;
        writeln!(writer, "):")?;

        // Arguments
        write!(writer, "{}  Arguments: [", indent_str)?;
        for (i, arg) in region_data.value_arguments().iter().enumerate() {
            if i > 0 {
                write!(writer, ", ")?;
            }
            self.write_region_id(writer, region)?;
            write!(writer, "a{}: ", i)?;
            self.write_type(writer, arg.ty)?;
        }
        write!(writer, ", ")?;
        self.write_region_id(writer, region)?;
        write!(writer, "s: State")?;
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
        write!(writer, "{}  Results: [", indent_str)?;
        let res_count = region_data.value_results().len();
        for (i, res) in region_data.value_results().iter().enumerate() {
            if i > 0 {
                write!(writer, ", ")?;
            }
            self.write_value_origin(writer, res.origin, Some(region))?;
        }

        match region_data.state_result() {
            StateOrigin::Argument | StateOrigin::Node(_) => {
                if res_count > 0 {
                    write!(writer, ", ")?;
                }
                self.write_region_id(writer, region)?;
                write!(writer, "s")?;
            }
        }
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
            write!(writer, "{}{} (", indent_str, header_name)?;
            self.write_region_id(writer, region)?;
            write!(writer, "): {} child nodes", node_count)?;
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
        let node_data = &self.rvsdg[node];
        let simple = node_data.expect_simple();
        match simple {
            SimpleNode::ConstU32(n) => self.write_const_u32(writer, node, n),
            SimpleNode::ConstI32(n) => self.write_const_i32(writer, node, n),
            SimpleNode::ConstF32(n) => self.write_const_f32(writer, node, n),
            SimpleNode::ConstBool(n) => self.write_const_bool(writer, node, n),
            SimpleNode::ConstPredicate(n) => self.write_const_predicate(writer, node, n),
            SimpleNode::ConstPtr(n) => self.write_const_ptr(writer, node, n),
            SimpleNode::ConstFallback(_) => self.write_node_common(writer, node, "ConstFallback"),
            SimpleNode::OpAlloca(n) => self.write_op_alloca(writer, node, n),
            SimpleNode::OpLoad(_) => self.write_node_common(writer, node, "OpLoad"),
            SimpleNode::OpStore(_) => self.write_node_common(writer, node, "OpStore"),
            SimpleNode::OpExtractField(n) => self.write_op_extract_field(writer, node, n),
            SimpleNode::OpExtractElement(_) => {
                self.write_node_common(writer, node, "OpExtractElement")
            }
            SimpleNode::OpFieldPtr(n) => self.write_op_field_ptr(writer, node, n),
            SimpleNode::OpElementPtr(_) => self.write_node_common(writer, node, "OpElementPtr"),
            SimpleNode::OpDiscriminantPtr(_) => {
                self.write_node_common(writer, node, "OpDiscriminantPtr")
            }
            SimpleNode::OpVariantPtr(n) => self.write_op_variant_ptr(writer, node, n),
            SimpleNode::OpGetDiscriminant(_) => {
                self.write_node_common(writer, node, "OpGetDiscriminant")
            }
            SimpleNode::OpSetDiscriminant(n) => self.write_op_set_discriminant(writer, node, n),
            SimpleNode::OpOffsetSlice(_) => self.write_node_common(writer, node, "OpOffsetSlice"),
            SimpleNode::OpGetSliceOffset(_) => {
                self.write_node_common(writer, node, "OpGetSliceOffset")
            }
            SimpleNode::OpUnary(n) => self.write_op_unary(writer, node, n),
            SimpleNode::OpBinary(n) => self.write_op_binary(writer, node, n),
            SimpleNode::OpVector(n) => self.write_op_vector(writer, node, n),
            SimpleNode::OpMatrix(n) => self.write_op_matrix(writer, node, n),
            SimpleNode::OpCaseToBranchSelector(n) => {
                self.write_op_case_to_branch_selector(writer, node, n)
            }
            SimpleNode::OpBoolToBranchSelector(_) => {
                self.write_node_common(writer, node, "OpBoolToBranchSelector")
            }
            SimpleNode::OpU32ToBranchSelector(n) => {
                self.write_op_u32_to_branch_selector(writer, node, n)
            }
            SimpleNode::OpBranchSelectorToCase(n) => {
                self.write_op_branch_selector_to_case(writer, node, n)
            }
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

    fn write_const_u32<W: std::io::Write>(
        &self,
        writer: &mut W,
        node: Node,
        n: &slir::rvsdg::ConstU32,
    ) -> std::io::Result<()> {
        write!(writer, "[")?;
        self.write_node_id(writer, node)?;
        write!(writer, "] ConstU32{{{}}}", n.value())?;
        self.write_node_io_signature(writer, node)
    }

    fn write_const_i32<W: std::io::Write>(
        &self,
        writer: &mut W,
        node: Node,
        n: &slir::rvsdg::ConstI32,
    ) -> std::io::Result<()> {
        write!(writer, "[")?;
        self.write_node_id(writer, node)?;
        write!(writer, "] ConstI32{{{}}}", n.value())?;
        self.write_node_io_signature(writer, node)
    }

    fn write_const_f32<W: std::io::Write>(
        &self,
        writer: &mut W,
        node: Node,
        n: &slir::rvsdg::ConstF32,
    ) -> std::io::Result<()> {
        write!(writer, "[")?;
        self.write_node_id(writer, node)?;
        write!(writer, "] ConstF32{{{}}}", n.value())?;
        self.write_node_io_signature(writer, node)
    }

    fn write_const_bool<W: std::io::Write>(
        &self,
        writer: &mut W,
        node: Node,
        n: &slir::rvsdg::ConstBool,
    ) -> std::io::Result<()> {
        write!(writer, "[")?;
        self.write_node_id(writer, node)?;
        write!(writer, "] ConstBool{{{}}}", n.value())?;
        self.write_node_io_signature(writer, node)
    }

    fn write_const_predicate<W: std::io::Write>(
        &self,
        writer: &mut W,
        node: Node,
        n: &slir::rvsdg::ConstPredicate,
    ) -> std::io::Result<()> {
        write!(writer, "[")?;
        self.write_node_id(writer, node)?;
        write!(writer, "] ConstPredicate{{{}}}", n.value())?;
        self.write_node_io_signature(writer, node)
    }

    fn write_const_ptr<W: std::io::Write>(
        &self,
        writer: &mut W,
        node: Node,
        n: &slir::rvsdg::ConstPtr,
    ) -> std::io::Result<()> {
        write!(writer, "[")?;
        self.write_node_id(writer, node)?;
        write!(writer, "] ConstPtr{{pointee_ty: ")?;
        self.write_type(writer, n.pointee_ty())?;
        write!(writer, "}}")?;
        self.write_node_io_signature(writer, node)
    }

    fn write_op_alloca<W: std::io::Write>(
        &self,
        writer: &mut W,
        node: Node,
        n: &slir::rvsdg::OpAlloca,
    ) -> std::io::Result<()> {
        write!(writer, "[")?;
        self.write_node_id(writer, node)?;
        write!(writer, "] OpAlloca{{ty: ")?;
        self.write_type(writer, n.ty())?;
        write!(writer, "}}")?;
        self.write_node_io_signature(writer, node)
    }

    fn write_op_extract_field<W: std::io::Write>(
        &self,
        writer: &mut W,
        node: Node,
        n: &slir::rvsdg::OpExtractField,
    ) -> std::io::Result<()> {
        write!(writer, "[")?;
        self.write_node_id(writer, node)?;
        write!(
            writer,
            "] OpExtractField{{field_index: {}}}",
            n.field_index()
        )?;
        self.write_node_io_signature(writer, node)
    }

    fn write_op_field_ptr<W: std::io::Write>(
        &self,
        writer: &mut W,
        node: Node,
        n: &slir::rvsdg::OpFieldPtr,
    ) -> std::io::Result<()> {
        write!(writer, "[")?;
        self.write_node_id(writer, node)?;
        write!(writer, "] OpFieldPtr{{field_index: {}}}", n.field_index())?;
        self.write_node_io_signature(writer, node)
    }

    fn write_op_variant_ptr<W: std::io::Write>(
        &self,
        writer: &mut W,
        node: Node,
        n: &slir::rvsdg::OpVariantPtr,
    ) -> std::io::Result<()> {
        write!(writer, "[")?;
        self.write_node_id(writer, node)?;
        write!(
            writer,
            "] OpVariantPtr{{variant_index: {}}}",
            n.variant_index()
        )?;
        self.write_node_io_signature(writer, node)
    }

    fn write_op_set_discriminant<W: std::io::Write>(
        &self,
        writer: &mut W,
        node: Node,
        n: &slir::rvsdg::OpSetDiscriminant,
    ) -> std::io::Result<()> {
        write!(writer, "[")?;
        self.write_node_id(writer, node)?;
        write!(
            writer,
            "] OpSetDiscriminant{{variant_index: {}}}",
            n.variant_index()
        )?;
        self.write_node_io_signature(writer, node)
    }

    fn write_op_unary<W: std::io::Write>(
        &self,
        writer: &mut W,
        node: Node,
        n: &slir::rvsdg::OpUnary,
    ) -> std::io::Result<()> {
        write!(writer, "[")?;
        self.write_node_id(writer, node)?;
        write!(writer, "] OpUnary{{operator: {}}}", n.operator())?;
        self.write_node_io_signature(writer, node)
    }

    fn write_op_binary<W: std::io::Write>(
        &self,
        writer: &mut W,
        node: Node,
        n: &slir::rvsdg::OpBinary,
    ) -> std::io::Result<()> {
        write!(writer, "[")?;
        self.write_node_id(writer, node)?;
        write!(writer, "] OpBinary{{operator: {}}}", n.operator())?;
        self.write_node_io_signature(writer, node)
    }

    fn write_op_vector<W: std::io::Write>(
        &self,
        writer: &mut W,
        node: Node,
        n: &slir::rvsdg::OpVector,
    ) -> std::io::Result<()> {
        write!(writer, "[")?;
        self.write_node_id(writer, node)?;
        write!(writer, "] OpVector{{ty: {}}}", n.ty())?;
        self.write_node_io_signature(writer, node)
    }

    fn write_op_matrix<W: std::io::Write>(
        &self,
        writer: &mut W,
        node: Node,
        n: &slir::rvsdg::OpMatrix,
    ) -> std::io::Result<()> {
        write!(writer, "[")?;
        self.write_node_id(writer, node)?;
        write!(writer, "] OpMatrix{{ty: {}}}", n.ty())?;
        self.write_node_io_signature(writer, node)
    }

    fn write_op_case_to_branch_selector<W: std::io::Write>(
        &self,
        writer: &mut W,
        node: Node,
        n: &slir::rvsdg::OpCaseToBranchSelector,
    ) -> std::io::Result<()> {
        write!(writer, "[")?;
        self.write_node_id(writer, node)?;
        write!(writer, "] OpCaseToBranchSelector{{cases: {:?}}}", n.cases())?;
        self.write_node_io_signature(writer, node)
    }

    fn write_op_u32_to_branch_selector<W: std::io::Write>(
        &self,
        writer: &mut W,
        node: Node,
        n: &slir::rvsdg::OpU32ToBranchSelector,
    ) -> std::io::Result<()> {
        write!(writer, "[")?;
        self.write_node_id(writer, node)?;
        write!(
            writer,
            "] OpU32ToBranchSelector{{branch_count: {}}}",
            n.branch_count()
        )?;
        self.write_node_io_signature(writer, node)
    }

    fn write_op_branch_selector_to_case<W: std::io::Write>(
        &self,
        writer: &mut W,
        node: Node,
        n: &slir::rvsdg::OpBranchSelectorToCase,
    ) -> std::io::Result<()> {
        write!(writer, "[")?;
        self.write_node_id(writer, node)?;
        write!(writer, "] OpBranchSelectorToCase{{cases: {:?}}}", n.cases())?;
        self.write_node_io_signature(writer, node)
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
        let n = node_data.expect_switch();
        for (i, &region) in n.branches().iter().enumerate() {
            let header = format!("Branch {}", i);
            self.write_nested_region_with_inlining(writer, region, &header, indent, nesting_level)?;
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
        let n = node_data.expect_loop();
        let region = n.loop_region();
        self.write_nested_region_with_inlining(
            writer,
            region,
            "Loop Region",
            indent,
            nesting_level,
        )?;
        Ok(())
    }

    fn write_node_common<W: std::io::Write>(
        &self,
        writer: &mut W,
        node: Node,
        op_str: &str,
    ) -> std::io::Result<()> {
        write!(writer, "[")?;
        self.write_node_id(writer, node)?;
        write!(writer, "] {}", op_str)?;

        self.write_node_io_signature(writer, node)
    }

    fn write_node_io_signature<W: std::io::Write>(
        &self,
        writer: &mut W,
        node: Node,
    ) -> std::io::Result<()> {
        let node_data = &self.rvsdg[node];

        write!(writer, "(")?;
        for (i, input) in node_data.value_inputs().iter().enumerate() {
            if i > 0 {
                write!(writer, ", ")?;
            }
            self.write_value_origin(writer, input.origin, node_data.region().into())?;
        }
        write!(writer, ")")?;

        if let Some(state) = node_data.state() {
            match state.origin {
                StateOrigin::Argument => write!(writer, " (state: Arg)")?,
                StateOrigin::Node(n) => {
                    write!(writer, " (state: ")?;
                    self.write_node_id(writer, n)?;
                    write!(writer, ")")?;
                }
            }
        }

        write!(writer, " -> ")?;
        let mut first = true;
        for (i, out) in node_data.value_outputs().iter().enumerate() {
            if !first {
                write!(writer, ", ")?;
            }
            self.write_node_id(writer, node)?;
            write!(writer, "e{} : ", i)?;
            self.write_type(writer, out.ty)?;
            first = false;
        }

        if node_data.state().is_some() {
            if !first {
                write!(writer, ", ")?;
            }
            self.write_node_id(writer, node)?;
            write!(writer, "s : State")?;
        }

        Ok(())
    }
}
