use std::fmt::Write;
use std::hash::Hash;
use std::rc::Rc;

use indexmap::IndexSet;
use rustc_hash::{FxHashMap, FxHashSet};
use slotmap::SecondaryMap;

use crate::scf::analyze::global_use::collect_used_global_bindings;
use crate::scf::analyze::local_binding_use;
use crate::scf::analyze::local_binding_use::count_local_binding_use;
use crate::scf::analyze::struct_use::collect_used_structs;
use crate::scf::{
    Alloca, Block, ExprBinding, Expression, ExpressionKind, GlobalPtr, If, LocalBinding,
    LocalBindingKind, Loop, LoopControl, OpBinary, OpCallBuiltin, OpExtractElement, OpMatrix,
    OpPtrElementPtr, OpUnary, OpVector, Return, Scf, Statement, StatementKind, Store, Switch,
};
use crate::ty::{ScalarKind, Struct, StructField, Type, TypeKind, VectorSize};
use crate::{
    BinaryOperator, BlendSrc, Constant, ConstantKind, EntryPointKind, Function, Interpolation,
    InterpolationSampling, InterpolationType, Module, OverridableConstantKind, ResourceBinding,
    ShaderIOBinding, StorageBinding, Symbol, UnaryOperator, UniformBinding, WorkgroupBinding, ty,
};

const INDENT: &'static str = "    ";

struct IdWriter<K> {
    current: u32,
    mapping: FxHashMap<K, u32>,
    prefix: &'static str,
    reserved_names: Rc<FxHashSet<String>>,
    buffer: String,
}

impl<K> IdWriter<K>
where
    K: Eq + Hash,
{
    fn new(prefix: &'static str, reserved_names: Rc<FxHashSet<String>>) -> Self {
        Self {
            current: 0,
            mapping: Default::default(),
            prefix,
            reserved_names,
            buffer: String::new(),
        }
    }

    fn write<W: Write>(&mut self, w: &mut W, key: K) {
        if let Some(id) = self.mapping.get(&key) {
            w.write_str(self.prefix);
            write!(w, "{}", id);
        } else {
            self.update_buffer();

            while self.reserved_names.contains(&self.buffer) {
                self.current += 1;
                self.update_buffer();
            }

            write!(w, "{}", self.buffer).unwrap();
            self.mapping.insert(key, self.current);
            self.current += 1;
        }
    }

    fn update_buffer(&mut self) {
        self.buffer.clear();

        self.buffer.push_str(self.prefix);
        write!(self.buffer, "{}", self.current).unwrap();
    }

    fn reset(&mut self) {
        self.mapping.clear();
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BinOpSide {
    Left,
    Right,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum InlineContext {
    /// Whatever expression we're inlining into does not need to pass any specific context
    /// information.
    None,

    /// We're inlining an expression as the operand of an [OpUnary] expression.
    UnOp(UnaryOperator),

    /// We're inlining an expression as either the left-hand-side or the right-hand-side of an
    /// [OpBinary] expression.
    BinOp(BinOpSide, BinaryOperator),

    /// We're inlining the result of an [OpLoad] expression, or we're inlining into the place
    /// expression of a [Store] statement.
    Deref {
        /// Whether the dereferencing operation needs parentheses.
        ///
        /// E.g.:
        ///
        /// ```pseudocode
        /// (*ptr)._1; // If `needs_parens` is true
        /// ```
        needs_parens: bool,
    },

    /// We're inlining an expression as the base value of an [OpPtrElementPtr] expression.
    Reref,

    /// We're inlining an expression as the base value of an [OpExtractElement] expression.
    Extract,
}

#[derive(Clone, Copy, Debug)]
struct Context<'a> {
    module: &'a Module,
    scf: &'a Scf,
    local_binding_use_counts: &'a SecondaryMap<LocalBinding, u32>,
    used_structs: &'a FxHashSet<Type>,
    used_workgroup_bindings: &'a FxHashSet<WorkgroupBinding>,
    used_constants: &'a FxHashSet<Constant>,
}

struct WgslModuleWriter {
    w: String,
    struct_id_writer: IdWriter<Type>,
    uniform_binding_id_writer: IdWriter<UniformBinding>,
    storage_binding_id_writer: IdWriter<StorageBinding>,
    workgroup_binding_id_writer: IdWriter<WorkgroupBinding>,
    constant_id_writer: IdWriter<Constant>,
    function_id_writer: IdWriter<Function>,
    local_binding_id_writer: IdWriter<LocalBinding>,
    indent_level: usize,
}

impl WgslModuleWriter {
    fn new(reserved_names: FxHashSet<String>) -> Self {
        let reserved_names = Rc::new(reserved_names);

        Self {
            w: String::new(),
            struct_id_writer: IdWriter::new("S", reserved_names.clone()),
            uniform_binding_id_writer: IdWriter::new("u", reserved_names.clone()),
            storage_binding_id_writer: IdWriter::new("s", reserved_names.clone()),
            workgroup_binding_id_writer: IdWriter::new("w", reserved_names.clone()),
            constant_id_writer: IdWriter::new("C", reserved_names.clone()),
            function_id_writer: IdWriter::new("f", reserved_names.clone()),
            local_binding_id_writer: IdWriter::new("l", reserved_names),
            indent_level: 0,
        }
    }

    fn write_required_language_extensions(&mut self) {
        // TODO: add 'uniform_buffer_standard_layout' when both naga and tint support it
        self.w.push_str("requires pointer_composite_access;");
        self.write_newline();
        self.write_newline();
    }

    fn write_struct_decls(&mut self, cx: Context) {
        for ty in cx.used_structs {
            self.write_struct_decl(cx, *ty);
        }
    }

    fn write_constant_bindings(&mut self, cx: Context) {
        for constant in cx.used_constants {
            self.write_constant_binding(cx, *constant);
        }
    }

    fn write_uniform_bindings(&mut self, cx: Context) {
        for binding in cx.module.uniform_bindings.keys() {
            self.write_uniform_binding(cx, binding);
        }
    }

    fn write_storage_bindings(&mut self, cx: Context) {
        for binding in cx.module.storage_bindings.keys() {
            self.write_storage_binding(cx, binding);
        }
    }

    fn write_workgroup_bindings(&mut self, cx: Context) {
        for binding in cx.module.workgroup_bindings.keys() {
            if cx.used_workgroup_bindings.contains(&binding) {
                self.write_workgroup_binding(cx, binding);
            }
        }
    }

    fn write_entry_points(&mut self, cx: Context) {
        for (entry_point, _) in cx.module.entry_points.iter() {
            self.write_function_decl(cx, entry_point);
        }
    }

    fn increment_indent(&mut self) {
        self.indent_level += 1;
    }

    fn decrement_indent(&mut self) {
        self.indent_level -= 1;
    }

    fn write_newline(&mut self) {
        self.w.push_str("\n");

        for _ in 0..self.indent_level {
            self.w.push_str(INDENT);
        }
    }

    fn write_optional_space(&mut self) {
        self.w.push_str(" ");
    }

    fn write_uniform_binding_id(&mut self, uniform_binding: UniformBinding) {
        self.uniform_binding_id_writer
            .write(&mut self.w, uniform_binding);
    }

    fn write_storage_binding_id(&mut self, storage_binding: StorageBinding) {
        self.storage_binding_id_writer
            .write(&mut self.w, storage_binding);
    }

    fn write_workgroup_binding_id(&mut self, workgroup_binding: WorkgroupBinding) {
        self.workgroup_binding_id_writer
            .write(&mut self.w, workgroup_binding);
    }

    fn write_constant_id(&mut self, constant: Constant) {
        self.constant_id_writer.write(&mut self.w, constant);
    }

    fn write_function_id(&mut self, cx: Context, function: Function) {
        if let Some(entry_point) = cx.module.entry_points.get(function) {
            write!(&mut self.w, "{}", entry_point.name).unwrap();
        } else {
            self.function_id_writer.write(&mut self.w, function);
        }
    }

    fn write_struct_id(&mut self, struct_ty: Type) {
        self.struct_id_writer.write(&mut self.w, struct_ty);
    }

    fn write_local_binding_id(&mut self, local_binding: LocalBinding) {
        self.local_binding_id_writer
            .write(&mut self.w, local_binding);
    }

    fn write_struct_decl(&mut self, cx: Context, ty: Type) {
        self.w.push_str("struct ");
        self.write_struct_id(ty);
        self.write_optional_space();
        self.w.push_str("{");
        self.increment_indent();
        self.write_newline();

        for (index, field) in cx
            .module
            .ty
            .kind(ty)
            .expect_struct()
            .fields
            .iter()
            .enumerate()
        {
            self.write_struct_field(cx, index, field);
        }

        self.decrement_indent();
        self.w.push_str("}");
        self.write_newline();
    }

    fn write_struct_field(&mut self, cx: Context, index: usize, field: &StructField) {
        // TODO: field offsets, but only for ABI compatible types (which notably does not
        // include booleans, as the size does not match; 1 byte CPU-side, 4 bytes GPU-side)

        if let Some(io_binding) = &field.io_binding {
            self.write_shader_io_binding(io_binding);
            self.write_newline();
        }

        self.write_struct_field_ident(index);
        self.w.push_str(":");
        self.write_optional_space();
        self.write_type(cx, field.ty);
        self.w.push_str(",");
        self.write_newline();
    }

    fn write_struct_field_ident(&mut self, index: usize) {
        write!(&mut self.w, "_{index}").unwrap();
    }

    fn write_shader_io_binding(&mut self, io_binding: &ShaderIOBinding) {
        match io_binding {
            ShaderIOBinding::VertexIndex => self.w.push_str("@builtin(vertex_index)"),
            ShaderIOBinding::InstanceIndex => self.w.push_str("@builtin(instance_index)"),
            ShaderIOBinding::Position { invariant } => {
                self.w.push_str("@builtin(position)");

                if *invariant {
                    self.write_optional_space();
                    self.w.push_str("@invariant");
                }
            }
            ShaderIOBinding::FrontFacing => self.w.push_str("@builtin(front_facing)"),
            ShaderIOBinding::FragDepth => self.w.push_str("@builtin(frag_depth)"),
            ShaderIOBinding::SampleIndex => self.w.push_str("@builtin(sample_index)"),
            ShaderIOBinding::SampleMask => self.w.push_str("@builtin(sample_mask)"),
            ShaderIOBinding::LocalInvocationId => self.w.push_str("@builtin(local_invocation_id)"),
            ShaderIOBinding::LocalInvocationIndex => {
                self.w.push_str("@builtin(local_invocation_index)")
            }
            ShaderIOBinding::GlobalInvocationId => {
                self.w.push_str("@builtin(global_invocation_id)")
            }
            ShaderIOBinding::WorkgroupId => self.w.push_str("@builtin(workgroup_id)"),
            ShaderIOBinding::NumWorkgroups => self.w.push_str("@builtin(num_workgroups)"),
            ShaderIOBinding::Location {
                location,
                blend_src,
                interpolation,
            } => {
                write!(&mut self.w, "@location{location}");

                if let Some(blend_src) = blend_src {
                    self.write_optional_space();
                    self.write_blend_src_attribute(blend_src);
                }

                if let Some(interpolation) = interpolation {
                    self.write_optional_space();
                    self.write_interpolation_attribute(interpolation);
                }
            }
        }
    }

    fn write_blend_src_attribute(&mut self, blend_src: &BlendSrc) {
        match blend_src {
            BlendSrc::Zero => self.w.push_str("@blend_src(0)"),
            BlendSrc::One => self.w.push_str("@blend_src(1)"),
        }
    }

    fn write_interpolation_attribute(&mut self, interpolation: &Interpolation) {
        self.w.push_str("@interpolate(");
        self.write_interpolation_type(interpolation.tpe);

        if let Some(sampling) = interpolation.sampling {
            self.w.push_str(",");
            self.write_optional_space();
            self.write_interpolation_sampling(sampling);
        }
    }

    fn write_interpolation_type(&mut self, interpolation_type: InterpolationType) {
        match interpolation_type {
            InterpolationType::Perspective => self.w.push_str("perspective"),
            InterpolationType::Linear => self.w.push_str("linear"),
            InterpolationType::Flat => self.w.push_str("flat"),
        }
    }

    fn write_interpolation_sampling(&mut self, interpolation_sampling: InterpolationSampling) {
        match interpolation_sampling {
            InterpolationSampling::Center => self.w.push_str("center"),
            InterpolationSampling::Centroid => self.w.push_str("centroid"),
            InterpolationSampling::Sample => self.w.push_str("sample"),
            InterpolationSampling::First => self.w.push_str("first"),
            InterpolationSampling::Either => self.w.push_str("either"),
        }
    }

    fn write_type(&mut self, cx: Context, ty: Type) {
        match &*cx.module.ty.kind(ty) {
            TypeKind::Scalar(kind) => self.write_scalar_ty(*kind),
            TypeKind::Atomic(kind) => self.write_atomic_ty(*kind),
            TypeKind::Vector(v) => self.write_vector_ty(v),
            TypeKind::Matrix(m) => self.write_matrix_ty(m),
            TypeKind::Array {
                element_ty, count, ..
            } => self.write_array_ty(cx, *element_ty, *count),
            TypeKind::Slice { element_ty, .. } => self.write_slice_ty(cx, *element_ty),
            TypeKind::Struct(_) => self.write_struct_id(ty),
            TypeKind::Ptr(_)
            | TypeKind::Enum(_)
            | TypeKind::Function(_)
            | TypeKind::Predicate
            | TypeKind::Dummy => panic!("type should not occur in final output"),
        }
    }

    fn write_scalar_ty(&mut self, kind: ScalarKind) {
        match kind {
            ScalarKind::I32 => self.w.push_str("i32"),
            ScalarKind::U32 => self.w.push_str("u32"),
            ScalarKind::F32 => self.w.push_str("f32"),
            ScalarKind::Bool => self.w.push_str("bool"),
        }
    }

    fn write_atomic_ty(&mut self, kind: ScalarKind) {
        self.w.push_str("atomic<");
        self.write_scalar_ty(kind);
        self.w.push_str(">");
    }

    fn write_vector_ty(&mut self, ty: &ty::Vector) {
        match ty.size {
            VectorSize::Two => self.w.push_str("vec2"),
            VectorSize::Three => self.w.push_str("vec3"),
            VectorSize::Four => self.w.push_str("vec4"),
        };

        self.w.push_str("<");
        self.write_scalar_ty(ty.scalar);
        self.w.push_str(">");
    }

    fn write_matrix_ty(&mut self, ty: &ty::Matrix) {
        let base = match (ty.rows, ty.columns) {
            (VectorSize::Two, VectorSize::Two) => "mat2x2",
            (VectorSize::Two, VectorSize::Three) => "mat2x3",
            (VectorSize::Two, VectorSize::Four) => "mat2x4",
            (VectorSize::Three, VectorSize::Two) => "mat3x2",
            (VectorSize::Three, VectorSize::Three) => "mat3x3",
            (VectorSize::Three, VectorSize::Four) => "mat3x4",
            (VectorSize::Four, VectorSize::Two) => "mat4x2",
            (VectorSize::Four, VectorSize::Three) => "mat4x3",
            (VectorSize::Four, VectorSize::Four) => "mat4x4",
        };

        self.w.push_str(base);
        self.w.push_str("<");
        self.write_scalar_ty(ty.scalar);
        self.w.push_str(">");
    }

    fn write_array_ty(&mut self, cx: Context, element_ty: Type, count: u64) {
        self.w.push_str("array<");
        self.write_type(cx, element_ty);
        self.w.push_str(",");
        self.write_optional_space();
        write!(&mut self.w, "{count}>").unwrap();
    }

    fn write_slice_ty(&mut self, cx: Context, element_ty: Type) {
        self.w.push_str("array<");
        self.write_type(cx, element_ty);
        self.w.push_str(">");
    }

    fn write_uniform_binding(&mut self, cx: Context, binding: UniformBinding) {
        let data = &cx.module.uniform_bindings[binding];

        self.write_resource_binding_attributes(&data.resource_binding);
        self.write_newline();
        self.w.push_str("var<uniform>");
        self.write_optional_space();
        self.write_uniform_binding_id(binding);
        self.w.push_str(":");
        self.write_optional_space();
        self.write_type(cx, data.ty);
        self.w.push_str(";");
        self.write_newline();
        self.write_newline();
    }

    fn write_storage_binding(&mut self, cx: Context, binding: StorageBinding) {
        let data = &cx.module.storage_bindings[binding];

        self.write_resource_binding_attributes(&data.resource_binding);
        self.write_newline();

        if data.writable {
            self.w.push_str("var<storage,read_write>");
        } else {
            self.w.push_str("var<storage>");
        }

        self.write_optional_space();
        self.write_storage_binding_id(binding);
        self.w.push_str(":");
        self.write_optional_space();
        self.write_type(cx, data.ty);
        self.w.push_str(";");
        self.write_newline();
        self.write_newline();
    }

    fn write_resource_binding_attributes(&mut self, resource_binding: &ResourceBinding) {
        let group = resource_binding.group;
        let binding = resource_binding.binding;

        write!(&mut self.w, "@group({group})").unwrap();
        self.write_optional_space();
        write!(&mut self.w, "@binding({binding})").unwrap();
    }

    fn write_workgroup_binding(&mut self, cx: Context, binding: WorkgroupBinding) {
        let data = &cx.module.workgroup_bindings[binding];

        self.w.push_str("var<workgroup>");
        self.write_optional_space();
        self.write_workgroup_binding_id(binding);
        self.w.push_str(":");
        self.write_optional_space();
        self.write_type(cx, data.ty);
        self.w.push_str(";");
        self.write_newline();
        self.write_newline();
    }

    fn write_constant_binding(&mut self, cx: Context, constant: Constant) {
        let data = &cx.module.constants[constant];
        let ty = data.ty();

        if let ConstantKind::Overridable(overridable) = data.kind() {
            write!(&mut self.w, "@id({})", overridable.id()).unwrap();
            self.write_newline();
            self.w.push_str("override ");
        } else {
            self.w.push_str("const ");
        };

        self.write_constant_id(constant);
        self.w.push_str(":");
        self.write_optional_space();
        self.write_type(cx, ty);

        match data.kind() {
            ConstantKind::ByteData(data) => {
                self.write_optional_space();
                self.w.push_str("=");
                self.write_optional_space();

                let mut writer = ConstantValueWriter { writer: self, data };

                writer.write_value(cx, ty, 0);
            }
            ConstantKind::Expression => todo!(),
            ConstantKind::Overridable(overridable) => {
                if overridable.kind().override_required() {
                    self.write_optional_space();
                    self.w.push_str("=");
                    self.write_optional_space();

                    match overridable.kind() {
                        OverridableConstantKind::Float(Some(v)) => self.write_f32(*v),
                        OverridableConstantKind::Bool(Some(v)) => self.write_bool(*v),
                        OverridableConstantKind::SignedInteger(Some(v)) => self.write_i32(*v),
                        OverridableConstantKind::UnsignedInteger(Some(v)) => self.write_u32(*v),
                        _ => unreachable!(),
                    }
                }
            }
        }

        self.w.push_str(";");
        self.write_newline();
        self.write_newline();
    }

    fn write_function_decl(&mut self, cx: Context, function: Function) {
        self.local_binding_id_writer.reset();

        let sig = &cx.module.fn_sigs[function];
        let body = cx
            .scf
            .get_function_body(function)
            .expect("function not registered with SCF");

        if let Some(entry_point) = cx.module.entry_points.get(function) {
            self.write_entry_point_kind(&entry_point.kind);
            self.write_newline();
        }

        self.w.push_str("fn ");
        self.write_function_id(cx, function);
        self.w.push_str("(");

        let last_arg_index = sig.args.len() - 1;

        for (i, (arg, binding)) in sig.args.iter().zip(body.argument_bindings()).enumerate() {
            if let Some(io_binding) = &arg.shader_io_binding {
                self.write_shader_io_binding(io_binding);
                self.w.push_str(" ");
            }

            self.write_local_binding_id(*binding);
            self.w.push_str(":");
            self.write_optional_space();
            self.write_type(cx, arg.ty);

            if i != last_arg_index {
                self.w.push_str(",");
                self.write_optional_space();
            }
        }

        self.w.push_str(")");

        if let Some(ret_ty) = sig.ret_ty {
            self.write_optional_space();
            self.w.push_str("->");
            self.write_optional_space();
            self.write_type(cx, ret_ty);
        }

        self.write_optional_space();
        self.write_block(cx, body.block());
        self.write_newline();
        self.write_newline();
    }

    fn write_entry_point_kind(&mut self, entry_point_kind: &EntryPointKind) {
        match entry_point_kind {
            EntryPointKind::Vertex => self.w.push_str("@vertex"),
            EntryPointKind::Fragment => self.w.push_str("@fragment"),
            EntryPointKind::Compute(x, y, z) => {
                self.w.push_str("@compute");
                self.write_optional_space();
                write!(&mut self.w, "@workgroup_size({x},{y},{z})").unwrap();
            }
        }
    }

    fn write_block(&mut self, cx: Context, block: Block) {
        self.write_block_start();
        self.write_block_content(cx, block);
        self.write_block_end();
    }

    fn write_block_start(&mut self) {
        self.w.push_str("{");
        self.increment_indent();
    }

    fn write_block_end(&mut self) {
        self.decrement_indent();
        self.write_newline();
        self.w.push_str("}");
    }

    fn write_block_content(&mut self, cx: Context, block: Block) {
        let data = &cx.scf[block];

        if !data.statements().is_empty() {
            self.write_newline();
        }

        for statement in data.statements() {
            self.write_statement(cx, *statement);
        }

        for (var, expr) in data.control_flow_var_iter() {
            self.write_newline();
            self.write_local_binding_id(var);
            self.write_optional_space();
            self.w.push_str("=");
            self.write_optional_space();
            self.write_local_value(cx, expr, InlineContext::None);
            self.w.push_str(";")
        }
    }

    fn write_var_uninit(&mut self, cx: Context, binding: LocalBinding) {
        let ty = cx.scf[binding].ty();

        self.w.push_str("var ");
        self.write_local_binding_id(binding);
        self.w.push_str(":");
        self.write_optional_space();
        self.write_type(cx, ty);
        self.w.push_str(";");
        self.write_newline();
    }

    fn write_statement(&mut self, cx: Context, statement: Statement) {
        match cx.scf[statement].kind() {
            StatementKind::If(stmt) => self.write_stmt_if(cx, stmt),
            StatementKind::Switch(stmt) => self.write_stmt_switch(cx, stmt),
            StatementKind::Loop(stmt) => self.write_stmt_loop(cx, stmt),
            StatementKind::Return(stmt) => self.write_stmt_return(cx, stmt),
            StatementKind::ExprBinding(stmt) => self.write_stmt_expr_binding(cx, stmt),
            StatementKind::Alloca(stmt) => self.write_stmt_alloca(cx, stmt),
            StatementKind::Store(stmt) => self.write_stmt_store(cx, stmt),
            StatementKind::CallBuiltin(stmt) => self.write_stmt_call_builtin(cx, stmt),
        }
    }

    fn write_stmt_if(&mut self, cx: Context, if_stmt: &If) {
        for var in if_stmt.out_vars() {
            self.write_var_uninit(cx, *var);
        }

        if !if_stmt.out_vars().is_empty() {
            self.write_newline();
        }

        self.w.push_str("if ");
        self.write_local_value(cx, if_stmt.condition(), InlineContext::None);
        self.write_optional_space();
        self.write_block(cx, if_stmt.then_block());

        if let Some(else_block) = if_stmt.else_block() {
            self.write_optional_space();
            self.w.push_str("else");
            self.write_optional_space();
            self.write_block(cx, else_block);
        }

        self.write_newline();
    }

    fn write_stmt_switch(&mut self, cx: Context, switch_stmt: &Switch) {
        for var in switch_stmt.out_vars() {
            self.write_var_uninit(cx, *var);
        }

        if !switch_stmt.out_vars().is_empty() {
            self.write_newline();
        }

        self.w.push_str("switch ");
        self.write_local_value(cx, switch_stmt.on(), InlineContext::None);
        self.write_optional_space();
        self.w.push_str("{");
        self.increment_indent();

        for case in switch_stmt.cases() {
            self.write_newline();
            self.w.push_str("case ");
            self.write_u32(case.case());
            self.w.push_str(":");
            self.write_optional_space();
            self.write_block(cx, case.block());
        }

        self.write_newline();
        self.w.push_str("default:");
        self.write_optional_space();
        self.write_block(cx, switch_stmt.default());

        self.decrement_indent();
        self.write_newline();
        self.w.push_str("}");
        self.write_newline();
    }

    fn write_stmt_loop(&mut self, cx: Context, loop_stmt: &Loop) {
        for var in loop_stmt.loop_vars() {
            self.w.push_str("var ");
            self.write_local_binding_id(var.binding());
            self.write_optional_space();
            self.w.push_str("=");
            self.write_optional_space();
            self.write_local_value(cx, var.initial_value(), InlineContext::None);
            self.w.push_str(";");
            self.write_newline();
        }

        if !loop_stmt.loop_vars().is_empty() {
            self.write_newline();
        }

        match loop_stmt.control() {
            LoopControl::Head(condition) => {
                self.w.push_str("while ");
                self.write_local_value(cx, condition, InlineContext::None);
                self.write_optional_space();
                self.write_block(cx, loop_stmt.block());
            }
            LoopControl::Tail(condition) => {
                // WGSL does not have a `do { ... } while ...` construct, so we write a `loop` with
                // a conditional `break` at the end.

                self.w.push_str("loop");
                self.write_optional_space();

                // Can't use the regular write_block implementation, as we need to add a conditional
                // `break` at the end.
                self.write_block_start();
                self.write_block_content(cx, loop_stmt.block());

                self.write_newline();
                self.w.push_str("if ");
                self.write_local_value(cx, condition, InlineContext::None);
                self.write_optional_space();
                self.w.push_str("{");
                self.increment_indent();
                self.write_newline();
                self.w.push_str("break;");
                self.decrement_indent();
                self.write_newline();
                self.w.push_str("}");

                self.write_block_end();
            }
            LoopControl::Infinite => {
                self.w.push_str("loop");
                self.write_optional_space();
                self.write_block(cx, loop_stmt.block());
            }
        }

        self.write_newline();
    }

    fn write_stmt_return(&mut self, cx: Context, return_stmt: &Return) {
        if let Some(value) = return_stmt.value() {
            self.w.push_str("return ");
            self.write_local_value(cx, value, InlineContext::None);
            self.w.push_str(";");
        } else {
            self.w.push_str("return;");
        }

        self.write_newline();
    }

    fn write_stmt_expr_binding(&mut self, cx: Context, stmt: &ExprBinding) {
        // Only write a binding if the expression is not to be inlined, otherwise don't write
        // anything
        if !self.should_inline(cx, stmt.binding()) {
            self.w.push_str("let ");
            self.write_local_binding_id(stmt.binding());
            self.write_optional_space();
            self.w.push_str("=");
            self.write_optional_space();
            self.write_expression(cx, stmt.expression(), InlineContext::None);
            self.w.push_str(";");
            self.write_newline();
        }
    }

    fn write_stmt_alloca(&mut self, cx: Context, stmt: &Alloca) {
        // In the SCF, an alloca statement is represented by a local-binding of a pointer type,
        // which may be passed around, rereferenced (with an OpPtrElementPtr expression), or
        // dereferenced by a load or store operation. In WGSL we use an uninitialized `var` to
        // represent an alloca statement, which does not produce a binding of a pointer type. An
        // alloca statement is not inlinable into an expression, it may only be interacted with via
        // its binding ID. We can therefore compensate for this discrepancy when we write the
        // binding ID, see `write_local_value`.

        self.w.push_str("var ");
        self.write_local_binding_id(stmt.binding());
        self.w.push_str(":");
        self.write_optional_space();
        self.write_type(cx, stmt.ty());
        self.w.push_str(";");
        self.write_newline();
    }

    fn write_stmt_store(&mut self, cx: Context, store_stmt: &Store) {
        self.write_local_value(
            cx,
            store_stmt.pointer(),
            InlineContext::Deref {
                needs_parens: false,
            },
        );
        self.write_optional_space();
        self.w.push_str("=");
        self.write_optional_space();
        self.write_local_value(cx, store_stmt.value(), InlineContext::None);
        self.w.push_str(";");
        self.write_newline();
    }

    fn write_stmt_call_builtin(&mut self, cx: Context, op_call_builtin: &OpCallBuiltin) {
        self.write_op_call_builtin(cx, op_call_builtin);
        self.w.push_str(";");
        self.write_newline();
    }

    fn write_op_call_builtin(&mut self, cx: Context, op_call_builtin: &OpCallBuiltin) {
        self.w.push_str(op_call_builtin.callee().ident().as_str());
        self.w.push_str("(");

        let last_arg_index = op_call_builtin.arguments().len() - 1;

        for (i, arg) in op_call_builtin.arguments().iter().enumerate() {
            self.write_local_value(cx, *arg, InlineContext::None);

            if i != last_arg_index {
                self.w.push_str(",");
                self.write_optional_space();
            }
        }

        self.w.push_str(")");
    }

    fn write_local_value(
        &mut self,
        cx: Context,
        local_binding: LocalBinding,
        inline_cx: InlineContext,
    ) {
        if self.should_inline(cx, local_binding) {
            let stmt = cx.scf[local_binding].kind().expect_expr_binding();
            let expression = cx.scf[stmt].expect_expr_binding().expression();

            self.write_expression(cx, expression, inline_cx);
        } else {
            if let InlineContext::Deref { needs_parens } = inline_cx {
                if needs_parens {
                    self.w.push_str("(");
                }

                // Add a dereference operator `*`, unless the local-binding references an alloca,
                // in which case we compensate for the discrepancy introduced when writing the
                // alloca statement, see `write_stmt_alloca` for details.
                if !cx.scf[local_binding].kind().is_alloca() {
                    self.w.push_str("*");
                }

                self.write_local_binding_id(local_binding);

                if needs_parens {
                    self.w.push_str(")");
                }
            } else {
                // If the binding represents an alloca statement, then we have to compensate for the
                // discrepancy introduced when writing the alloca statement by adding a referencing
                // operator, unless this value is used in a "rereferencing" operation. In the
                // rereferencing case, the alloca value is being implicitly dereferenced, which
                // already compensates for the alloca discrepancy. See `write_stmt_alloca` for
                // details on the SCF/WGSL alloca discrepancy.
                if cx.scf[local_binding].kind().is_alloca() && inline_cx != InlineContext::Reref {
                    self.w.push_str("&");
                }

                self.write_local_binding_id(local_binding);
            }
        }
    }

    fn should_inline(&mut self, cx: Context, local_binding: LocalBinding) -> bool {
        cx.scf[local_binding].kind().is_expr_binding()
            && cx.local_binding_use_counts[local_binding] < 2
    }

    fn write_expression(&mut self, cx: Context, expression: &Expression, inline_cx: InlineContext) {
        match expression.kind() {
            ExpressionKind::FallbackValue => todo!(),
            ExpressionKind::ConstU32(v) => self.write_u32(*v),
            ExpressionKind::ConstI32(v) => self.write_i32(*v),
            ExpressionKind::ConstF32(v) => self.write_f32(*v),
            ExpressionKind::ConstBool(v) => self.write_bool(*v),
            ExpressionKind::GlobalPtr(ptr) => self.write_expr_global_pointer(ptr, inline_cx),
            ExpressionKind::OpUnary(op) => self.write_expr_op_unary(cx, op),
            ExpressionKind::OpBinary(op) => self.write_expr_op_binary(cx, op, inline_cx),
            ExpressionKind::OpVector(op) => self.write_expr_op_vector(cx, op),
            ExpressionKind::OpMatrix(op) => self.write_expr_op_matrix(cx, op),
            ExpressionKind::OpPtrElementPtr(op) => {
                self.write_expr_op_ptr_element_ptr(cx, op, inline_cx)
            }
            ExpressionKind::OpExtractElement(op) => self.write_expr_op_extract_value(cx, op),
            ExpressionKind::OpLoad(b) => self.write_expr_op_load(cx, *b, inline_cx),
            ExpressionKind::OpCallBuiltin(op) => self.write_op_call_builtin(cx, op),
        }
    }

    fn write_expr_global_pointer(&mut self, expr: &GlobalPtr, inline_cx: InlineContext) {
        match inline_cx {
            InlineContext::None => self.w.push_str("&"),
            InlineContext::Deref { .. } | InlineContext::Reref => (),
            InlineContext::UnOp(_) | InlineContext::BinOp(_, _) | InlineContext::Extract => {
                panic!("cannot not be used without dereferencing")
            }
        }

        match expr {
            GlobalPtr::Uniform(b) => self.write_uniform_binding_id(*b),
            GlobalPtr::Storage(b) => self.write_storage_binding_id(*b),
            GlobalPtr::Workgroup(b) => self.write_workgroup_binding_id(*b),
            GlobalPtr::Constant(c) => self.write_constant_id(*c),
        }
    }

    fn write_expr_op_unary(&mut self, cx: Context, op: &OpUnary) {
        // Unary operators have precedence over all binary operators and evaluate in right-to-left
        // order ("inside out"), which means they never need to be parenthesized.

        match op.operator() {
            UnaryOperator::Not => self.w.push_str("!"),
            UnaryOperator::Neg => self.w.push_str("-"),
        };

        self.write_local_value(cx, op.operand(), InlineContext::None);
    }

    fn write_expr_op_binary(&mut self, cx: Context, op: &OpBinary, inline_cx: InlineContext) {
        let needs_parens = op_binary_needs_parens(op.operator(), inline_cx);

        if needs_parens {
            self.w.push_str("(");
        }

        self.write_local_value(
            cx,
            op.lhs(),
            InlineContext::BinOp(BinOpSide::Left, op.operator()),
        );
        self.write_optional_space();

        use BinaryOperator::*;

        match op.operator() {
            And => self.w.push_str("&&"),
            Or => self.w.push_str("||"),
            Add => self.w.push_str("+"),
            Sub => self.w.push_str("-"),
            Mul => self.w.push_str("*"),
            Div => self.w.push_str("/"),
            Mod => self.w.push_str("%"),
            Shl => self.w.push_str("<<"),
            Shr => self.w.push_str(">>"),
            Eq => self.w.push_str("=="),
            NotEq => self.w.push_str("!="),
            Gt => self.w.push_str(">"),
            GtEq => self.w.push_str(">="),
            Lt => self.w.push_str("<"),
            LtEq => self.w.push_str("<="),
            BitOr => self.w.push_str("|"),
            BitAnd => self.w.push_str("&"),
            BitXor => self.w.push_str("^"),
        }

        self.write_optional_space();
        self.write_local_value(
            cx,
            op.rhs(),
            InlineContext::BinOp(BinOpSide::Right, op.operator()),
        );

        if needs_parens {
            self.w.push_str(")");
        }
    }

    fn write_expr_op_vector(&mut self, cx: Context, expr: &OpVector) {
        self.write_vector_ty(expr.vector_ty());
        self.w.push_str("(");

        let size = expr.vector_ty().size.to_usize();
        let last_index = size - 1;

        for i in 0..size {
            self.write_local_value(cx, expr.elements()[i], InlineContext::None);

            if i != last_index {
                self.w.push_str(",");
                self.write_optional_space();
            }
        }

        self.w.push_str(")");
    }

    fn write_expr_op_matrix(&mut self, cx: Context, expr: &OpMatrix) {
        self.write_matrix_ty(expr.matrix_ty());
        self.w.push_str("(");

        let column_count = expr.matrix_ty().columns.to_usize();
        let last_index = column_count - 1;

        for i in 0..column_count {
            self.write_local_value(cx, expr.columns()[i], InlineContext::None);

            if i != last_index {
                self.w.push_str(",");
                self.write_optional_space();
            }
        }

        self.w.push_str(")");
    }

    fn write_expr_op_ptr_element_ptr(
        &mut self,
        cx: Context,
        expr: &OpPtrElementPtr,
        inline_cx: InlineContext,
    ) {
        match inline_cx {
            InlineContext::None => self.w.push_str("&"),
            InlineContext::Deref { .. } | InlineContext::Reref => (),
            InlineContext::UnOp(_) | InlineContext::BinOp(_, _) | InlineContext::Extract => {
                panic!("cannot not be used without dereferencing")
            }
        }

        let ptr = expr.pointer();

        self.write_local_value(cx, ptr, InlineContext::Reref);

        let ptr_ty = cx.scf[ptr].ty();
        let pointee_ty = cx.module.ty.kind(ptr_ty).expect_ptr();

        self.write_access_chain(cx, pointee_ty, expr.indices().iter().copied());
    }

    fn write_expr_op_extract_value(&mut self, cx: Context, expr: &OpExtractElement) {
        let value = expr.value();

        self.write_local_value(cx, value, InlineContext::Extract);
        self.write_access_chain(cx, cx.scf[value].ty(), expr.indices().iter().copied());
    }

    fn write_access_chain(
        &mut self,
        cx: Context,
        ty: Type,
        mut indices: impl Iterator<Item = LocalBinding>,
    ) {
        if let Some(index) = indices.next() {
            match &*cx.module.ty.kind(ty) {
                TypeKind::Vector(_) => {
                    self.write_array_like_access(cx, index);

                    // Can't project any deeper into a vector type.
                    debug_assert!(indices.next().is_none());
                }
                TypeKind::Matrix(m) => {
                    self.write_array_like_access(cx, index);
                    self.write_access_chain(cx, m.column_ty(), indices);
                }
                TypeKind::Array { element_ty, .. } | TypeKind::Slice { element_ty, .. } => {
                    self.write_array_like_access(cx, index);
                    self.write_access_chain(cx, *element_ty, indices);
                }
                TypeKind::Struct(s) => {
                    let stmt = cx.scf[index].kind().expect_expr_binding();
                    let index = cx.scf[stmt]
                        .expect_expr_binding()
                        .expression()
                        .kind()
                        .expect_const_u32() as usize;

                    self.w.push_str(".");
                    self.write_struct_field_ident(index);

                    self.write_access_chain(cx, s.fields[index].ty, indices);
                }
                TypeKind::Enum(_)
                | TypeKind::Ptr(_)
                | TypeKind::Scalar(_)
                | TypeKind::Atomic(_)
                | TypeKind::Function(_)
                | TypeKind::Predicate
                | TypeKind::Dummy => panic!("type cannot be projected to an element"),
            }
        }
    }

    fn write_expr_op_load(&mut self, cx: Context, binding: LocalBinding, inline_cx: InlineContext) {
        // If the OpLoad is being inlined as the base value of an OpExtractElement expression, then
        // we may need parenthesis, e.g.:
        //
        // ```
        // (*ptr)._1 // *ptr._1 would be incorrect
        // ```
        //
        // The OpLoad's operand may be an expression that is to be inlined. If such an expression is
        // a "referencing" expression (`GlobalPtr` or `OpPtrElementPtr`), the referencing operator
        // `&` and the dereferencing operator `*` cancel out, and we write no operator at all. In
        // this case parentheses are never needed.
        //
        // The OpLoad's operand may also be a local binding for an alloca statement. Due to the
        // "alloca discrepancy" (see `write_stmt_alloca`) we also never write a dereferencing
        // operator `*` in this case, so this case also never needs parentheses.
        let needs_parens = if inline_cx == InlineContext::Extract {
            match cx.scf[binding].kind() {
                LocalBindingKind::Alloca(_) => false,
                LocalBindingKind::ExprBinding(stmt) => {
                    let expr_binding = cx.scf[*stmt].expect_expr_binding();

                    match expr_binding.expression().kind() {
                        ExpressionKind::GlobalPtr(_) | ExpressionKind::OpPtrElementPtr(_) => {
                            !self.should_inline(cx, binding)
                        }
                        _ => true,
                    }
                }
                _ => true,
            }
        } else {
            false
        };

        self.write_local_value(cx, binding, InlineContext::Deref { needs_parens });
    }

    fn write_array_like_access(&mut self, cx: Context, index: LocalBinding) {
        self.w.push_str("[");
        self.write_local_value(cx, index, InlineContext::None);
        self.w.push_str("]");
    }

    fn write_bool(&mut self, value: bool) {
        write!(&mut self.w, "{value}").unwrap();
    }

    fn write_u32(&mut self, value: u32) {
        write!(&mut self.w, "{value}u").unwrap();
    }

    fn write_i32(&mut self, value: i32) {
        if value == i32::MIN {
            // Per a comment in Naga, `-2147483648i` is not a valid WGSL token; `i32::MIN` can only
            // be represented with a negated abstract integer.
            write!(&mut self.w, "i32({value})").unwrap();
        } else {
            write!(&mut self.w, "{value}i").unwrap();
        }
    }

    fn write_f32(&mut self, value: f32) {
        write!(&mut self.w, "{value}f").unwrap();
    }
}

struct ConstantValueWriter<'a, 'b> {
    writer: &'a mut WgslModuleWriter,
    data: &'b [u8],
}

impl ConstantValueWriter<'_, '_> {
    fn write_value(&mut self, cx: Context, ty: Type, offset: usize) {
        match &*cx.module.ty.kind(ty) {
            TypeKind::Scalar(kind) => self.write_scalar_value(*kind, offset),
            TypeKind::Vector(v) => self.write_vector_value(v, offset),
            TypeKind::Matrix(m) => self.write_matrix_value(m, offset),
            TypeKind::Array {
                element_ty,
                count,
                stride,
            } => self.write_array_value(cx, *element_ty, *count, *stride, offset),
            TypeKind::Struct(def) => self.write_struct_value(cx, ty, def, offset),
            TypeKind::Atomic(_)
            | TypeKind::Slice { .. }
            | TypeKind::Enum(_)
            | TypeKind::Ptr(_)
            | TypeKind::Function(_)
            | TypeKind::Predicate
            | TypeKind::Dummy => panic!("not a legal value type in a constant"),
        }
    }

    fn write_scalar_value(&mut self, kind: ScalarKind, offset: usize) {
        match kind {
            ScalarKind::I32 => self.writer.write_i32(self.read_i32(offset)),
            ScalarKind::U32 => self.writer.write_u32(self.read_u32(offset)),
            ScalarKind::F32 => self.writer.write_f32(self.read_f32(offset)),
            ScalarKind::Bool => self.writer.write_bool(self.read_bool(offset)),
        }
    }

    fn read_bool(&self, offset: usize) -> bool {
        self.data[offset] != 0
    }

    fn read_u32(&self, offset: usize) -> u32 {
        let slice = self
            .data
            .get(offset..offset + 4)
            .expect("not enough data available at the `offset` to represent a u32 value");
        let bytes: [u8; 4] = slice.try_into().unwrap();

        u32::from_ne_bytes(bytes)
    }

    fn read_i32(&self, offset: usize) -> i32 {
        let slice = self
            .data
            .get(offset..offset + 4)
            .expect("not enough data available at the `offset` to represent a i32 value");
        let bytes: [u8; 4] = slice.try_into().unwrap();

        i32::from_ne_bytes(bytes)
    }

    fn read_f32(&self, offset: usize) -> f32 {
        let slice = self
            .data
            .get(offset..offset + 4)
            .expect("not enough data available at the `offset` to represent a f32 value");
        let bytes: [u8; 4] = slice.try_into().unwrap();

        f32::from_ne_bytes(bytes)
    }

    fn write_vector_value(&mut self, ty: &ty::Vector, offset: usize) {
        self.writer.write_vector_ty(ty);
        self.writer.w.push_str("(");

        self.write_vector_element_value(ty.scalar, offset, 0);

        for index in 1..ty.size.to_usize() {
            self.writer.w.push_str(",");
            self.writer.write_optional_space();
            self.write_vector_element_value(ty.scalar, offset, index);
        }

        self.writer.w.push_str(")");
    }

    fn write_vector_element_value(
        &mut self,
        scalar_kind: ScalarKind,
        vector_offset: usize,
        index: usize,
    ) {
        match scalar_kind {
            ScalarKind::I32 => {
                let element_offset = vector_offset + index * 4;

                self.writer.write_i32(self.read_i32(element_offset));
            }
            ScalarKind::U32 => {
                let element_offset = vector_offset + index * 4;

                self.writer.write_u32(self.read_u32(element_offset));
            }
            ScalarKind::F32 => {
                let element_offset = vector_offset + index * 4;

                self.writer.write_f32(self.read_f32(element_offset));
            }
            ScalarKind::Bool => {
                let element_offset = vector_offset + index;

                self.writer.write_i32(self.read_i32(element_offset));
            }
        }
    }

    fn write_matrix_value(&mut self, ty: &ty::Matrix, offset: usize) {
        let col_ty = ty.column_vector();

        self.writer.write_matrix_ty(ty);
        self.writer.w.push_str("(");
        self.writer.increment_indent();

        let column_stride = match ty.rows {
            // 2 element vectors will always have a stride of 8 bytes, regardless of the element
            // type.
            VectorSize::Two => 8,
            // both 3 and 4 element vectors will always have a stride of 16 bytes, regardless
            // of the element type.
            VectorSize::Three | VectorSize::Four => 16,
        };

        for index in 0..ty.columns.to_usize() {
            let column_offset = offset + index * column_stride;

            self.writer.write_newline();
            self.write_vector_value(&col_ty, column_offset);
            self.writer.w.push_str(",");
        }

        self.writer.decrement_indent();
        self.writer.write_newline();
        self.writer.w.push_str(")");
    }

    fn write_array_value(
        &mut self,
        cx: Context,
        element_ty: Type,
        count: u64,
        stride: u64,
        offset: usize,
    ) {
        let count = count as usize;
        let stride = stride as usize;

        self.writer.w.push_str("array(");
        self.writer.increment_indent();

        for i in 0..count {
            let element_offset = offset + i * stride;

            self.writer.write_newline();
            self.write_value(cx, element_ty, element_offset);
            self.writer.w.push_str(",");
        }

        self.writer.decrement_indent();
        self.writer.write_newline();
        self.writer.w.push_str(")");
    }

    fn write_struct_value(&mut self, cx: Context, ty: Type, struct_def: &Struct, offset: usize) {
        self.writer.write_struct_id(ty);
        self.writer.write_optional_space();
        self.writer.w.push_str("{");
        self.writer.increment_indent();

        let last_field_index = struct_def.fields.len() - 1;

        for (index, field) in struct_def.fields.iter().enumerate() {
            let field_offset = offset + field.offset as usize;

            self.writer.write_newline();
            self.writer.write_struct_field_ident(index);
            self.writer.w.push_str(":");
            self.writer.write_optional_space();
            self.write_value(cx, field.ty, field_offset);

            if index != last_field_index {
                self.writer.w.push_str(",");
            }
        }

        self.writer.decrement_indent();
        self.writer.write_newline();
        self.writer.w.push_str("}");
    }
}

fn op_binary_needs_parens(operator: BinaryOperator, inline_cx: InlineContext) -> bool {
    use BinOpSide::*;
    use BinaryOperator::*;
    use InlineContext::*;

    // This is based on section 8.19 of the WGSL specification:
    // https://gpuweb.github.io/gpuweb/wgsl/#operator-precedence-associativity

    match (inline_cx, operator) {
        (Deref { .. } | Reref, _) => {
            panic!("inappropriate inline-context for a binary operation")
        }
        (None | Extract, _) => false,

        // A bitwise operator always needs parentheses, except when inlined into an operation with
        // the same operator.
        (BinOp(_, BitOr), BitOr) | (BinOp(_, BitAnd), BitAnd) | (BinOp(_, BitXor), BitXor) => false,

        // A logic operator always needs parentheses, except when inlined into an operation with
        // the same operator.
        (BinOp(_, Or), Or) | (BinOp(_, And), And) => false,

        // The multiplicative operators take precedence over the additive operators, so these cases
        // do not need parentheses.
        (BinOp(_, Add | Sub), Mul | Div | Mod) => false,

        // The multiplicative, additive and shift operators take precedence over the relational
        // operators, so these cases do not need parentheses.
        (BinOp(_, Eq | NotEq | Lt | LtEq | Gt | GtEq), Mul | Div | Mod | Add | Sub | Shl | Shr) => {
            false
        }

        // The multiplicative operators take left-to-right precedence amongst themselves, so if
        // a multiplicative operation is inlined as the left-hand-side of another multiplicative
        // operation, we do not need parentheses.
        (BinOp(Left, Mul | Div | Mod), Mul | Div | Mod) => false,

        // The additive operators take left-to-right precedence amongst themselves, so if an
        // additive operation is inlined as the left-hand-side of another additive operation, we do
        // not need parentheses.
        (BinOp(Left, Add | Sub), Add | Sub) => false,

        // The relational operators take precedence over the logic operators, so these cases do not
        // need parentheses.
        (BinOp(_, Or | And), Eq | NotEq | Lt | LtEq | Gt | GtEq) => false,

        _ => true,
    }
}

pub fn write_wgsl(module: &Module, scf: &Scf) -> String {
    let used_globals = collect_used_global_bindings(module, scf);
    let used_structs = collect_used_structs(module, scf, &used_globals);

    let local_binding_use_counts = count_local_binding_use(
        scf,
        local_binding_use::Config {
            count_const_index_use: false,
        },
    );

    let cx = Context {
        module,
        scf,
        local_binding_use_counts: &local_binding_use_counts,
        used_structs: &used_structs,
        used_workgroup_bindings: &used_globals.workgroup_bindings,
        used_constants: &used_globals.constants,
    };

    let reserved_names = module
        .entry_points
        .iter()
        .map(|(_, e)| e.name.to_string())
        .collect::<FxHashSet<_>>();

    let mut writer = WgslModuleWriter::new(reserved_names);

    writer.write_required_language_extensions();
    writer.write_struct_decls(cx);
    writer.write_constant_bindings(cx);
    writer.write_uniform_bindings(cx);
    writer.write_storage_bindings(cx);
    writer.write_workgroup_bindings(cx);
    writer.write_entry_points(cx);

    writer.w
}
