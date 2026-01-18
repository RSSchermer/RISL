use std::hash::{Hash, Hasher};
use std::ops::Index;

use delegate::delegate;
use indexmap::{IndexMap, IndexSet};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use slotmap::{Key, SlotMap};
use smallvec::{SmallVec, smallvec};

use crate::builtin_function::BuiltinFunction;
use crate::intrinsic::Intrinsic;
use crate::ty::{TY_BOOL, TY_F32, TY_I32, TY_U32, Type, TypeKind, TypeRegistry};
use crate::{
    BinaryOperator, Constant, ConstantRegistry, Function, Module, StorageBinding,
    StorageBindingRegistry, UnaryOperator, UniformBinding, UniformBindingRegistry,
    WorkgroupBinding, WorkgroupBindingRegistry, intrinsic, ty,
};

slotmap::new_key_type! {
    pub struct Statement;
    pub struct Block;
    pub struct LocalBinding;
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct LocalBindingData {
    ty: Type,
    kind: LocalBindingKind,
}

impl LocalBindingData {
    pub fn ty(&self) -> Type {
        self.ty
    }

    pub fn kind(&self) -> &LocalBindingKind {
        &self.kind
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum LocalBindingKind {
    Argument(u32),
    Alloca(Statement),
    ExprBinding(Statement),
    ControlFlowVar {
        statement: Statement,
        out_var: usize,
    },
}

impl LocalBindingKind {
    pub fn is_argument(&self) -> bool {
        matches!(self, LocalBindingKind::Argument(_))
    }

    pub fn expect_argument(&self) -> u32 {
        if let LocalBindingKind::Argument(index) = self {
            *index
        } else {
            panic!("expected an argument");
        }
    }

    pub fn is_alloca(&self) -> bool {
        matches!(self, LocalBindingKind::Alloca(_))
    }

    pub fn expect_alloca(&self) -> Statement {
        if let LocalBindingKind::Alloca(stmt) = self {
            *stmt
        } else {
            panic!("expected an alloca statement");
        }
    }

    pub fn is_expr_binding(&self) -> bool {
        matches!(self, LocalBindingKind::ExprBinding(_))
    }

    pub fn expect_expr_binding(&self) -> Statement {
        if let LocalBindingKind::ExprBinding(binding) = self {
            *binding
        } else {
            panic!("expected an expression-binding statement");
        }
    }

    pub fn is_control_flow_var(&self) -> bool {
        matches!(self, LocalBindingKind::ControlFlowVar { .. })
    }

    pub fn expect_control_flow_var(&self) -> (Statement, usize) {
        if let LocalBindingKind::ControlFlowVar { statement, out_var } = self {
            (*statement, *out_var)
        } else {
            panic!("expected a control-flow variable");
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct IntrinsicOp<T> {
    intrinsic: T,
    arguments: SmallVec<[LocalBinding; 2]>,
}

impl<T> IntrinsicOp<T> {
    pub fn intrinsic(&self) -> &T {
        &self.intrinsic
    }

    pub fn arguments(&self) -> &[LocalBinding] {
        &self.arguments
    }
}

macro_rules! gen_intrinsic_arg_getter {
    ($name:ident, $index:literal) => {
        pub fn $name(&self) -> LocalBinding {
            self.arguments[$index]
        }
    };
}

pub type OpUnary = IntrinsicOp<intrinsic::OpUnary>;

impl OpUnary {
    pub fn operator(&self) -> UnaryOperator {
        self.intrinsic.operator
    }

    gen_intrinsic_arg_getter!(operand, 0);
}

pub type OpBinary = IntrinsicOp<intrinsic::OpBinary>;

impl OpBinary {
    pub fn operator(&self) -> BinaryOperator {
        self.intrinsic.operator
    }

    gen_intrinsic_arg_getter!(lhs, 0);
    gen_intrinsic_arg_getter!(rhs, 1);
}

pub type OpVector = IntrinsicOp<intrinsic::OpVector>;

impl OpVector {
    pub fn vector_ty(&self) -> &ty::Vector {
        &self.intrinsic.ty
    }

    pub fn elements(&self) -> &[LocalBinding] {
        &self.arguments
    }
}

pub type OpMatrix = IntrinsicOp<intrinsic::OpMatrix>;

impl OpMatrix {
    pub fn matrix_ty(&self) -> &ty::Matrix {
        &self.intrinsic.ty
    }

    pub fn columns(&self) -> &[LocalBinding] {
        &self.arguments
    }
}

pub type OpConvertToU32 = IntrinsicOp<intrinsic::OpConvertToU32>;

impl OpConvertToU32 {
    gen_intrinsic_arg_getter!(value, 0);
}

pub type OpConvertToI32 = IntrinsicOp<intrinsic::OpConvertToI32>;

impl OpConvertToI32 {
    gen_intrinsic_arg_getter!(value, 0);
}

pub type OpConvertToF32 = IntrinsicOp<intrinsic::OpConvertToF32>;

impl OpConvertToF32 {
    gen_intrinsic_arg_getter!(value, 0);
}

pub type OpConvertToBool = IntrinsicOp<intrinsic::OpConvertToBool>;

impl OpConvertToBool {
    gen_intrinsic_arg_getter!(value, 0);
}

pub type OpLoad = IntrinsicOp<intrinsic::OpLoad>;

impl OpLoad {
    gen_intrinsic_arg_getter!(ptr, 0);
}

pub type OpStore = IntrinsicOp<intrinsic::OpStore>;

impl OpStore {
    gen_intrinsic_arg_getter!(ptr, 0);
    gen_intrinsic_arg_getter!(value, 1);
}

pub type OpFieldPtr = IntrinsicOp<intrinsic::OpFieldPtr>;

impl OpFieldPtr {
    pub fn field_index(&self) -> u32 {
        self.intrinsic.field_index
    }

    gen_intrinsic_arg_getter!(ptr, 0);
}

pub type OpElementPtr = IntrinsicOp<intrinsic::OpElementPtr>;

impl OpElementPtr {
    gen_intrinsic_arg_getter!(ptr, 0);
    gen_intrinsic_arg_getter!(index, 1);
}

pub type OpExtractField = IntrinsicOp<intrinsic::OpExtractField>;

impl OpExtractField {
    pub fn field_index(&self) -> u32 {
        self.intrinsic.field_index
    }

    gen_intrinsic_arg_getter!(value, 0);
}

pub type OpExtractElement = IntrinsicOp<intrinsic::OpExtractElement>;

impl OpExtractElement {
    gen_intrinsic_arg_getter!(value, 0);
    gen_intrinsic_arg_getter!(index, 1);
}

pub type OpArrayLength = IntrinsicOp<intrinsic::OpArrayLength>;

impl OpArrayLength {
    gen_intrinsic_arg_getter!(ptr, 0);
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub enum GlobalPtr {
    Uniform(UniformBinding),
    Storage(StorageBinding),
    Workgroup(WorkgroupBinding),
    Constant(Constant),
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum ExpressionKind {
    FallbackValue,
    ConstU32(u32),
    ConstI32(i32),
    ConstF32(f32),
    ConstBool(bool),
    GlobalPtr(GlobalPtr),
    OpUnary(OpUnary),
    OpBinary(OpBinary),
    OpVector(OpVector),
    OpMatrix(OpMatrix),
    OpConvertToU32(OpConvertToU32),
    OpConvertToI32(OpConvertToI32),
    OpConvertToF32(OpConvertToF32),
    OpConvertToBool(OpConvertToBool),
    OpFieldPtr(OpFieldPtr),
    OpElementPtr(OpElementPtr),
    OpExtractField(OpExtractField),
    OpExtractElement(OpExtractElement),
    OpLoad(OpLoad),
    OpArrayLength(OpArrayLength),
}

macro_rules! gen_expression_kind_from {
    ($($kind:ident : $kind_ty:ident,)*) => {
        $(
            impl From<$kind_ty> for ExpressionKind {
                fn from(value: $kind_ty) -> Self {
                    Self::$kind(value)
                }
            }
        )*
    }
}

gen_expression_kind_from! {
    ConstU32: u32,
    ConstI32: i32,
    ConstF32: f32,
    ConstBool: bool,
    GlobalPtr: GlobalPtr,
    OpUnary: OpUnary,
    OpBinary: OpBinary,
    OpVector: OpVector,
    OpMatrix: OpMatrix,
    OpConvertToU32: OpConvertToU32,
    OpConvertToI32: OpConvertToI32,
    OpConvertToF32: OpConvertToF32,
    OpConvertToBool: OpConvertToBool,
    OpFieldPtr: OpFieldPtr,
    OpElementPtr: OpElementPtr,
    OpExtractField: OpExtractField,
    OpExtractElement: OpExtractElement,
    OpLoad: OpLoad,
    OpArrayLength: OpArrayLength,
}

macro_rules! gen_expression_kind_is_and_expect {
    ($($name:ident $is_name:ident $expect_name:ident $label:literal,)*) => {
        $(
            pub fn $is_name(&self) -> bool {
                matches!(self, ExpressionKind::$name(_))
            }

            pub fn $expect_name(&self) -> &$name {
                if let ExpressionKind::$name(expr) = self {
                    expr
                } else {
                    panic!("expected a(n) {} expression", $label);
                }
            }
        )*
    }
}

impl ExpressionKind {
    pub fn is_const_u32(&self) -> bool {
        matches!(self, ExpressionKind::ConstU32(_))
    }

    pub fn expect_const_u32(&self) -> u32 {
        if let ExpressionKind::ConstU32(value) = self {
            *value
        } else {
            panic!("expected a constant u32 expression");
        }
    }

    pub fn is_const_i32(&self) -> bool {
        matches!(self, ExpressionKind::ConstI32(_))
    }

    pub fn expect_const_i32(&self) -> i32 {
        if let ExpressionKind::ConstI32(value) = self {
            *value
        } else {
            panic!("expected a constant i32 expression");
        }
    }

    pub fn is_const_f32(&self) -> bool {
        matches!(self, ExpressionKind::ConstF32(_))
    }

    pub fn expect_const_f32(&self) -> f32 {
        if let ExpressionKind::ConstF32(value) = self {
            *value
        } else {
            panic!("expected a constant f32 expression");
        }
    }

    pub fn is_const_bool(&self) -> bool {
        matches!(self, ExpressionKind::ConstBool(_))
    }

    pub fn expect_const_bool(&self) -> bool {
        if let ExpressionKind::ConstBool(value) = self {
            *value
        } else {
            panic!("expected a constant bool expression");
        }
    }

    gen_expression_kind_is_and_expect! {
        GlobalPtr is_global_ptr expect_global_ptr "global-pointer",
        OpUnary is_op_unary expect_op_unary "unary operation",
        OpBinary is_op_binary expect_op_binary "binary operation",
        OpVector is_op_vector expect_op_vector "vector operation",
        OpMatrix is_op_matrix expect_op_matrix "matrix operation",
        OpConvertToU32 is_op_convert_to_u32 expect_op_convert_to_u32 "convert-to-u32 operation",
        OpConvertToI32 is_op_convert_to_i32 expect_op_convert_to_i32 "convert-to-i32 operation",
        OpConvertToF32 is_op_convert_to_f32 expect_op_convert_to_f32 "convert-to-f32 operation",
        OpConvertToBool is_op_convert_to_bool expect_op_convert_to_bool "convert-to-bool operation",
        OpFieldPtr is_op_field_ptr expect_op_field_ptr "field-pointer operation",
        OpElementPtr is_op_element_ptr expect_op_element_ptr "element-pointer operation",
        OpExtractField is_op_extract_field expect_op_extract_field "extract-field operation",
        OpExtractElement is_op_extract_element expect_op_extract_element "extract-element operation",
        OpLoad is_op_load expect_op_load "load operation",
        OpArrayLength is_op_array_length expect_op_array_length "array-length operation",
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Expression {
    ty: Type,
    kind: ExpressionKind,
}

impl Expression {
    pub fn ty(&self) -> Type {
        self.ty
    }

    pub fn kind(&self) -> &ExpressionKind {
        &self.kind
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct BlockData {
    statements: IndexSet<Statement>,
    control_flow_vars: IndexMap<LocalBinding, LocalBinding>,
}

impl BlockData {
    fn new() -> Self {
        Self {
            statements: Default::default(),
            control_flow_vars: Default::default(),
        }
    }

    pub fn statements(&self) -> &IndexSet<Statement> {
        &self.statements
    }

    pub fn control_flow_var(&self, binding: LocalBinding) -> LocalBinding {
        *self
            .control_flow_vars
            .get(&binding)
            .expect("no control-flow variable associated with the local binding")
    }

    pub fn control_flow_var_iter(
        &self,
    ) -> impl Iterator<Item = (LocalBinding, LocalBinding)> + use<'_> {
        self.control_flow_vars
            .iter()
            .map(|(binding, expr)| (*binding, *expr))
    }

    pub fn is_empty(&self) -> bool {
        self.statements.is_empty() && self.control_flow_vars.is_empty()
    }

    fn add_statement(&mut self, position: BlockPosition, statement: Statement) {
        match position {
            BlockPosition::Append => self.append_statement(statement),
            BlockPosition::Prepend => self.prepend_statement(statement),
            BlockPosition::InsertBefore(before) => self.insert_before(before, statement),
            BlockPosition::InsertAfter(after) => self.insert_after(after, statement),
        }
    }

    fn append_statement(&mut self, statement: Statement) {
        self.statements.insert(statement);
    }

    fn prepend_statement(&mut self, statement: Statement) {
        // Note that `IndexSet::insert_before` accepts indices in the `0..=set.len()` range, so
        // this works even if the block is currently empty.
        self.statements.insert_before(0, statement);
    }

    fn insert_before(&mut self, before: Statement, statement: Statement) {
        let Some(index) = self.statements.get_index_of(&before) else {
            panic!("'before' statement not found in the block")
        };

        self.statements.insert_before(index, statement);
    }

    fn insert_after(&mut self, after: Statement, statement: Statement) {
        let Some(index) = self.statements.get_index_of(&after) else {
            panic!("'after' statement not found in the block")
        };

        // Note that `IndexSet::insert_before` accepts indices in the `0..=set.len()` range, so
        // this works even if the `after` statement is the last statement in the block.
        self.statements.insert_before(index + 1, statement);
    }

    fn remove_statement(&mut self, statement: Statement) -> bool {
        self.statements.shift_remove(&statement)
    }

    fn set_control_flow_var(&mut self, index: usize, value: LocalBinding) {
        if let Some(mut entry) = self.control_flow_vars.get_index_entry(index) {
            entry.insert(value);
        } else {
            panic!("no control-flow variable associated with the index");
        }
    }

    fn add_control_flow_var(&mut self, binding: LocalBinding, value: LocalBinding) {
        self.control_flow_vars.insert(binding, value);
    }

    fn remove_control_flow_var(&mut self, binding: LocalBinding) {
        self.control_flow_vars.shift_remove(&binding);
    }
}

#[derive(Clone, Copy, PartialEq, Serialize, Deserialize, Debug)]
pub enum LoopControl {
    Head(LocalBinding),
    Tail(LocalBinding),
    Infinite,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct LoopVar {
    binding: LocalBinding,
    initial_value: LocalBinding,
}

impl LoopVar {
    pub fn binding(&self) -> LocalBinding {
        self.binding
    }

    pub fn initial_value(&self) -> LocalBinding {
        self.initial_value
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Loop {
    loop_block: Block,
    control: LoopControl,
    loop_vars: Vec<LoopVar>,
}

impl Loop {
    pub fn block(&self) -> Block {
        self.loop_block
    }

    pub fn control(&self) -> LoopControl {
        self.control
    }

    pub fn loop_vars(&self) -> &[LoopVar] {
        &self.loop_vars
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct If {
    condition: LocalBinding,
    then_block: Block,
    else_block: Option<Block>,
    out_vars: Vec<LocalBinding>,
}

impl If {
    pub fn condition(&self) -> LocalBinding {
        self.condition
    }

    pub fn then_block(&self) -> Block {
        self.then_block
    }

    pub fn else_block(&self) -> Option<Block> {
        self.else_block
    }

    pub fn out_vars(&self) -> &[LocalBinding] {
        &self.out_vars
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct SwitchCase {
    case: u32,
    block: Block,
}

impl SwitchCase {
    pub fn case(&self) -> u32 {
        self.case
    }

    pub fn block(&self) -> Block {
        self.block
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Switch {
    on: LocalBinding,
    cases: Vec<SwitchCase>,
    default: Block,
    out_vars: Vec<LocalBinding>,
}

impl Switch {
    pub fn on(&self) -> LocalBinding {
        self.on
    }

    pub fn cases(&self) -> &[SwitchCase] {
        &self.cases
    }

    pub fn default(&self) -> Block {
        self.default
    }

    pub fn out_vars(&self) -> &[LocalBinding] {
        &self.out_vars
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Return {
    value: Option<LocalBinding>,
}

impl Return {
    pub fn value(&self) -> Option<LocalBinding> {
        self.value
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ExprBinding {
    binding: LocalBinding,
    expression: Expression,
}

impl ExprBinding {
    pub fn binding(&self) -> LocalBinding {
        self.binding
    }

    pub fn expression(&self) -> &Expression {
        &self.expression
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Alloca {
    binding: LocalBinding,
    ty: Type,
}

impl Alloca {
    pub fn binding(&self) -> LocalBinding {
        self.binding
    }

    pub fn ty(&self) -> Type {
        self.ty
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct StatementData {
    block: Block,
    kind: StatementKind,
}

impl StatementData {
    pub fn block(&self) -> Block {
        self.block
    }

    pub fn kind(&self) -> &StatementKind {
        &self.kind
    }

    delegate! {
        to self.kind {
            pub fn expect_if(&self) -> &If;
            pub fn expect_switch(&self) -> &Switch;
            pub fn expect_loop(&self) -> &Loop;
            pub fn expect_return(&self) -> &Return;
            pub fn expect_expr_binding(&self) -> &ExprBinding;
            pub fn expect_alloca(&self) -> &Alloca;
            pub fn expect_op_store(&self) -> &OpStore;
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum StatementKind {
    If(If),
    Switch(Switch),
    Loop(Loop),
    Return(Return),
    ExprBinding(ExprBinding),
    Alloca(Alloca),
    OpStore(OpStore),
}

macro_rules! gen_statement_kind_from_kind {
    ($($kind:ident),*) => {
        $(
            impl From<$kind> for StatementKind {
                fn from(value: $kind) -> Self {
                    StatementKind::$kind(value)
                }
            }
        )*
    };
}

gen_statement_kind_from_kind!(If, Switch, Loop, Return, ExprBinding, Alloca, OpStore);

impl StatementKind {
    pub fn expect_if(&self) -> &If {
        if let StatementKind::If(stmt) = self {
            stmt
        } else {
            panic!("expected an if statement");
        }
    }

    fn expect_if_mut(&mut self) -> &mut If {
        if let StatementKind::If(stmt) = self {
            stmt
        } else {
            panic!("expected an if statement");
        }
    }

    pub fn expect_switch(&self) -> &Switch {
        if let StatementKind::Switch(stmt) = self {
            stmt
        } else {
            panic!("expected a switch statement");
        }
    }

    fn expect_switch_mut(&mut self) -> &mut Switch {
        if let StatementKind::Switch(stmt) = self {
            stmt
        } else {
            panic!("expected a switch statement");
        }
    }

    pub fn expect_loop(&self) -> &Loop {
        if let StatementKind::Loop(stmt) = self {
            stmt
        } else {
            panic!("expected a loop statement");
        }
    }

    fn expect_loop_mut(&mut self) -> &mut Loop {
        if let StatementKind::Loop(stmt) = self {
            stmt
        } else {
            panic!("expected a loop statement");
        }
    }

    pub fn expect_return(&self) -> &Return {
        if let StatementKind::Return(stmt) = self {
            stmt
        } else {
            panic!("expected a return statement");
        }
    }

    pub fn expect_expr_binding(&self) -> &ExprBinding {
        if let StatementKind::ExprBinding(stmt) = self {
            stmt
        } else {
            panic!("expected an expression-binding statement");
        }
    }

    pub fn expect_alloca(&self) -> &Alloca {
        if let StatementKind::Alloca(stmt) = self {
            stmt
        } else {
            panic!("expected an alloca statement");
        }
    }

    pub fn expect_op_store(&self) -> &OpStore {
        if let StatementKind::OpStore(stmt) = self {
            stmt
        } else {
            panic!("expected a store statement");
        }
    }
}

pub enum BlockPosition {
    Append,
    Prepend,
    InsertBefore(Statement),
    InsertAfter(Statement),
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct FunctionBody {
    block: Block,
    argument_bindings: Vec<LocalBinding>,
}

impl FunctionBody {
    pub fn block(&self) -> Block {
        self.block
    }

    pub fn argument_bindings(&self) -> &[LocalBinding] {
        &self.argument_bindings
    }
}

#[derive(Clone, Deserialize, Debug)]
pub struct ScfData {
    statements: SlotMap<Statement, StatementData>,
    blocks: SlotMap<Block, BlockData>,
    function_bodies: FxHashMap<Function, FunctionBody>,
    local_bindings: SlotMap<LocalBinding, LocalBindingData>,
}

#[derive(Serialize, Debug)]
pub struct Scf {
    #[serde(skip_serializing)]
    ty: TypeRegistry,
    statements: SlotMap<Statement, StatementData>,
    blocks: SlotMap<Block, BlockData>,
    function_bodies: FxHashMap<Function, FunctionBody>,
    local_bindings: SlotMap<LocalBinding, LocalBindingData>,
}

impl Scf {
    pub fn new(type_registry: TypeRegistry) -> Self {
        Self {
            ty: type_registry,
            statements: Default::default(),
            blocks: Default::default(),
            function_bodies: Default::default(),
            local_bindings: Default::default(),
        }
    }

    pub fn from_ty_and_data(ty: TypeRegistry, data: ScfData) -> Self {
        let ScfData {
            statements,
            blocks,
            function_bodies,
            local_bindings,
        } = data;

        Scf {
            ty,
            statements,
            blocks,
            function_bodies,
            local_bindings,
        }
    }

    /// The complete collection of all statements across all functions currently registered with
    /// this SCF representation.
    pub fn statements(&self) -> &SlotMap<Statement, StatementData> {
        &self.statements
    }

    /// The complete collection of all blocks across all functions currently registered with
    /// this SCF representation.
    pub fn blocks(&self) -> &SlotMap<Block, BlockData> {
        &self.blocks
    }

    /// The complete collection of all statements across all functions currently registered with
    /// this SCF representation.
    pub fn local_bindings(&self) -> &SlotMap<LocalBinding, LocalBindingData> {
        &self.local_bindings
    }

    pub fn ty(&self) -> &TypeRegistry {
        &self.ty
    }

    pub fn register_function(&mut self, module: &Module, function: Function) -> &FunctionBody {
        let sig = &module.fn_sigs[function];
        let argument_bindings = sig
            .args
            .iter()
            .enumerate()
            .map(|(i, arg)| {
                self.local_bindings.insert(LocalBindingData {
                    ty: arg.ty,
                    kind: LocalBindingKind::Argument(i as u32),
                })
            })
            .collect();
        let block = self.blocks.insert(BlockData::new());

        self.function_bodies
            .entry(function)
            .or_insert(FunctionBody {
                block,
                argument_bindings,
            })
    }

    pub fn registered_functions(&self) -> impl Iterator<Item = Function> + use<'_> {
        self.function_bodies.keys().copied()
    }

    pub fn get_function_body(&self, function: Function) -> Option<&FunctionBody> {
        self.function_bodies.get(&function)
    }

    pub fn add_bind_fallback_value(
        &mut self,
        block: Block,
        position: BlockPosition,
        ty: Type,
    ) -> (Statement, LocalBinding) {
        let binding = self.local_bindings.insert(LocalBindingData {
            ty,
            // Initialize with a temporary value, remember to adjust after statement initialization.
            kind: LocalBindingKind::ExprBinding(Statement::default()),
        });

        let statement = self.statements.insert(StatementData {
            block,
            kind: StatementKind::ExprBinding(ExprBinding {
                binding,
                expression: Expression {
                    ty,
                    kind: ExpressionKind::FallbackValue,
                },
            }),
        });

        self.blocks[block].add_statement(position, statement);

        // Adjust the temporary value we set above to the actual statement.
        self.local_bindings[binding].kind = LocalBindingKind::ExprBinding(statement);

        (statement, binding)
    }

    pub fn add_bind_const_u32(
        &mut self,
        block: Block,
        position: BlockPosition,
        value: u32,
    ) -> (Statement, LocalBinding) {
        let binding = self.local_bindings.insert(LocalBindingData {
            ty: TY_U32,
            // Initialize with a temporary value, remember to adjust after statement initialization.
            kind: LocalBindingKind::ExprBinding(Statement::default()),
        });

        let statement = self.statements.insert(StatementData {
            block,
            kind: StatementKind::ExprBinding(ExprBinding {
                binding,
                expression: Expression {
                    ty: TY_U32,
                    kind: ExpressionKind::ConstU32(value),
                },
            }),
        });

        self.blocks[block].add_statement(position, statement);

        // Adjust the temporary value we set above to the actual statement.
        self.local_bindings[binding].kind = LocalBindingKind::ExprBinding(statement);

        (statement, binding)
    }

    pub fn add_bind_const_i32(
        &mut self,
        block: Block,
        position: BlockPosition,
        value: i32,
    ) -> (Statement, LocalBinding) {
        let binding = self.local_bindings.insert(LocalBindingData {
            ty: TY_I32,
            // Initialize with a temporary value, remember to adjust after statement initialization.
            kind: LocalBindingKind::ExprBinding(Statement::default()),
        });

        let statement = self.statements.insert(StatementData {
            block,
            kind: StatementKind::ExprBinding(ExprBinding {
                binding,
                expression: Expression {
                    ty: TY_I32,
                    kind: ExpressionKind::ConstI32(value),
                },
            }),
        });

        self.blocks[block].add_statement(position, statement);

        // Adjust the temporary value we set above to the actual statement.
        self.local_bindings[binding].kind = LocalBindingKind::ExprBinding(statement);

        (statement, binding)
    }

    pub fn add_bind_const_f32(
        &mut self,
        block: Block,
        position: BlockPosition,
        value: f32,
    ) -> (Statement, LocalBinding) {
        let binding = self.local_bindings.insert(LocalBindingData {
            ty: TY_F32,
            // Initialize with a temporary value, remember to adjust after statement initialization.
            kind: LocalBindingKind::ExprBinding(Statement::default()),
        });

        let statement = self.statements.insert(StatementData {
            block,
            kind: StatementKind::ExprBinding(ExprBinding {
                binding,
                expression: Expression {
                    ty: TY_F32,
                    kind: ExpressionKind::ConstF32(value),
                },
            }),
        });

        self.blocks[block].add_statement(position, statement);

        // Adjust the temporary value we set above to the actual statement.
        self.local_bindings[binding].kind = LocalBindingKind::ExprBinding(statement);

        (statement, binding)
    }

    pub fn add_bind_const_bool(
        &mut self,
        block: Block,
        position: BlockPosition,
        value: bool,
    ) -> (Statement, LocalBinding) {
        let binding = self.local_bindings.insert(LocalBindingData {
            ty: TY_BOOL,
            // Initialize with a temporary value, remember to adjust after statement initialization.
            kind: LocalBindingKind::ExprBinding(Statement::default()),
        });

        let statement = self.statements.insert(StatementData {
            block,
            kind: StatementKind::ExprBinding(ExprBinding {
                binding,
                expression: Expression {
                    ty: TY_BOOL,
                    kind: ExpressionKind::ConstBool(value),
                },
            }),
        });

        self.blocks[block].add_statement(position, statement);

        // Adjust the temporary value we set above to the actual statement.
        self.local_bindings[binding].kind = LocalBindingKind::ExprBinding(statement);

        (statement, binding)
    }

    pub fn add_bind_uniform_ptr(
        &mut self,
        block: Block,
        position: BlockPosition,
        registry: &UniformBindingRegistry,
        binding: UniformBinding,
    ) -> (Statement, LocalBinding) {
        let ty = registry[binding].ty;
        let ptr_ty = self.ty().register(TypeKind::Ptr(ty));
        let global_ptr = GlobalPtr::Uniform(binding);

        let binding = self.local_bindings.insert(LocalBindingData {
            ty: ptr_ty,
            // Initialize with a temporary value, remember to adjust after statement initialization.
            kind: LocalBindingKind::ExprBinding(Statement::default()),
        });

        let statement = self.statements.insert(StatementData {
            block,
            kind: StatementKind::ExprBinding(ExprBinding {
                binding,
                expression: Expression {
                    ty: ptr_ty,
                    kind: ExpressionKind::GlobalPtr(global_ptr),
                },
            }),
        });

        self.blocks[block].add_statement(position, statement);

        // Adjust the temporary value we set above to the actual statement.
        self.local_bindings[binding].kind = LocalBindingKind::ExprBinding(statement);

        (statement, binding)
    }

    pub fn add_bind_storage_ptr(
        &mut self,
        block: Block,
        position: BlockPosition,
        registry: &StorageBindingRegistry,
        binding: StorageBinding,
    ) -> (Statement, LocalBinding) {
        let ty = registry[binding].ty;
        let ptr_ty = self.ty().register(TypeKind::Ptr(ty));
        let global_ptr = GlobalPtr::Storage(binding);

        let binding = self.local_bindings.insert(LocalBindingData {
            ty: ptr_ty,
            // Initialize with a temporary value, remember to adjust after statement initialization.
            kind: LocalBindingKind::ExprBinding(Statement::default()),
        });

        let statement = self.statements.insert(StatementData {
            block,
            kind: StatementKind::ExprBinding(ExprBinding {
                binding,
                expression: Expression {
                    ty: ptr_ty,
                    kind: ExpressionKind::GlobalPtr(global_ptr),
                },
            }),
        });

        self.blocks[block].add_statement(position, statement);

        // Adjust the temporary value we set above to the actual statement.
        self.local_bindings[binding].kind = LocalBindingKind::ExprBinding(statement);

        (statement, binding)
    }

    pub fn add_bind_workgroup_ptr(
        &mut self,
        block: Block,
        position: BlockPosition,
        registry: &WorkgroupBindingRegistry,
        binding: WorkgroupBinding,
    ) -> (Statement, LocalBinding) {
        let ty = registry[binding].ty;
        let ptr_ty = self.ty().register(TypeKind::Ptr(ty));
        let global_ptr = GlobalPtr::Workgroup(binding);

        let binding = self.local_bindings.insert(LocalBindingData {
            ty: ptr_ty,
            // Initialize with a temporary value, remember to adjust after statement initialization.
            kind: LocalBindingKind::ExprBinding(Statement::default()),
        });

        let statement = self.statements.insert(StatementData {
            block,
            kind: StatementKind::ExprBinding(ExprBinding {
                binding,
                expression: Expression {
                    ty: ptr_ty,
                    kind: ExpressionKind::GlobalPtr(global_ptr),
                },
            }),
        });

        self.blocks[block].add_statement(position, statement);

        // Adjust the temporary value we set above to the actual statement.
        self.local_bindings[binding].kind = LocalBindingKind::ExprBinding(statement);

        (statement, binding)
    }

    pub fn add_bind_constant_ptr(
        &mut self,
        block: Block,
        position: BlockPosition,
        registry: &ConstantRegistry,
        constant: Constant,
    ) -> (Statement, LocalBinding) {
        let ty = registry[constant].ty();
        let ptr_ty = self.ty().register(TypeKind::Ptr(ty));
        let global_ptr = GlobalPtr::Constant(constant);

        let binding = self.local_bindings.insert(LocalBindingData {
            ty: ptr_ty,
            // Initialize with a temporary value, remember to adjust after statement initialization.
            kind: LocalBindingKind::ExprBinding(Statement::default()),
        });

        let statement = self.statements.insert(StatementData {
            block,
            kind: StatementKind::ExprBinding(ExprBinding {
                binding,
                expression: Expression {
                    ty: ptr_ty,
                    kind: ExpressionKind::GlobalPtr(global_ptr),
                },
            }),
        });

        self.blocks[block].add_statement(position, statement);

        // Adjust the temporary value we set above to the actual statement.
        self.local_bindings[binding].kind = LocalBindingKind::ExprBinding(statement);

        (statement, binding)
    }

    pub fn add_bind_intrinsic_op<T>(
        &mut self,
        block: Block,
        position: BlockPosition,
        intrinsic: T,
        arguments: impl IntoIterator<Item = LocalBinding>,
    ) -> (Statement, LocalBinding)
    where
        T: Intrinsic,
        ExpressionKind: From<IntrinsicOp<T>>,
    {
        let arguments: SmallVec<[LocalBinding; 2]> = arguments.into_iter().collect();

        let ty = intrinsic
            .process_args(
                self.ty(),
                arguments.iter().map(|a| self.local_bindings[*a].ty()),
            )
            .unwrap()
            .expect("bindable intrinsics should have a return value");

        let binding = self.local_bindings.insert(LocalBindingData {
            ty,
            // Initialize with a temporary value, remember to adjust after statement initialization.
            kind: LocalBindingKind::ExprBinding(Statement::default()),
        });

        let statement = self.statements.insert(StatementData {
            block,
            kind: StatementKind::ExprBinding(ExprBinding {
                binding,
                expression: Expression {
                    ty,
                    kind: IntrinsicOp {
                        intrinsic,
                        arguments,
                    }
                    .into(),
                },
            }),
        });

        self.blocks[block].add_statement(position, statement);

        // Adjust the temporary value we set above to the actual statement.
        self.local_bindings[binding].kind = LocalBindingKind::ExprBinding(statement);

        (statement, binding)
    }

    pub fn add_bind_op_unary(
        &mut self,
        block: Block,
        position: BlockPosition,
        operator: UnaryOperator,
        operand: LocalBinding,
    ) -> (Statement, LocalBinding) {
        self.add_bind_intrinsic_op(block, position, intrinsic::OpUnary { operator }, [operand])
    }

    pub fn add_bind_op_binary(
        &mut self,
        block: Block,
        position: BlockPosition,
        operator: BinaryOperator,
        lhs: LocalBinding,
        rhs: LocalBinding,
    ) -> (Statement, LocalBinding) {
        self.add_bind_intrinsic_op(
            block,
            position,
            intrinsic::OpBinary { operator },
            [lhs, rhs],
        )
    }

    pub fn add_bind_op_vector(
        &mut self,
        block: Block,
        position: BlockPosition,
        vector_ty: ty::Vector,
        elements: impl IntoIterator<Item = LocalBinding>,
    ) -> (Statement, LocalBinding) {
        self.add_bind_intrinsic_op(
            block,
            position,
            intrinsic::OpVector { ty: vector_ty },
            elements,
        )
    }

    pub fn add_bind_op_matrix(
        &mut self,
        block: Block,
        position: BlockPosition,
        matrix_ty: ty::Matrix,
        columns: impl IntoIterator<Item = LocalBinding>,
    ) -> (Statement, LocalBinding) {
        self.add_bind_intrinsic_op(
            block,
            position,
            intrinsic::OpMatrix { ty: matrix_ty },
            columns,
        )
    }

    pub fn add_bind_op_convert_to_u32(
        &mut self,
        block: Block,
        position: BlockPosition,
        value: LocalBinding,
    ) -> (Statement, LocalBinding) {
        self.add_bind_intrinsic_op(block, position, intrinsic::OpConvertToU32, [value])
    }

    pub fn add_bind_op_convert_to_i32(
        &mut self,
        block: Block,
        position: BlockPosition,
        value: LocalBinding,
    ) -> (Statement, LocalBinding) {
        self.add_bind_intrinsic_op(block, position, intrinsic::OpConvertToI32, [value])
    }

    pub fn add_bind_op_convert_to_f32(
        &mut self,
        block: Block,
        position: BlockPosition,
        value: LocalBinding,
    ) -> (Statement, LocalBinding) {
        self.add_bind_intrinsic_op(block, position, intrinsic::OpConvertToF32, [value])
    }

    pub fn add_bind_op_convert_to_bool(
        &mut self,
        block: Block,
        position: BlockPosition,
        value: LocalBinding,
    ) -> (Statement, LocalBinding) {
        self.add_bind_intrinsic_op(block, position, intrinsic::OpConvertToBool, [value])
    }

    pub fn add_bind_op_field_ptr(
        &mut self,
        block: Block,
        position: BlockPosition,
        pointer: LocalBinding,
        field_index: u32,
    ) -> (Statement, LocalBinding) {
        self.add_bind_intrinsic_op(
            block,
            position,
            intrinsic::OpFieldPtr { field_index },
            [pointer],
        )
    }

    pub fn add_bind_op_element_ptr(
        &mut self,
        block: Block,
        position: BlockPosition,
        pointer: LocalBinding,
        element_index: LocalBinding,
    ) -> (Statement, LocalBinding) {
        self.add_bind_intrinsic_op(
            block,
            position,
            intrinsic::OpElementPtr,
            [pointer, element_index],
        )
    }

    pub fn add_bind_op_extract_field(
        &mut self,
        block: Block,
        position: BlockPosition,
        value: LocalBinding,
        field_index: u32,
    ) -> (Statement, LocalBinding) {
        self.add_bind_intrinsic_op(
            block,
            position,
            intrinsic::OpExtractField { field_index },
            [value],
        )
    }

    pub fn add_bind_op_extract_element(
        &mut self,
        block: Block,
        position: BlockPosition,
        value: LocalBinding,
        element_index: LocalBinding,
    ) -> (Statement, LocalBinding) {
        self.add_bind_intrinsic_op(
            block,
            position,
            intrinsic::OpExtractElement,
            [value, element_index],
        )
    }

    pub fn add_bind_op_load(
        &mut self,
        block: Block,
        position: BlockPosition,
        ptr: LocalBinding,
    ) -> (Statement, LocalBinding) {
        self.add_bind_intrinsic_op(block, position, intrinsic::OpLoad, [ptr])
    }

    pub fn add_bind_op_array_length(
        &mut self,
        block: Block,
        position: BlockPosition,
        ptr: LocalBinding,
    ) -> (Statement, LocalBinding) {
        self.add_bind_intrinsic_op(block, position, intrinsic::OpArrayLength, [ptr])
    }

    pub fn add_if(
        &mut self,
        block: Block,
        position: BlockPosition,
        condition: LocalBinding,
    ) -> (Statement, Block) {
        let then_block = self.blocks.insert(BlockData::new());

        let statement = self.statements.insert(StatementData {
            block,
            kind: StatementKind::If(If {
                condition,
                then_block,
                else_block: None,
                out_vars: vec![],
            }),
        });

        self.blocks[block].add_statement(position, statement);

        (statement, then_block)
    }

    pub fn add_if_out_var(&mut self, if_statement: Statement, ty: Type) -> LocalBinding {
        let stmt = self.statements[if_statement].kind.expect_if_mut();
        let index = stmt.out_vars.len();

        let binding = self.local_bindings.insert(LocalBindingData {
            ty,
            kind: LocalBindingKind::ControlFlowVar {
                statement: Default::default(),
                out_var: index,
            },
        });

        stmt.out_vars.push(binding);

        let then_block = stmt.then_block;
        let else_block = stmt.else_block;

        self.blocks[then_block].add_control_flow_var(binding, LocalBinding::null());

        if let Some(else_block) = else_block {
            self.blocks[else_block].add_control_flow_var(binding, LocalBinding::null());
        }

        binding
    }

    pub fn remove_if_out_var(&mut self, if_statement: Statement, binding: LocalBinding) -> bool {
        let stmt = self.statements[if_statement].kind.expect_if_mut();

        if let Some(index) = stmt.out_vars.iter().position(|b| *b == binding) {
            stmt.out_vars.remove(index);

            self.blocks[stmt.then_block].remove_control_flow_var(binding);

            if let Some(else_block) = stmt.else_block {
                self.blocks[else_block].remove_control_flow_var(binding);
            }

            self.local_bindings.remove(binding);

            true
        } else {
            false
        }
    }

    pub fn add_else_block(&mut self, if_statement: Statement) -> Block {
        let stmt = self.statements[if_statement].kind.expect_if_mut();

        let else_block = self.blocks.insert(BlockData::new());

        let mut case_block_data = BlockData::new();
        let out_var_count = stmt.out_vars.len();

        for i in 0..out_var_count {
            let binding = self.statements[if_statement].kind.expect_if().out_vars[i];

            case_block_data.add_control_flow_var(binding, LocalBinding::null());
        }

        let stmt = self.statements[if_statement].kind.expect_if_mut();

        stmt.else_block = Some(else_block);

        else_block
    }

    pub fn remove_else_block(&mut self, if_statement: Statement) -> bool {
        let stmt = self.statements[if_statement].kind.expect_if_mut();

        if let Some(else_block) = stmt.else_block.take() {
            self.remove_block(else_block);

            true
        } else {
            false
        }
    }

    pub fn add_switch(
        &mut self,
        block: Block,
        position: BlockPosition,
        on: LocalBinding,
    ) -> Statement {
        let default = self.blocks.insert(BlockData::new());

        let statement = self.statements.insert(StatementData {
            block,
            kind: StatementKind::Switch(Switch {
                on,
                cases: vec![],
                default,
                out_vars: vec![],
            }),
        });

        self.blocks[block].add_statement(position, statement);

        statement
    }

    pub fn add_switch_out_var(&mut self, switch_statement: Statement, ty: Type) -> LocalBinding {
        let stmt = self.statements[switch_statement].kind.expect_switch_mut();
        let index = stmt.out_vars.len();

        let binding = self.local_bindings.insert(LocalBindingData {
            ty,
            kind: LocalBindingKind::ControlFlowVar {
                statement: switch_statement,
                out_var: index,
            },
        });

        stmt.out_vars.push(binding);

        let case_count = stmt.cases.len();
        let default = stmt.default;

        for i in 0..case_count {
            let block = self.statements[switch_statement].kind.expect_switch().cases[i].block;

            self.blocks[block].add_control_flow_var(binding, LocalBinding::null());
        }

        self.blocks[default].add_control_flow_var(binding, LocalBinding::null());

        binding
    }

    pub fn remove_switch_out_var(
        &mut self,
        switch_statement: Statement,
        binding: LocalBinding,
    ) -> bool {
        let stmt = self.statements[switch_statement].kind.expect_switch_mut();

        if let Some(index) = stmt.out_vars.iter().position(|b| *b == binding) {
            stmt.out_vars.remove(index);

            for case in &stmt.cases {
                self.blocks[case.block].remove_control_flow_var(binding);
            }

            self.blocks[stmt.default].remove_control_flow_var(binding);
            self.local_bindings.remove(binding);

            true
        } else {
            false
        }
    }

    pub fn add_switch_case(&mut self, switch_statement: Statement, case: u32) -> Block {
        let stmt = self.statements[switch_statement].kind.expect_switch_mut();

        if stmt.cases.iter().any(|c| c.case == case) {
            panic!("switch already covers the given `case`")
        };

        let case_block = self.blocks.insert(BlockData::new());
        let out_var_count = stmt.out_vars.len();

        stmt.cases.push(SwitchCase {
            case,
            block: case_block,
        });

        for i in 0..out_var_count {
            let binding = self.statements[switch_statement]
                .kind
                .expect_switch()
                .out_vars[i];
            let ty = self.local_bindings[binding].ty();

            self.blocks[case_block].add_control_flow_var(binding, LocalBinding::null());
        }

        case_block
    }

    pub fn remove_switch_case(&mut self, switch_statement: Statement, case: u32) -> bool {
        let stmt = self.statements[switch_statement].kind.expect_switch_mut();

        if let Some(index) = stmt.cases.iter().position(|c| c.case == case) {
            let block = stmt.cases[index].block;

            stmt.cases.remove(index);
            self.remove_block(block);

            true
        } else {
            false
        }
    }

    pub fn add_loop(&mut self, block: Block, position: BlockPosition) -> (Statement, Block) {
        let loop_block = self.blocks.insert(BlockData::new());
        let loop_statement = self.statements.insert(StatementData {
            block,
            kind: StatementKind::Loop(Loop {
                loop_block,
                control: LoopControl::Infinite,
                loop_vars: vec![],
            }),
        });

        self.blocks[block].add_statement(position, loop_statement);

        (loop_statement, loop_block)
    }

    pub fn set_loop_control(&mut self, loop_statement: Statement, control: LoopControl) {
        let stmt = self.statements[loop_statement].kind.expect_loop_mut();

        stmt.control = control;
    }

    pub fn add_loop_var(
        &mut self,
        loop_statement: Statement,
        initial_value: LocalBinding,
    ) -> LocalBinding {
        let stmt = self.statements[loop_statement].kind.expect_loop_mut();
        let ty = self.local_bindings[initial_value].ty();
        let index = stmt.loop_vars.len();

        let binding = self.local_bindings.insert(LocalBindingData {
            ty,
            kind: LocalBindingKind::ControlFlowVar {
                statement: loop_statement,
                out_var: index,
            },
        });

        let loop_block = stmt.loop_block;

        stmt.loop_vars.push(LoopVar {
            binding,
            initial_value,
        });

        self.blocks[loop_block].add_control_flow_var(binding, LocalBinding::null());

        binding
    }

    pub fn remove_loop_var(&mut self, loop_statement: Statement, binding: LocalBinding) -> bool {
        let stmt = self.statements[loop_statement].kind.expect_loop_mut();

        if let Some(index) = stmt.loop_vars.iter().position(|c| c.binding == binding) {
            stmt.loop_vars.remove(index);

            let loop_block = stmt.loop_block;

            self.blocks[loop_block].remove_control_flow_var(binding);
            self.local_bindings.remove(binding);

            true
        } else {
            false
        }
    }

    pub fn add_return(
        &mut self,
        block: Block,
        position: BlockPosition,
        value: Option<LocalBinding>,
    ) -> Statement {
        let statement = self.statements.insert(StatementData {
            block,
            kind: StatementKind::Return(Return { value }),
        });

        self.blocks[block].add_statement(position, statement);

        statement
    }

    pub fn add_alloca(
        &mut self,
        block: Block,
        position: BlockPosition,
        ty: Type,
    ) -> (Statement, LocalBinding) {
        let ptr_ty = self.ty().register(TypeKind::Ptr(ty));

        let binding = self.local_bindings.insert(LocalBindingData {
            ty: ptr_ty,
            // Initialize with a temporary value, remember to adjust after statement initialization.
            kind: LocalBindingKind::Alloca(Statement::default()),
        });

        let statement = self.statements.insert(StatementData {
            block,
            kind: StatementKind::Alloca(Alloca { binding, ty }),
        });

        self.blocks[block].add_statement(position, statement);

        // Adjust the temporary value we set above to the actual statement.
        self.local_bindings[binding].kind = LocalBindingKind::Alloca(statement);

        (statement, binding)
    }

    pub fn add_intrinsic_stmt<T>(
        &mut self,
        block: Block,
        position: BlockPosition,
        intrinsic: T,
        arguments: impl IntoIterator<Item = LocalBinding>,
    ) -> Statement
    where
        T: Intrinsic,
        StatementKind: From<IntrinsicOp<T>>,
    {
        let arguments: SmallVec<[LocalBinding; 2]> = arguments.into_iter().collect();

        intrinsic
            .process_args(
                self.ty(),
                arguments.iter().map(|a| self.local_bindings[*a].ty()),
            )
            .unwrap();

        let statement = self.statements.insert(StatementData {
            block,
            kind: IntrinsicOp {
                intrinsic,
                arguments,
            }
            .into(),
        });

        self.blocks[block].add_statement(position, statement);

        statement
    }

    pub fn add_store(
        &mut self,
        block: Block,
        position: BlockPosition,
        pointer: LocalBinding,
        value: LocalBinding,
    ) -> Statement {
        self.add_intrinsic_stmt(block, position, intrinsic::OpStore, [pointer, value])
    }

    pub fn remove_statement(&mut self, statement: Statement) {
        let block = self.statements[statement].block;

        self.blocks[block].remove_statement(statement);
        self.remove_statement_and_bindings(statement);
    }

    pub fn set_control_flow_var(&mut self, block: Block, index: usize, value: LocalBinding) {
        self.blocks[block].set_control_flow_var(index, value);
    }

    fn remove_statement_and_bindings(&mut self, statement: Statement) {
        match self.statements[statement].kind() {
            StatementKind::If(stmt) => {
                for out_var in &stmt.out_vars {
                    self.local_bindings.remove(*out_var);
                }
            }
            StatementKind::Switch(stmt) => {
                for out_var in &stmt.out_vars {
                    self.local_bindings.remove(*out_var);
                }
            }
            StatementKind::Loop(stmt) => {
                for loop_var in &stmt.loop_vars {
                    self.local_bindings.remove(loop_var.binding);
                }
            }
            StatementKind::ExprBinding(stmt) => {
                self.local_bindings.remove(stmt.binding());
            }
            _ => {}
        }

        self.statements.remove(statement);
    }

    fn remove_block(&mut self, block: Block) {
        let stmt_count = self.blocks[block].statements.len();

        for i in 0..stmt_count {
            let stmt = self.blocks[block].statements[i];

            self.remove_statement_and_bindings(stmt);
        }
    }
}

impl Index<LocalBinding> for Scf {
    type Output = LocalBindingData;

    fn index(&self, index: LocalBinding) -> &Self::Output {
        &self.local_bindings[index]
    }
}

impl Index<Statement> for Scf {
    type Output = StatementData;

    fn index(&self, index: Statement) -> &Self::Output {
        &self.statements[index]
    }
}

impl Index<Block> for Scf {
    type Output = BlockData;

    fn index(&self, index: Block) -> &Self::Output {
        &self.blocks[index]
    }
}
