use std::fmt;
use std::ops::Index;

use indexmap::IndexSet;
use ordered_float::OrderedFloat;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use slotmap::SlotMap;
use smallvec::{SmallVec, smallvec};
use thin_vec::{ThinVec, thin_vec};

use crate::builtin_function::BuiltinFunction;
use crate::ty::{TY_BOOL, TY_F32, TY_I32, TY_PREDICATE, TY_U32, Type, TypeKind, TypeRegistry};
use crate::{
    BinaryOperator, Constant, Function, Module, StorageBinding, UnaryOperator, UniformBinding,
    WorkgroupBinding,
};

slotmap::new_key_type! {
    pub struct LocalBinding;
    pub struct BasicBlock;
    pub struct Statement;
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct LocalBindingData {
    owner: Function,
    ty: Type,
}

impl LocalBindingData {
    pub fn owner(&self) -> Function {
        self.owner
    }

    pub fn ty(&self) -> Type {
        self.ty
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct ConstPtr {
    root_identifier: RootIdentifier,
    ty: Type,
}

impl ConstPtr {
    pub fn new(module: &Module, cfg: &Cfg, root_identifier: RootIdentifier) -> Self {
        let pointee_ty = match root_identifier {
            RootIdentifier::Local(b) => cfg.local_bindings[b].ty,
            RootIdentifier::Uniform(b) => module.uniform_bindings[b].ty,
            RootIdentifier::Storage(b) => module.storage_bindings[b].ty,
            RootIdentifier::Workgroup(b) => module.workgroup_bindings[b].ty,
            RootIdentifier::Constant(c) => module.constants[c].ty(),
        };

        let ty = module.ty.register(TypeKind::Ptr(pointee_ty));

        ConstPtr {
            root_identifier,
            ty,
        }
    }

    pub fn local_binding(cfg: &Cfg, local_binding: LocalBinding) -> Self {
        let pointee_ty = cfg[local_binding].ty();
        let ty = cfg.ty().register(TypeKind::Ptr(pointee_ty));

        ConstPtr {
            root_identifier: RootIdentifier::Local(local_binding),
            ty,
        }
    }

    pub fn root_identifier(&self) -> RootIdentifier {
        self.root_identifier
    }

    pub fn ty(&self) -> Type {
        self.ty
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub enum RootIdentifier {
    Local(LocalBinding),
    Uniform(UniformBinding),
    Storage(StorageBinding),
    Workgroup(WorkgroupBinding),
    Constant(Constant),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub enum InlineConst {
    U32(u32),
    I32(i32),
    F32(OrderedFloat<f32>),
    Bool(bool),
    Ptr(ConstPtr),
}

impl InlineConst {
    pub fn ty(&self) -> Type {
        match self {
            InlineConst::U32(_) => TY_U32,
            InlineConst::I32(_) => TY_I32,
            InlineConst::F32(_) => TY_F32,
            InlineConst::Bool(_) => TY_BOOL,
            InlineConst::Ptr(ptr) => ptr.ty(),
        }
    }
}

impl From<f32> for InlineConst {
    fn from(value: f32) -> Self {
        InlineConst::F32(OrderedFloat(value))
    }
}

impl fmt::Display for InlineConst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InlineConst::U32(v) => write!(f, "{}u32", v),
            InlineConst::I32(v) => write!(f, "{}i32", v),
            InlineConst::F32(v) => write!(f, "{}f32", v),
            InlineConst::Bool(v) => write!(f, "{}", v),
            InlineConst::Ptr(_) => write!(f, "ptr"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub enum Value {
    Local(LocalBinding),
    InlineConst(InlineConst),
}

impl Value {
    pub fn is_local(&self) -> bool {
        matches!(self, Value::Local(_))
    }

    pub fn expect_local(&self) -> LocalBinding {
        if let Value::Local(v) = self {
            *v
        } else {
            panic!("expected local value")
        }
    }

    pub fn expect_inline_const(&self) -> InlineConst {
        if let Value::InlineConst(v) = self {
            *v
        } else {
            panic!("expected inline const")
        }
    }
}

impl From<LocalBinding> for Value {
    fn from(value: LocalBinding) -> Self {
        Value::Local(value)
    }
}

impl From<&'_ LocalBinding> for Value {
    fn from(value: &LocalBinding) -> Self {
        Value::Local(*value)
    }
}

impl From<InlineConst> for Value {
    fn from(value: InlineConst) -> Self {
        Value::InlineConst(value)
    }
}

impl From<u32> for Value {
    fn from(value: u32) -> Self {
        InlineConst::U32(value).into()
    }
}

impl From<i32> for Value {
    fn from(value: i32) -> Self {
        InlineConst::I32(value).into()
    }
}

impl From<f32> for Value {
    fn from(value: f32) -> Self {
        InlineConst::F32(OrderedFloat(value)).into()
    }
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        InlineConst::Bool(value).into()
    }
}

impl From<ConstPtr> for Value {
    fn from(ptr: ConstPtr) -> Self {
        InlineConst::Ptr(ptr).into()
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct Bind {
    local_binding: LocalBinding,
    value: Value,
}

impl Bind {
    pub fn local_binding(&self) -> LocalBinding {
        self.local_binding
    }

    pub fn value(&self) -> Value {
        self.value
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct Uninitialized {
    local_binding: LocalBinding,
}

impl Uninitialized {
    pub fn local_binding(&self) -> LocalBinding {
        self.local_binding
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct Assign {
    local_binding: LocalBinding,
    value: Value,
}

impl Assign {
    pub fn local_binding(&self) -> LocalBinding {
        self.local_binding
    }

    pub fn value(&self) -> Value {
        self.value
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpAlloca {
    ty: Type,
    result: LocalBinding,
}

impl OpAlloca {
    pub fn ty(&self) -> Type {
        self.ty
    }

    pub fn result(&self) -> LocalBinding {
        self.result
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpLoad {
    pointer: Value,
    result: LocalBinding,
}

impl OpLoad {
    pub fn pointer(&self) -> Value {
        self.pointer
    }

    pub fn result(&self) -> LocalBinding {
        self.result
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpStore {
    pointer: Value,
    value: Value,
}

impl OpStore {
    pub fn pointer(&self) -> Value {
        self.pointer
    }

    pub fn value(&self) -> Value {
        self.value
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpExtractValue {
    element_ty: Type,
    aggregate: Value,
    indices: ThinVec<Value>,
    result: LocalBinding,
}

impl OpExtractValue {
    pub fn element_ty(&self) -> Type {
        self.element_ty
    }

    pub fn aggregate(&self) -> Value {
        self.aggregate
    }

    pub fn indices(&self) -> &[Value] {
        &self.indices
    }

    pub fn result(&self) -> LocalBinding {
        self.result
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpPtrElementPtr {
    element_ty: Type,
    pointer: Value,
    indices: ThinVec<Value>,
    result: LocalBinding,
}

impl OpPtrElementPtr {
    pub fn element_ty(&self) -> Type {
        self.element_ty
    }

    pub fn pointer(&self) -> Value {
        self.pointer
    }

    pub fn indices(&self) -> &[Value] {
        &self.indices
    }

    pub fn result(&self) -> LocalBinding {
        self.result
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpPtrVariantPtr {
    pointer: Value,
    variant_index: u32,
    result: LocalBinding,
}

impl OpPtrVariantPtr {
    pub fn pointer(&self) -> Value {
        self.pointer
    }

    pub fn variant_index(&self) -> u32 {
        self.variant_index
    }

    pub fn result(&self) -> LocalBinding {
        self.result
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpGetDiscriminant {
    pointer: Value,
    result: LocalBinding,
}

impl OpGetDiscriminant {
    pub fn pointer(&self) -> Value {
        self.pointer
    }

    pub fn result(&self) -> LocalBinding {
        self.result
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpSetDiscriminant {
    pointer: Value,
    variant_index: u32,
}

impl OpSetDiscriminant {
    pub fn pointer(&self) -> Value {
        self.pointer
    }

    pub fn variant_index(&self) -> u32 {
        self.variant_index
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpOffsetSlicePtr {
    pointer: Value,
    offset: Value,
    result: LocalBinding,
}

impl OpOffsetSlicePtr {
    pub fn pointer(&self) -> Value {
        self.pointer
    }

    pub fn offset(&self) -> Value {
        self.offset
    }

    pub fn result(&self) -> LocalBinding {
        self.result
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpUnary {
    operator: UnaryOperator,
    operand: Value,
    result: LocalBinding,
}

impl OpUnary {
    pub fn operator(&self) -> UnaryOperator {
        self.operator
    }

    pub fn operand(&self) -> Value {
        self.operand
    }

    pub fn result(&self) -> LocalBinding {
        self.result
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpBinary {
    operator: BinaryOperator,
    lhs: Value,
    rhs: Value,
    result: LocalBinding,
}

impl OpBinary {
    pub fn operator(&self) -> BinaryOperator {
        self.operator
    }

    pub fn lhs(&self) -> Value {
        self.lhs
    }

    pub fn rhs(&self) -> Value {
        self.rhs
    }

    pub fn result(&self) -> LocalBinding {
        self.result
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpCall {
    callee: Function,
    arguments: ThinVec<Value>,
    result: Option<LocalBinding>,
}

impl OpCall {
    pub fn callee(&self) -> Function {
        self.callee
    }

    pub fn arguments(&self) -> &[Value] {
        &self.arguments
    }

    pub fn result(&self) -> Option<LocalBinding> {
        self.result
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpCallBuiltin {
    callee: BuiltinFunction,
    arguments: ThinVec<Value>,
    result: Option<LocalBinding>,
}

impl OpCallBuiltin {
    pub fn callee(&self) -> &BuiltinFunction {
        &self.callee
    }

    pub fn arguments(&self) -> &[Value] {
        &self.arguments
    }

    pub fn result(&self) -> Option<LocalBinding> {
        self.result
    }
}

/// Converts an integer [value] into a branch selector predicate by comparing it against a list of
/// cases.
///
/// If it matches one case at index `n` in the [cases] list, then the predicate produced will select
/// branch `n`. If it matches multiple cases, then `n` will be the index of the first case matched
/// in list-order. If it matches none of the cases, then the predicate will select branch
/// [cases.len()].
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpCaseToBranchPredicate {
    value: Value,
    cases: Vec<u32>,
    result: LocalBinding,
}

impl OpCaseToBranchPredicate {
    pub fn value(&self) -> Value {
        self.value
    }

    pub fn cases(&self) -> &[u32] {
        &self.cases
    }

    pub fn result(&self) -> LocalBinding {
        self.result
    }
}

/// Converts a boolean [value] into a branch selector predicate.
///
/// If [value] is [true], then the predicate will select branch `0`. If [value] is [false] then the
/// predicate will select branch `1`.
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpBoolToBranchPredicate {
    value: Value,
    result: LocalBinding,
}

impl OpBoolToBranchPredicate {
    pub fn value(&self) -> Value {
        self.value
    }

    pub fn result(&self) -> LocalBinding {
        self.result
    }
}

/// Converts a `u32`, `i32`, `f32`, or `bool` value into a `u32` value.
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpConvertToU32 {
    value: Value,
    result: LocalBinding,
}

impl OpConvertToU32 {
    pub fn value(&self) -> Value {
        self.value
    }

    pub fn result(&self) -> LocalBinding {
        self.result
    }
}

/// Converts a `u32`, `i32`, `f32`, or `bool` value into a `i32` value.
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpConvertToI32 {
    value: Value,
    result: LocalBinding,
}

impl OpConvertToI32 {
    pub fn value(&self) -> Value {
        self.value
    }

    pub fn result(&self) -> LocalBinding {
        self.result
    }
}

/// Converts a `u32`, `i32`, or `f32` value into a `f32` value.
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpConvertToF32 {
    value: Value,
    result: LocalBinding,
}

impl OpConvertToF32 {
    pub fn value(&self) -> Value {
        self.value
    }

    pub fn result(&self) -> LocalBinding {
        self.result
    }
}

/// Converts a `u32`, `i32`, `f32`, or `bool` value into a `bool` value.
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpConvertToBool {
    value: Value,
    result: LocalBinding,
}

impl OpConvertToBool {
    pub fn value(&self) -> Value {
        self.value
    }

    pub fn result(&self) -> LocalBinding {
        self.result
    }
}

macro_rules! gen_statement {
    ($($op:ident,)*) => {
        #[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
        pub enum StatementData {
            $($op($op),)*
        }

        $(impl From<$op> for StatementData {
            fn from(op: $op) -> Self {
                StatementData::$op(op)
            }
        })*
    };
}

gen_statement! {
    Bind,
    Uninitialized,
    Assign,
    OpAlloca,
    OpLoad,
    OpStore,
    OpExtractValue,
    OpPtrElementPtr,
    OpPtrVariantPtr,
    OpGetDiscriminant,
    OpSetDiscriminant,
    OpOffsetSlicePtr,
    OpUnary,
    OpBinary,
    OpCall,
    OpCallBuiltin,
    OpCaseToBranchPredicate,
    OpBoolToBranchPredicate,
    OpConvertToU32,
    OpConvertToI32,
    OpConvertToF32,
    OpConvertToBool,
}

impl StatementData {
    pub fn expect_bind(&self) -> &Bind {
        if let StatementData::Bind(op) = self {
            op
        } else {
            panic!("expected statement to be a bind statement")
        }
    }

    pub fn expect_uninitialized(&self) -> &Uninitialized {
        if let StatementData::Uninitialized(op) = self {
            op
        } else {
            panic!("expected statement to be a uninitialized statement")
        }
    }

    pub fn expect_assign(&self) -> &Assign {
        if let StatementData::Assign(op) = self {
            op
        } else {
            panic!("expected statement to be an assign statement")
        }
    }

    pub fn expect_op_alloca(&self) -> &OpAlloca {
        if let StatementData::OpAlloca(op) = self {
            op
        } else {
            panic!("expected statement to be an alloca operation")
        }
    }

    pub fn expect_op_load(&self) -> &OpLoad {
        if let StatementData::OpLoad(op) = self {
            op
        } else {
            panic!("expected statement to be a load operation")
        }
    }

    pub fn expect_op_store(&self) -> &OpStore {
        if let StatementData::OpStore(op) = self {
            op
        } else {
            panic!("expected statement to be a store operation")
        }
    }

    pub fn expect_op_extract_value(&self) -> &OpExtractValue {
        if let StatementData::OpExtractValue(op) = self {
            op
        } else {
            panic!("expected statement to be a extract-value operation")
        }
    }

    pub fn expect_op_ptr_element_ptr(&self) -> &OpPtrElementPtr {
        if let StatementData::OpPtrElementPtr(op) = self {
            op
        } else {
            panic!("expected statement to be a ptr-element-ptr operation")
        }
    }

    pub fn expect_op_ptr_variant_ptr(&self) -> &OpPtrVariantPtr {
        if let StatementData::OpPtrVariantPtr(op) = self {
            op
        } else {
            panic!("expected statement to be a ptr-variant-ptr operation")
        }
    }

    pub fn expect_op_get_discriminant(&self) -> &OpGetDiscriminant {
        if let StatementData::OpGetDiscriminant(op) = self {
            op
        } else {
            panic!("expected statement to be a get-discriminant operation")
        }
    }

    pub fn expect_op_set_discriminant(&self) -> &OpSetDiscriminant {
        if let StatementData::OpSetDiscriminant(op) = self {
            op
        } else {
            panic!("expected statement to be a set-discriminant operation")
        }
    }

    pub fn expect_op_offset_slice_ptr(&self) -> &OpOffsetSlicePtr {
        if let StatementData::OpOffsetSlicePtr(op) = self {
            op
        } else {
            panic!("expected statement to be an offset-slice-ptr operation")
        }
    }

    pub fn expect_op_unary(&self) -> &OpUnary {
        if let StatementData::OpUnary(op) = self {
            op
        } else {
            panic!("expected statement to be a unary operation")
        }
    }

    pub fn expect_op_binary(&self) -> &OpBinary {
        if let StatementData::OpBinary(op) = self {
            op
        } else {
            panic!("expected statement to be a binary operation")
        }
    }

    pub fn expect_op_call(&self) -> &OpCall {
        if let StatementData::OpCall(op) = self {
            op
        } else {
            panic!("expected statement to be a call operation")
        }
    }

    pub fn expect_op_call_builtin(&self) -> &OpCallBuiltin {
        if let StatementData::OpCallBuiltin(op) = self {
            op
        } else {
            panic!("expected statement to be a call-builtin operation")
        }
    }

    pub fn expect_op_case_to_branch_predicate(&self) -> &OpCaseToBranchPredicate {
        if let StatementData::OpCaseToBranchPredicate(op) = self {
            op
        } else {
            panic!("expected statement to be a case-to-branch-predicate operation")
        }
    }

    pub fn expect_op_bool_to_branch_predicate(&self) -> &OpBoolToBranchPredicate {
        if let StatementData::OpBoolToBranchPredicate(op) = self {
            op
        } else {
            panic!("expected statement to be a bool-to-branch-predicate operation")
        }
    }

    pub fn expect_op_convert_to_u32(&self) -> &OpConvertToU32 {
        if let StatementData::OpConvertToU32(op) = self {
            op
        } else {
            panic!("expected statement to be a `convert-to-u32` operation")
        }
    }

    pub fn expect_op_convert_to_i32(&self) -> &OpConvertToI32 {
        if let StatementData::OpConvertToI32(op) = self {
            op
        } else {
            panic!("expected statement to be a `convert-to-i32` operation")
        }
    }

    pub fn expect_op_convert_to_f32(&self) -> &OpConvertToF32 {
        if let StatementData::OpConvertToF32(op) = self {
            op
        } else {
            panic!("expected statement to be a `convert-to-f32` operation")
        }
    }

    pub fn expect_op_convert_to_bool(&self) -> &OpConvertToBool {
        if let StatementData::OpConvertToBool(op) = self {
            op
        } else {
            panic!("expected statement to be a `convert-to-bool` operation")
        }
    }
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize, Default, Debug)]
pub struct Branch {
    selector: Option<LocalBinding>,
    targets: SmallVec<[BasicBlock; 2]>,
}

impl Branch {
    pub fn single(dest: BasicBlock) -> Self {
        Branch {
            selector: None,
            targets: smallvec![dest],
        }
    }

    pub fn multiple(selector: LocalBinding, targets: impl IntoIterator<Item = BasicBlock>) -> Self {
        Branch {
            selector: Some(selector),
            targets: targets.into_iter().collect(),
        }
    }

    /// A local value that selects the branch.
    ///
    /// If `None` then the first branch is selected.
    pub fn selector(&self) -> Option<LocalBinding> {
        self.selector
    }

    pub fn targets(&self) -> &[BasicBlock] {
        &self.targets
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub enum Terminator {
    Branch(Branch),
    Return(Option<Value>),
}

impl Terminator {
    pub fn branch_single(dest: BasicBlock) -> Self {
        Terminator::Branch(Branch::single(dest))
    }

    pub fn branch_multiple(
        selector: LocalBinding,
        branches: impl IntoIterator<Item = BasicBlock>,
    ) -> Self {
        Terminator::Branch(Branch::multiple(selector, branches))
    }

    pub fn return_value(value: Value) -> Self {
        Terminator::Return(Some(value))
    }

    pub fn return_void() -> Self {
        Terminator::Return(None)
    }

    pub fn is_return(&self) -> bool {
        matches!(self, Terminator::Return(_))
    }

    pub fn expect_return(&self) -> &Option<Value> {
        if let Terminator::Return(v) = self {
            v
        } else {
            panic!("expected terminator to return")
        }
    }

    fn expect_return_mut(&mut self) -> &mut Option<Value> {
        if let Terminator::Return(v) = self {
            v
        } else {
            panic!("expected terminator to return")
        }
    }

    pub fn is_branch(&self) -> bool {
        matches!(self, Terminator::Branch(_))
    }

    pub fn expect_branch(&self) -> &Branch {
        if let Terminator::Branch(b) = self {
            b
        } else {
            panic!("expected terminator to branch")
        }
    }

    fn expect_branch_mut(&mut self) -> &mut Branch {
        if let Terminator::Branch(b) = self {
            b
        } else {
            panic!("expected terminator to branch")
        }
    }

    fn get_or_make_branch(&mut self) -> &mut Branch {
        match self {
            Terminator::Branch(b) => b,
            terminator @ Terminator::Return(_) => {
                *terminator = Terminator::Branch(Branch::default());

                terminator.expect_branch_mut()
            }
        }
    }
}

impl Default for Terminator {
    fn default() -> Self {
        Terminator::Return(None)
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct BasicBlockData {
    owner: Function,
    statements: IndexSet<Statement>,
    terminator: Terminator,
}

impl BasicBlockData {
    pub fn owner(&self) -> Function {
        self.owner
    }

    pub fn statements(&self) -> &IndexSet<Statement> {
        &self.statements
    }

    pub fn terminator(&self) -> &Terminator {
        &self.terminator
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
}

#[derive(Clone, Copy, PartialEq, Serialize, Deserialize, Default, Debug)]
pub enum BlockPosition {
    #[default]
    Append,
    Prepend,
    InsertBefore(Statement),
    InsertAfter(Statement),
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct FunctionBody {
    local_bindings: Vec<LocalBinding>,
    ret_ty: Option<Type>,
    arg_count: usize,
    basic_blocks: IndexSet<BasicBlock>,
    entry: BasicBlock,
}

impl FunctionBody {
    pub fn return_ty(&self) -> Option<Type> {
        self.ret_ty
    }

    pub fn local_bindings(&self) -> &[LocalBinding] {
        &self.local_bindings
    }

    pub fn argument_values(&self) -> &[LocalBinding] {
        &self.local_bindings[0..self.arg_count]
    }

    pub fn basic_blocks(&self) -> &IndexSet<BasicBlock> {
        &self.basic_blocks
    }

    pub fn entry_block(&self) -> BasicBlock {
        self.entry
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CfgData {
    function_body_map: FxHashMap<Function, FunctionBody>,
    local_bindings: SlotMap<LocalBinding, LocalBindingData>,
    basic_blocks: SlotMap<BasicBlock, BasicBlockData>,
    statements: SlotMap<Statement, StatementData>,
}

#[derive(Serialize, Debug)]
pub struct Cfg {
    #[serde(skip_serializing)]
    ty: TypeRegistry,
    function_body_map: FxHashMap<Function, FunctionBody>,
    local_bindings: SlotMap<LocalBinding, LocalBindingData>,
    basic_blocks: SlotMap<BasicBlock, BasicBlockData>,
    statements: SlotMap<Statement, StatementData>,
}

impl Cfg {
    pub fn new(ty: TypeRegistry) -> Self {
        Self {
            ty,
            function_body_map: Default::default(),
            local_bindings: Default::default(),
            basic_blocks: Default::default(),
            statements: Default::default(),
        }
    }

    pub fn from_ty_and_data(ty: TypeRegistry, data: CfgData) -> Self {
        let CfgData {
            function_body_map,
            local_bindings,
            basic_blocks,
            statements,
        } = data;

        Cfg {
            ty,
            function_body_map,
            local_bindings,
            basic_blocks,
            statements,
        }
    }

    pub fn ty(&self) -> &TypeRegistry {
        &self.ty
    }

    pub fn value_ty(&self, value: &Value) -> Type {
        match value {
            Value::Local(v) => self.local_bindings[*v].ty,
            Value::InlineConst(v) => v.ty(),
        }
    }

    pub fn register_function(&mut self, module: &Module, function: Function) -> &FunctionBody {
        let sig = module
            .fn_sigs
            .get(function)
            .expect("function not registered with module");

        let mut local_bindings = Vec::with_capacity(sig.args.len());

        for arg in &sig.args {
            let value = self.add_local_binding(function, arg.ty);

            local_bindings.push(value);
        }

        let entry = self.basic_blocks.insert(BasicBlockData {
            owner: function,
            statements: IndexSet::new(),
            terminator: Default::default(),
        });

        let mut basic_blocks = IndexSet::new();

        basic_blocks.insert(entry);

        self.function_body_map.insert(
            function,
            FunctionBody {
                local_bindings,
                ret_ty: sig.ret_ty,
                arg_count: sig.args.len(),
                basic_blocks,
                entry,
            },
        );

        self.get_function_body(function)
            .expect("we just registered the function")
    }

    pub fn get_function_body(&self, function: Function) -> Option<&FunctionBody> {
        self.function_body_map.get(&function)
    }

    pub fn registered_functions(&self) -> impl Iterator<Item = Function> + '_ {
        self.function_body_map.keys().copied()
    }

    pub fn add_basic_block(&mut self, function: Function) -> BasicBlock {
        let bb = self.basic_blocks.insert(BasicBlockData {
            owner: function,
            statements: IndexSet::new(),
            terminator: Default::default(),
        });

        self.function_mut(function).basic_blocks.insert(bb);

        bb
    }

    pub fn remove_basic_block(&mut self, bb: BasicBlock) {
        let function = self.basic_blocks[bb].owner;
        let body = self.function_mut(function);

        if bb == body.entry {
            panic!("cannot remove entry block");
        }

        body.basic_blocks.shift_remove(&bb);

        for statement in self.basic_blocks[bb].statements() {
            self.statements.remove(*statement);
        }

        self.basic_blocks.remove(bb);
    }

    pub fn make_entry_block(&mut self, bb: BasicBlock) {
        let function = self.basic_blocks[bb].owner;
        let body = self.function_mut(function);

        body.entry = bb;
    }

    pub fn set_terminator(&mut self, bb: BasicBlock, terminator: Terminator) {
        let owner = self.basic_blocks[bb].owner;

        match &terminator {
            Terminator::Branch(branch) => {
                if let Some(selector) = branch.selector {
                    assert_eq!(
                        self.local_bindings[selector].owner, owner,
                        "branch selector must belong to the basic-block's owner function"
                    );

                    for branch in &branch.targets {
                        assert_eq!(
                            self.basic_blocks[*branch].owner, owner,
                            "branch destination must belong to the basic-block's owner function"
                        );
                    }
                }
            }
            Terminator::Return(value) => {
                let ret_ty = self.function_mut(owner).ret_ty;

                match (ret_ty, value) {
                    (Some(ret_ty), Some(value)) => {
                        let value_ty = self.value_ty(&value);

                        assert_eq!(
                            value_ty,
                            ret_ty,
                            "the returned value's type (`{}`) must be the same as the function's \
                            return type (`{}`)",
                            value_ty.to_string(self.ty()),
                            ret_ty.to_string(self.ty()),
                        );
                    }
                    (Some(_), None) => {
                        panic!("function must have a return value");
                    }
                    (None, Some(_)) => {
                        panic!("function does not have a return value");
                    }
                    (None, None) => {}
                }
            }
        }

        self.basic_blocks[bb].terminator = terminator;
    }

    pub fn set_branch_selector(&mut self, bb: BasicBlock, selector: LocalBinding) {
        self.basic_blocks[bb]
            .terminator
            .get_or_make_branch()
            .selector = Some(selector);
    }

    pub fn add_branch_target(&mut self, bb: BasicBlock, target: BasicBlock) {
        let bb_data = self.basic_blocks[bb].terminator.expect_branch_mut();

        bb_data.targets.push(target);
    }

    pub fn replace_branch_target(
        &mut self,
        bb: BasicBlock,
        old_target: BasicBlock,
        new_target: BasicBlock,
    ) -> bool {
        let bb_data = self.basic_blocks[bb].terminator.expect_branch_mut();

        let mut old_target_found = false;

        for branch in &mut bb_data.targets {
            if *branch == old_target {
                *branch = new_target;
                old_target_found = true;
            }
        }

        old_target_found
    }

    pub fn add_stmt_bind(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        value: Value,
    ) -> (Statement, LocalBinding) {
        let owner = self.basic_blocks[bb].owner;

        self.validate_value(owner, &value, "value");

        let ty = self.value_ty(&value);
        let local_binding = self.add_local_binding(owner, ty);

        let stmt = self.statements.insert(
            Bind {
                local_binding,
                value,
            }
            .into(),
        );

        self.basic_blocks[bb].add_statement(position, stmt);

        (stmt, local_binding)
    }

    pub fn add_stmt_uninitialized(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        ty: Type,
    ) -> (Statement, LocalBinding) {
        let owner = self.basic_blocks[bb].owner;

        let local_binding = self.add_local_binding(owner, ty);

        let stmt = self
            .statements
            .insert(Uninitialized { local_binding }.into());

        self.basic_blocks[bb].add_statement(position, stmt);

        (stmt, local_binding)
    }

    pub fn add_stmt_assign(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        local_binding: LocalBinding,
        value: Value,
    ) -> Statement {
        let owner = self.basic_blocks[bb].owner;

        assert_eq!(
            self.local_bindings[local_binding].owner, owner,
            "local binding must belong to the basic-block's function"
        );

        self.validate_value(owner, &value, "value");

        let stmt = self.statements.insert(
            Assign {
                local_binding,
                value,
            }
            .into(),
        );

        self.basic_blocks[bb].add_statement(position, stmt);

        stmt
    }

    pub fn add_stmt_op_alloca(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        ty: Type,
    ) -> (Statement, LocalBinding) {
        let function = self.basic_blocks[bb].owner;
        let result_ty = self.ty.register(TypeKind::Ptr(ty));
        let result = self.add_local_binding(function, result_ty);
        let stmt = self.statements.insert(OpAlloca { ty, result }.into());

        self.basic_blocks[bb].add_statement(position, stmt);

        (stmt, result)
    }

    pub fn add_stmt_op_load(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        pointer: Value,
    ) -> (Statement, LocalBinding) {
        let owner = self.basic_blocks[bb].owner;

        self.validate_value(owner, &pointer, "pointer");

        let pointer_ty = self.value_ty(&pointer);

        let TypeKind::Ptr(pointee_ty) = *self.ty.kind(pointer_ty) else {
            panic!("expected pointer argument to have a pointer type")
        };

        let result = self.add_local_binding(owner, pointee_ty);

        let stmt = self.statements.insert(
            OpLoad {
                pointer: pointer,
                result,
            }
            .into(),
        );

        self.basic_blocks[bb].add_statement(position, stmt);

        (stmt, result)
    }

    pub fn add_stmt_op_store(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        pointer: Value,
        value: Value,
    ) -> Statement {
        let owner = self.basic_blocks[bb].owner;

        self.validate_value(owner, &pointer, "pointer");
        self.validate_value(owner, &value, "value");

        let stmt = self.statements.insert(
            OpStore {
                pointer: pointer,
                value,
            }
            .into(),
        );

        self.basic_blocks[bb].add_statement(position, stmt);

        stmt
    }

    pub fn add_stmt_op_extract_value(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        aggregate: Value,
        indices: impl IntoIterator<Item = Value>,
    ) -> (Statement, LocalBinding) {
        let owner = self.basic_blocks[bb].owner;

        self.validate_value(owner, &aggregate, "aggregate");

        let aggregate_ty = self.value_ty(&aggregate);

        assert!(
            self.ty.kind(aggregate_ty).is_aggregate(),
            "expected aggregate argument to have an aggregate type"
        );

        let mut element_ty = aggregate_ty;
        let mut collected_indices = indices.into_iter().collect::<ThinVec<_>>();

        if collected_indices.is_empty() {
            panic!("expected at least one index");
        }

        for (i, value) in collected_indices.iter().enumerate() {
            self.validate_value(owner, &value, &format!("index {}", i));
            element_ty = self.project_ty((i, value), element_ty);
        }

        let result = self.add_local_binding(owner, element_ty);

        let stmt = self.statements.insert(
            OpExtractValue {
                element_ty,
                aggregate,
                indices: collected_indices,
                result,
            }
            .into(),
        );

        self.basic_blocks[bb].add_statement(position, stmt);

        (stmt, result)
    }

    pub fn add_stmt_op_ptr_element_ptr(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        pointer: Value,
        indices: impl IntoIterator<Item = Value>,
    ) -> (Statement, LocalBinding) {
        let owner = self.basic_blocks[bb].owner;

        self.validate_value(owner, &pointer, "pointer");

        let pointer_ty = self.value_ty(&pointer);

        let TypeKind::Ptr(pointee_ty) = *self.ty.kind(pointer_ty) else {
            panic!("expected pointer argument to have a pointer type");
        };

        let mut element_ty = pointee_ty;
        let mut collected_indices = indices.into_iter().collect::<ThinVec<_>>();

        if collected_indices.is_empty() {
            panic!("expected at least one index");
        }

        for (i, value) in collected_indices.iter().enumerate() {
            self.validate_value(owner, value, &format!("index {}", i));
            element_ty = self.project_ty((i, value), element_ty);
        }

        let result_ty = self.ty.register(TypeKind::Ptr(element_ty));
        let result = self.add_local_binding(owner, result_ty);

        let stmt = self.statements.insert(
            OpPtrElementPtr {
                element_ty,
                pointer,
                indices: collected_indices,
                result,
            }
            .into(),
        );

        self.basic_blocks[bb].add_statement(position, stmt);

        (stmt, result)
    }

    pub fn add_stmt_op_ptr_variant_ptr(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        pointer: Value,
        variant_index: u32,
    ) -> (Statement, LocalBinding) {
        let owner = self.basic_blocks[bb].owner;

        self.validate_value(owner, &pointer, "pointer");

        let pointer_ty = self.value_ty(&pointer);

        let TypeKind::Ptr(pointee_ty) = *self.ty.kind(pointer_ty) else {
            panic!(
                "expected pointer argument to have a pointer type (got `{}`)",
                pointer_ty.to_string(self.ty())
            );
        };

        let TypeKind::Enum(enum_ty) = &*self.ty.kind(pointee_ty) else {
            panic!(
                "expected pointer argument to point to an enum (got `{}`)",
                pointer_ty.to_string(self.ty())
            );
        };

        let Some(variant_ty) = enum_ty.variants.get(variant_index as usize).copied() else {
            panic!("enum does not have a variant `{}`", variant_index);
        };

        let result_ty = self.ty.register(TypeKind::Ptr(variant_ty));
        let result = self.add_local_binding(owner, result_ty);

        let stmt = self.statements.insert(
            OpPtrVariantPtr {
                pointer,
                variant_index,
                result,
            }
            .into(),
        );

        self.basic_blocks[bb].add_statement(position, stmt);

        (stmt, result)
    }

    pub fn add_stmt_op_get_discriminant(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        pointer: Value,
    ) -> (Statement, LocalBinding) {
        let owner = self.basic_blocks[bb].owner;

        self.validate_value(owner, &pointer, "pointer");

        let pointer_ty = self.value_ty(&pointer);

        let TypeKind::Ptr(pointee_ty) = *self.ty.kind(pointer_ty) else {
            panic!("expected pointer argument to have a pointer type");
        };

        assert!(
            self.ty.kind(pointee_ty).is_enum(),
            "expected pointer argument to point to an enum"
        );

        let result = self.add_local_binding(owner, TY_U32);

        let stmt = self
            .statements
            .insert(OpGetDiscriminant { pointer, result }.into());

        self.basic_blocks[bb].add_statement(position, stmt);

        (stmt, result)
    }

    pub fn add_stmt_op_set_discriminant(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        pointer: Value,
        variant_index: u32,
    ) -> Statement {
        let owner = self.basic_blocks[bb].owner;

        self.validate_value(owner, &pointer, "pointer");

        let pointer_ty = self.value_ty(&pointer);

        let TypeKind::Ptr(pointee_ty) = *self.ty.kind(pointer_ty) else {
            panic!("expected pointer argument to have a pointer type");
        };

        let TypeKind::Enum(enum_ty) = &*self.ty.kind(pointee_ty) else {
            panic!("expected pointer argument to point to an enum");
        };

        assert!(
            (variant_index as usize) < enum_ty.variants.len(),
            "enum does not have a variant `{}`",
            variant_index
        );

        let stmt = self.statements.insert(
            OpSetDiscriminant {
                pointer,
                variant_index,
            }
            .into(),
        );

        self.basic_blocks[bb].add_statement(position, stmt);

        stmt
    }

    pub fn add_stmt_op_offset_slice_pointer(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        pointer: Value,
        offset: Value,
    ) -> (Statement, LocalBinding) {
        let owner = self.basic_blocks[bb].owner;

        self.validate_value(owner, &pointer, "pointer");
        self.validate_value(owner, &offset, "offset");

        let pointer_ty = self.value_ty(&pointer);

        let TypeKind::Ptr(pointee_ty) = *self.ty.kind(pointer_ty) else {
            panic!("expected pointer argument to have a pointer type");
        };

        assert_eq!(self.value_ty(&offset), TY_U32, "offset must be an u32");

        let (element_ty, stride) = match *self.ty.kind(pointee_ty) {
            TypeKind::Array {
                element_ty, stride, ..
            }
            | TypeKind::Slice { element_ty, stride } => (element_ty, stride),
            _ => panic!("expected pointer argument to point to an array or slice"),
        };

        let slice_ty = self.ty.register(TypeKind::Slice { element_ty, stride });
        let result_ty = self.ty.register(TypeKind::Ptr(slice_ty));
        let result = self.add_local_binding(owner, result_ty);

        let stmt = self.statements.insert(
            OpOffsetSlicePtr {
                pointer,
                offset,
                result,
            }
            .into(),
        );

        self.basic_blocks[bb].add_statement(position, stmt);

        (stmt, result)
    }

    pub fn add_stmt_op_unary(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        operator: UnaryOperator,
        operand: Value,
    ) -> (Statement, LocalBinding) {
        let owner = self.basic_blocks[bb].owner;

        self.validate_value(owner, &operand, "value");

        let value_ty = self.value_ty(&operand);

        let result = self.add_local_binding(owner, value_ty);

        let stmt = self.statements.insert(
            OpUnary {
                operator,
                operand,
                result,
            }
            .into(),
        );

        self.basic_blocks[bb].add_statement(position, stmt);

        (stmt, result)
    }

    pub fn add_stmt_op_binary(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        operator: BinaryOperator,
        lhs: Value,
        rhs: Value,
    ) -> (Statement, LocalBinding) {
        let owner = self.basic_blocks[bb].owner;

        self.validate_value(owner, &lhs, "LHS-value");
        self.validate_value(owner, &rhs, "RHS-value");

        let output_ty =
            match self
                .ty()
                .check_binary_op(operator, self.value_ty(&lhs), self.value_ty(&rhs))
            {
                Ok(ty) => ty,
                Err(err) => panic!("invalid operation: {}", err),
            };

        let result = self.add_local_binding(owner, output_ty);

        let stmt = self.statements.insert(
            OpBinary {
                operator,
                lhs,
                rhs,
                result,
            }
            .into(),
        );

        self.basic_blocks[bb].add_statement(position, stmt);

        (stmt, result)
    }

    pub fn add_stmt_op_call(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        callee: Function,
        ret_ty: Option<Type>,
        arguments: impl IntoIterator<Item = Value>,
    ) -> (Statement, Option<LocalBinding>) {
        let owner = self.basic_blocks[bb].owner;

        let mut collected_args = thin_vec![];

        for (i, arg) in arguments.into_iter().enumerate() {
            self.validate_value(owner, &arg, &format!("argument {}", i));

            collected_args.push(arg);
        }

        let result = ret_ty.map(|ty| self.add_local_binding(owner, ty));

        let stmt = self.statements.insert(
            OpCall {
                callee,
                arguments: collected_args,
                result,
            }
            .into(),
        );

        self.basic_blocks[bb].add_statement(position, stmt);

        (stmt, result)
    }

    pub fn add_stmt_op_call_builtin(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        callee: BuiltinFunction,
        arguments: impl IntoIterator<Item = Value>,
    ) -> (Statement, Option<LocalBinding>) {
        let owner = self.basic_blocks[bb].owner;

        let mut collected_args = thin_vec![];

        for (i, arg) in arguments.into_iter().enumerate() {
            self.validate_value(owner, &arg, &format!("argument {}", i));
            assert_eq!(
                self.value_ty(&arg),
                callee.arguments()[i],
                "argument {} does not match the expected type",
                i
            );

            collected_args.push(arg);
        }

        let result = callee
            .return_type()
            .map(|ty| self.add_local_binding(owner, ty));

        let stmt = self.statements.insert(
            OpCallBuiltin {
                callee,
                arguments: collected_args,
                result,
            }
            .into(),
        );

        self.basic_blocks[bb].add_statement(position, stmt);

        (stmt, result)
    }

    pub fn add_stmt_op_case_to_branch_predicate(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        case: Value,
        cases: impl IntoIterator<Item = u32>,
    ) -> (Statement, LocalBinding) {
        let owner = self.basic_blocks[bb].owner;

        self.validate_value(owner, &case, "case");
        assert_eq!(self.value_ty(&case), TY_U32, "case must be a u32");

        let cases = cases.into_iter().collect();
        let result = self.add_local_binding(owner, TY_PREDICATE);

        let stmt = self.statements.insert(
            OpCaseToBranchPredicate {
                value: case,
                cases,
                result,
            }
            .into(),
        );

        self.basic_blocks[bb].add_statement(position, stmt);

        (stmt, result)
    }

    pub fn add_stmt_op_bool_to_branch_predicate(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        value: Value,
    ) -> (Statement, LocalBinding) {
        let owner = self.basic_blocks[bb].owner;

        self.validate_value(owner, &value, "value");
        assert_eq!(self.value_ty(&value), TY_BOOL, "value must be a bool");

        let result = self.add_local_binding(owner, TY_PREDICATE);

        let stmt = self
            .statements
            .insert(OpBoolToBranchPredicate { value, result }.into());

        self.basic_blocks[bb].add_statement(position, stmt);

        (stmt, result)
    }

    pub fn add_stmt_op_convert_to_u32(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        value: Value,
    ) -> (Statement, LocalBinding) {
        let owner = self.basic_blocks[bb].owner;

        self.validate_value(owner, &value, "value");

        if !self.value_ty(&value).is_scalar() {
            panic!(
                "expected value to be a `u32`, `i32`, `f32`, or `bool`; found `{}`",
                self.value_ty(&value).to_string(self.ty())
            );
        }

        let result = self.add_local_binding(owner, TY_U32);

        let stmt = self
            .statements
            .insert(OpConvertToU32 { value, result }.into());

        self.basic_blocks[bb].add_statement(position, stmt);

        (stmt, result)
    }

    pub fn add_stmt_op_convert_to_i32(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        value: Value,
    ) -> (Statement, LocalBinding) {
        let owner = self.basic_blocks[bb].owner;

        self.validate_value(owner, &value, "value");

        if !self.value_ty(&value).is_scalar() {
            panic!(
                "expected value to be a `u32`, `i32`, `f32`, or `bool`; found `{}`",
                self.value_ty(&value).to_string(self.ty())
            );
        }

        let result = self.add_local_binding(owner, TY_I32);

        let stmt = self
            .statements
            .insert(OpConvertToI32 { value, result }.into());

        self.basic_blocks[bb].add_statement(position, stmt);

        (stmt, result)
    }

    pub fn add_stmt_op_convert_to_f32(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        value: Value,
    ) -> (Statement, LocalBinding) {
        let owner = self.basic_blocks[bb].owner;

        self.validate_value(owner, &value, "value");

        if !self.value_ty(&value).is_numeric_scalar() {
            panic!(
                "expected value to be a `u32`, `i32`, or `f32`; found `{}`",
                self.value_ty(&value).to_string(self.ty())
            );
        }

        let result = self.add_local_binding(owner, TY_F32);

        let stmt = self
            .statements
            .insert(OpConvertToF32 { value, result }.into());

        self.basic_blocks[bb].add_statement(position, stmt);

        (stmt, result)
    }

    pub fn add_stmt_op_convert_to_bool(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        value: Value,
    ) -> (Statement, LocalBinding) {
        let owner = self.basic_blocks[bb].owner;

        self.validate_value(owner, &value, "value");

        if !self.value_ty(&value).is_scalar() {
            panic!(
                "expected value to be a `u32`, `i32`, `f32`, or `bool`; found `{}`",
                self.value_ty(&value).to_string(self.ty())
            );
        }

        let result = self.add_local_binding(owner, TY_BOOL);

        let stmt = self
            .statements
            .insert(OpConvertToBool { value, result }.into());

        self.basic_blocks[bb].add_statement(position, stmt);

        (stmt, result)
    }

    fn add_local_binding(&mut self, function: Function, ty: Type) -> LocalBinding {
        self.local_bindings.insert(LocalBindingData {
            owner: function,
            ty,
        })
    }

    fn validate_value(&self, owner: Function, value: &Value, desc: &str) {
        if let Value::Local(value) = value {
            assert_eq!(
                self.local_bindings[*value].owner, owner,
                "{} must belong to the basic block's function",
                desc
            );
        }
    }

    fn function_mut(&mut self, function: Function) -> &mut FunctionBody {
        self.function_body_map
            .get_mut(&function)
            .expect("function not registered")
    }

    fn project_ty(&self, (i, value): (usize, &Value), base_ty: Type) -> Type {
        let ty = self.value_ty(value);

        assert_eq!(ty, TY_U32, "index `{}` must be a `u32`", i);

        match &*self.ty.kind(base_ty) {
            TypeKind::Struct(s) => {
                let Value::InlineConst(InlineConst::U32(index)) = *value else {
                    panic!(
                        "index `{}` tried to index into a struct type with a non-constant index",
                        i
                    );
                };

                s.fields[index as usize].ty
            }
            TypeKind::Vector(v) => v.scalar.ty(),
            TypeKind::Matrix(m) => m.column_ty(),
            TypeKind::Array { element_ty, .. } | TypeKind::Slice { element_ty, .. } => *element_ty,
            _ => panic!(
                "index `{}` tried to index a non-aggregate type (`{}`)",
                i,
                base_ty.to_string(self.ty())
            ),
        }
    }
}

impl Index<Function> for Cfg {
    type Output = FunctionBody;

    fn index(&self, index: Function) -> &Self::Output {
        self.function_body_map
            .get(&index)
            .expect("function not registered with CFG")
    }
}

impl Index<LocalBinding> for Cfg {
    type Output = LocalBindingData;

    fn index(&self, index: LocalBinding) -> &Self::Output {
        &self.local_bindings[index]
    }
}

impl Index<BasicBlock> for Cfg {
    type Output = BasicBlockData;

    fn index(&self, index: BasicBlock) -> &Self::Output {
        &self.basic_blocks[index]
    }
}

impl Index<Statement> for Cfg {
    type Output = StatementData;

    fn index(&self, index: Statement) -> &Self::Output {
        &self.statements[index]
    }
}
