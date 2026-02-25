use std::fmt;
use std::ops::Index;

use indexmap::IndexSet;
use ordered_float::OrderedFloat;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use slotmap::SlotMap;
use smallvec::{SmallVec, smallvec};
use thin_vec::{ThinVec, thin_vec};

use crate::intrinsic::Intrinsic;
use crate::ty::{TY_BOOL, TY_F32, TY_I32, TY_U32, Type, TypeKind, TypeRegistry};
use crate::{
    BinaryOperator, Constant, Function, Module, StorageBinding, UnaryOperator, UniformBinding,
    WorkgroupBinding, intrinsic,
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

    pub fn expect_u32(&self) -> u32 {
        if let InlineConst::U32(v) = self {
            *v
        } else {
            panic!("expected a `u32` constant")
        }
    }

    pub fn expect_i32(&self) -> i32 {
        if let InlineConst::I32(v) = self {
            *v
        } else {
            panic!("expected an `i32` constant")
        }
    }

    pub fn expect_f32(&self) -> f32 {
        if let InlineConst::F32(v) = self {
            v.0
        } else {
            panic!("expected an `f32` constant")
        }
    }

    pub fn expect_bool(&self) -> bool {
        if let InlineConst::Bool(v) = self {
            *v
        } else {
            panic!("expected a `bool` constant")
        }
    }

    pub fn expect_ptr(&self) -> ConstPtr {
        if let InlineConst::Ptr(ptr) = self {
            *ptr
        } else {
            panic!("expected a pointer constant")
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
pub struct IntrinsicOp<T> {
    intrinsic: T,
    arguments: SmallVec<[Value; 2]>,
    result: Option<LocalBinding>,
}

impl<T> IntrinsicOp<T> {
    /// The intrinsic operation being invoked.
    pub fn intrinsic(&self) -> &T {
        &self.intrinsic
    }

    /// A list of argument values for invoking the intrinsic operation.
    pub fn arguments(&self) -> &[Value] {
        &self.arguments
    }

    /// A local-binding representing the result of invoking intrinsic operation if the intrinsic
    /// operation has a result, `None` otherwise.
    pub fn maybe_result(&self) -> Option<LocalBinding> {
        self.result
    }
}

macro_rules! gen_intrinsic_arg {
    ($i:literal, $arg:ident) => {
        pub fn $arg(&self) -> Value {
            self.arguments[$i]
        }
    };
}

macro_rules! gen_intrinsic_result {
    () => {
        pub fn result(&self) -> LocalBinding {
            self.maybe_result().unwrap()
        }
    };
}

pub type OpAlloca = IntrinsicOp<intrinsic::OpAlloca>;

impl OpAlloca {
    pub fn ty(&self) -> Type {
        self.intrinsic.ty
    }

    gen_intrinsic_result!();
}

pub type OpLoad = IntrinsicOp<intrinsic::OpLoad>;

impl OpLoad {
    gen_intrinsic_arg!(0, ptr);
    gen_intrinsic_result!();
}

pub type OpStore = IntrinsicOp<intrinsic::OpStore>;

impl OpStore {
    gen_intrinsic_arg!(0, ptr);
    gen_intrinsic_arg!(1, value);
    gen_intrinsic_result!();
}

pub type OpElementPtr = IntrinsicOp<intrinsic::OpElementPtr>;

impl OpElementPtr {
    gen_intrinsic_arg!(0, ptr);
    gen_intrinsic_arg!(1, element_index);
    gen_intrinsic_result!();
}

pub type OpFieldPtr = IntrinsicOp<intrinsic::OpFieldPtr>;

impl OpFieldPtr {
    pub fn field_index(&self) -> u32 {
        self.intrinsic.field_index
    }

    gen_intrinsic_arg!(0, ptr);
    gen_intrinsic_result!();
}

pub type OpExtractElement = IntrinsicOp<intrinsic::OpExtractElement>;

impl OpExtractElement {
    gen_intrinsic_arg!(0, value);
    gen_intrinsic_arg!(1, element_index);
    gen_intrinsic_result!();
}

pub type OpExtractField = IntrinsicOp<intrinsic::OpExtractField>;

impl OpExtractField {
    pub fn field_index(&self) -> u32 {
        self.intrinsic.field_index
    }

    gen_intrinsic_arg!(0, value);
    gen_intrinsic_result!();
}

pub type OpVariantPtr = IntrinsicOp<intrinsic::OpVariantPtr>;

impl OpVariantPtr {
    pub fn variant_index(&self) -> u32 {
        self.intrinsic.variant_index
    }

    gen_intrinsic_arg!(0, ptr);
    gen_intrinsic_result!();
}

pub type OpGetDiscriminant = IntrinsicOp<intrinsic::OpGetDiscriminant>;

impl OpGetDiscriminant {
    gen_intrinsic_arg!(0, ptr);
    gen_intrinsic_result!();
}

pub type OpSetDiscriminant = IntrinsicOp<intrinsic::OpSetDiscriminant>;

impl OpSetDiscriminant {
    pub fn variant_index(&self) -> u32 {
        self.intrinsic.variant_index
    }

    gen_intrinsic_arg!(0, ptr);
    gen_intrinsic_result!();
}

pub type OpOffsetSlice = IntrinsicOp<intrinsic::OpOffsetSlice>;

impl OpOffsetSlice {
    gen_intrinsic_arg!(0, ptr);
    gen_intrinsic_arg!(1, offset);
    gen_intrinsic_result!();
}

pub type OpArrayLength = IntrinsicOp<intrinsic::OpArrayLength>;

impl OpArrayLength {
    gen_intrinsic_arg!(0, ptr);
    gen_intrinsic_result!();
}

pub type OpUnary = IntrinsicOp<intrinsic::OpUnary>;

impl OpUnary {
    pub fn operator(&self) -> UnaryOperator {
        self.intrinsic.operator
    }

    gen_intrinsic_arg!(0, value);
    gen_intrinsic_result!();
}

pub type OpBinary = IntrinsicOp<intrinsic::OpBinary>;

impl OpBinary {
    pub fn operator(&self) -> BinaryOperator {
        self.intrinsic.operator
    }

    gen_intrinsic_arg!(0, lhs);
    gen_intrinsic_arg!(1, rhs);
    gen_intrinsic_result!();
}

pub type OpBoolToBranchSelector = IntrinsicOp<intrinsic::OpBoolToBranchSelector>;

impl OpBoolToBranchSelector {
    gen_intrinsic_arg!(0, value);
    gen_intrinsic_result!();
}

pub type OpCaseToBranchSelector = IntrinsicOp<intrinsic::OpCaseToBranchSelector>;

impl OpCaseToBranchSelector {
    pub fn cases(&self) -> &[u32] {
        &self.intrinsic.cases
    }

    gen_intrinsic_arg!(0, value);
    gen_intrinsic_result!();
}

pub type OpConvertToBool = IntrinsicOp<intrinsic::OpConvertToBool>;

impl OpConvertToBool {
    gen_intrinsic_arg!(0, value);
    gen_intrinsic_result!();
}

pub type OpConvertToF32 = IntrinsicOp<intrinsic::OpConvertToF32>;

impl OpConvertToF32 {
    gen_intrinsic_arg!(0, value);
    gen_intrinsic_result!();
}

pub type OpConvertToI32 = IntrinsicOp<intrinsic::OpConvertToI32>;

impl OpConvertToI32 {
    gen_intrinsic_arg!(0, value);
    gen_intrinsic_result!();
}

pub type OpConvertToU32 = IntrinsicOp<intrinsic::OpConvertToU32>;

impl OpConvertToU32 {
    gen_intrinsic_arg!(0, value);
    gen_intrinsic_result!();
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

    pub fn maybe_result(&self) -> Option<LocalBinding> {
        self.result
    }
}

macro_rules! gen_statement_data {
    ($($op:ident $is:ident $expect:ident $label:literal,)*) => {
        #[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
        pub enum StatementData {
            $($op($op),)*
        }

        $(impl From<$op> for StatementData {
            fn from(op: $op) -> Self {
                StatementData::$op(op)
            }
        })*

        impl StatementData {
            $(
                pub fn $is(&self) -> bool {
                    matches!(self, StatementData::$op(_))
                }

                pub fn $expect(&self) -> &$op {
                    if let StatementData::$op(op) = self {
                        op
                    } else {
                        panic!("expected statement to be a {} statement", $label)
                    }
                }
            )*
        }
    };
}

gen_statement_data! {
    Bind is_bind expect_bind "bind",
    Uninitialized is_uninitialized expect_uninitialized "uninitialized",
    Assign is_assign expect_assign "assign",
    OpAlloca is_op_alloca expect_op_alloca "alloca",
    OpLoad is_op_load expect_op_load "load",
    OpStore is_op_store expect_op_store "store",
    OpExtractElement is_op_extract_element expect_op_extract_element "extract-element",
    OpExtractField is_op_extract_field expect_op_extract_field "extract-field",
    OpElementPtr is_op_element_ptr expect_op_element_ptr "element-ptr",
    OpFieldPtr is_op_field_ptr expect_op_field_ptr "field-ptr",
    OpVariantPtr is_op_variant_ptr expect_op_variant_ptr "variant-ptr",
    OpGetDiscriminant is_op_get_discriminant expect_op_get_discriminant "get-discriminant",
    OpSetDiscriminant is_op_set_discriminant expect_op_set_discriminant "set-discriminant",
    OpOffsetSlice is_op_offset_slice_ptr expect_op_offset_slice_ptr "offset-slice-ptr",
    OpUnary is_op_unary expect_op_unary "unary",
    OpBinary is_op_binary expect_op_binary "binary",
    OpCall is_op_call expect_op_call "call",
    OpCaseToBranchSelector is_op_case_to_branch_selector expect_op_case_to_branch_selector "case-to-branch-selector",
    OpBoolToBranchSelector is_op_bool_to_branch_selector expect_op_bool_to_branch_selector "bool-to-branch-selector",
    OpConvertToU32 is_op_convert_to_u32 expect_op_convert_to_u32 "convert-to-u32",
    OpConvertToI32 is_op_convert_to_i32 expect_op_convert_to_i32 "convert-to-i32",
    OpConvertToF32 is_op_convert_to_f32 expect_op_convert_to_f32 "convert-to-f32",
    OpConvertToBool is_op_convert_to_bool expect_op_convert_to_bool "convert-to-bool",
    OpArrayLength is_op_array_length expect_op_array_length "array-length",
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

    /// Adds a statement to a basic-block that binds the given `value` as a local value with the
    /// given `dest_ty`.
    ///
    /// Very similar to [add_stmt_bind], but the local value may have a different type than the
    /// value being bound. The `dest_ty` must be compatible with the `value`'s ty as per
    /// [TyRegistry::is_compatible]. The statement that is added is a [Bind] statement.
    pub fn add_stmt_cast(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        value: Value,
        dest_ty: Type,
    ) -> (Statement, LocalBinding) {
        let owner = self.basic_blocks[bb].owner;

        self.validate_value(owner, &value, "value");

        let ty = self.value_ty(&value);

        assert!(self.ty().is_compatible(ty, dest_ty));

        let local_binding = self.add_local_binding(owner, dest_ty);

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

    pub fn add_stmt_intrinsic_op<T>(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        op: T,
        arguments: impl IntoIterator<Item = Value>,
    ) -> (Statement, Option<LocalBinding>)
    where
        T: Intrinsic,
        StatementData: From<IntrinsicOp<T>>,
    {
        let arguments: SmallVec<[Value; 2]> = arguments.into_iter().collect();
        let owner = self.basic_blocks[bb].owner;

        for (i, arg) in arguments.iter().enumerate() {
            if let Value::Local(local) = *arg {
                assert_eq!(
                    self.local_bindings[local].owner, owner,
                    "argument {i} must belong to the basic-block's owner function"
                )
            }
        }

        let arg_types = arguments.iter().map(|v| self.value_ty(v));
        let ret_ty = op.process_args(self.ty(), arg_types).unwrap();
        let result = ret_ty.map(|ty| self.add_local_binding(owner, ty));

        let intrinsic_op = IntrinsicOp {
            intrinsic: op,
            arguments,
            result,
        };

        let stmt = self.statements.insert(StatementData::from(intrinsic_op));

        self.basic_blocks[bb].add_statement(position, stmt);

        (stmt, result)
    }

    fn add_stmt_intrinsic_op_internal<T>(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        op: T,
        arguments: impl IntoIterator<Item = (Value, &'static str)>,
    ) -> (Statement, Option<LocalBinding>)
    where
        T: Intrinsic,
        StatementData: From<IntrinsicOp<T>>,
    {
        let owner = self.basic_blocks[bb].owner;
        let mut args = SmallVec::new();

        for (value, label) in arguments.into_iter() {
            if let Value::Local(local) = value {
                assert_eq!(
                    self.local_bindings[local].owner, owner,
                    "`{label}` argument must belong to the basic-block's owner function"
                )
            }

            args.push(value);
        }

        let arg_types = args.iter().map(|v| self.value_ty(v));
        let ret_ty = op.process_args(self.ty(), arg_types).unwrap();
        let result = ret_ty.map(|ty| self.add_local_binding(owner, ty));

        let intrinsic_op = IntrinsicOp {
            intrinsic: op,
            arguments: args,
            result,
        };

        let stmt = self.statements.insert(StatementData::from(intrinsic_op));

        self.basic_blocks[bb].add_statement(position, stmt);

        (stmt, result)
    }

    pub fn add_stmt_op_alloca(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        ty: Type,
    ) -> (Statement, LocalBinding) {
        let (stmt, result) =
            self.add_stmt_intrinsic_op_internal(bb, position, intrinsic::OpAlloca { ty }, []);

        (stmt, result.unwrap())
    }

    pub fn add_stmt_op_load(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        ptr: Value,
    ) -> (Statement, LocalBinding) {
        let (stmt, result) =
            self.add_stmt_intrinsic_op_internal(bb, position, intrinsic::OpLoad, [(ptr, "ptr")]);

        (stmt, result.unwrap())
    }

    pub fn add_stmt_op_store(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        ptr: Value,
        value: Value,
    ) -> Statement {
        let (stmt, _) = self.add_stmt_intrinsic_op_internal(
            bb,
            position,
            intrinsic::OpStore,
            [(ptr, "ptr"), (value, "value")],
        );

        stmt
    }

    pub fn add_stmt_op_extract_field(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        value: Value,
        field_index: u32,
    ) -> (Statement, LocalBinding) {
        let (stmt, result) = self.add_stmt_intrinsic_op_internal(
            bb,
            position,
            intrinsic::OpExtractField { field_index },
            [(value, "value")],
        );

        (stmt, result.unwrap())
    }

    pub fn add_stmt_op_extract_element(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        value: Value,
        element_index: Value,
    ) -> (Statement, LocalBinding) {
        let (stmt, result) = self.add_stmt_intrinsic_op_internal(
            bb,
            position,
            intrinsic::OpExtractElement,
            [(value, "value"), (element_index, "element_index")],
        );

        (stmt, result.unwrap())
    }

    pub fn add_stmt_op_field_ptr(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        ptr: Value,
        field_index: u32,
    ) -> (Statement, LocalBinding) {
        let (stmt, result) = self.add_stmt_intrinsic_op_internal(
            bb,
            position,
            intrinsic::OpFieldPtr { field_index },
            [(ptr, "ptr")],
        );

        (stmt, result.unwrap())
    }

    pub fn add_stmt_op_element_ptr(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        ptr: Value,
        element_index: Value,
    ) -> (Statement, LocalBinding) {
        let (stmt, result) = self.add_stmt_intrinsic_op_internal(
            bb,
            position,
            intrinsic::OpElementPtr,
            [(ptr, "ptr"), (element_index, "element_index")],
        );

        (stmt, result.unwrap())
    }

    pub fn add_stmt_op_variant_ptr(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        ptr: Value,
        variant_index: u32,
    ) -> (Statement, LocalBinding) {
        let (stmt, result) = self.add_stmt_intrinsic_op_internal(
            bb,
            position,
            intrinsic::OpVariantPtr { variant_index },
            [(ptr, "ptr")],
        );

        (stmt, result.unwrap())
    }

    pub fn add_stmt_op_get_discriminant(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        ptr: Value,
    ) -> (Statement, LocalBinding) {
        let (stmt, result) = self.add_stmt_intrinsic_op_internal(
            bb,
            position,
            intrinsic::OpGetDiscriminant,
            [(ptr, "ptr")],
        );

        (stmt, result.unwrap())
    }

    pub fn add_stmt_op_set_discriminant(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        ptr: Value,
        variant_index: u32,
    ) -> Statement {
        let (stmt, _) = self.add_stmt_intrinsic_op_internal(
            bb,
            position,
            intrinsic::OpSetDiscriminant { variant_index },
            [(ptr, "ptr")],
        );

        stmt
    }

    pub fn add_stmt_op_offset_slice(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        ptr: Value,
        offset: Value,
    ) -> (Statement, LocalBinding) {
        let (stmt, result) = self.add_stmt_intrinsic_op_internal(
            bb,
            position,
            intrinsic::OpOffsetSlice,
            [(ptr, "ptr"), (offset, "offset")],
        );

        (stmt, result.unwrap())
    }

    pub fn add_stmt_op_unary(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        operator: UnaryOperator,
        operand: Value,
    ) -> (Statement, LocalBinding) {
        let (stmt, result) = self.add_stmt_intrinsic_op_internal(
            bb,
            position,
            intrinsic::OpUnary { operator },
            [(operand, "operand")],
        );

        (stmt, result.unwrap())
    }

    pub fn add_stmt_op_binary(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        operator: BinaryOperator,
        lhs: Value,
        rhs: Value,
    ) -> (Statement, LocalBinding) {
        let (stmt, result) = self.add_stmt_intrinsic_op_internal(
            bb,
            position,
            intrinsic::OpBinary { operator },
            [(lhs, "lhs"), (rhs, "rhs")],
        );

        (stmt, result.unwrap())
    }

    pub fn add_stmt_op_case_to_branch_selector(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        case: Value,
        cases: impl IntoIterator<Item = u32>,
    ) -> (Statement, LocalBinding) {
        let (stmt, result) = self.add_stmt_intrinsic_op_internal(
            bb,
            position,
            intrinsic::OpCaseToBranchSelector {
                cases: cases.into_iter().collect(),
            },
            [(case, "case")],
        );

        (stmt, result.unwrap())
    }

    pub fn add_stmt_op_bool_to_branch_selector(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        value: Value,
    ) -> (Statement, LocalBinding) {
        let (stmt, result) = self.add_stmt_intrinsic_op_internal(
            bb,
            position,
            intrinsic::OpBoolToBranchSelector,
            [(value, "value")],
        );

        (stmt, result.unwrap())
    }

    pub fn add_stmt_op_convert_to_u32(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        value: Value,
    ) -> (Statement, LocalBinding) {
        let (stmt, result) = self.add_stmt_intrinsic_op_internal(
            bb,
            position,
            intrinsic::OpConvertToU32,
            [(value, "value")],
        );

        (stmt, result.unwrap())
    }

    pub fn add_stmt_op_convert_to_i32(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        value: Value,
    ) -> (Statement, LocalBinding) {
        let (stmt, result) = self.add_stmt_intrinsic_op_internal(
            bb,
            position,
            intrinsic::OpConvertToI32,
            [(value, "value")],
        );

        (stmt, result.unwrap())
    }

    pub fn add_stmt_op_convert_to_f32(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        value: Value,
    ) -> (Statement, LocalBinding) {
        let (stmt, result) = self.add_stmt_intrinsic_op_internal(
            bb,
            position,
            intrinsic::OpConvertToF32,
            [(value, "value")],
        );

        (stmt, result.unwrap())
    }

    pub fn add_stmt_op_convert_to_bool(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        value: Value,
    ) -> (Statement, LocalBinding) {
        let (stmt, result) = self.add_stmt_intrinsic_op_internal(
            bb,
            position,
            intrinsic::OpConvertToBool,
            [(value, "value")],
        );

        (stmt, result.unwrap())
    }

    pub fn add_stmt_op_array_length(
        &mut self,
        bb: BasicBlock,
        position: BlockPosition,
        ptr: Value,
    ) -> (Statement, LocalBinding) {
        let (stmt, result) = self.add_stmt_intrinsic_op_internal(
            bb,
            position,
            intrinsic::OpArrayLength,
            [(ptr, "ptr")],
        );

        (stmt, result.unwrap())
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
