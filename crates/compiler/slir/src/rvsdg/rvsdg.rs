use std::ops::Index;
use std::slice;

use indexmap::IndexSet;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use slotmap::SlotMap;
use thiserror::Error;

use crate::builtin_function::BuiltinFunction;
use crate::ty::{
    TY_BOOL, TY_F32, TY_I32, TY_PREDICATE, TY_PTR_U32, TY_U32, Type, TypeKind, TypeRegistry,
};
use crate::util::thin_set::ThinSet;
use crate::{
    BinaryOperator, Constant, Function, Module, StorageBinding, UnaryOperator, UniformBinding,
    WorkgroupBinding, thin_set, ty,
};

pub trait Connectivity {
    fn value_inputs(&self) -> &[ValueInput];

    fn value_inputs_mut(&mut self) -> &mut [ValueInput];

    fn value_outputs(&self) -> &[ValueOutput];

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput];

    fn state(&self) -> Option<&State>;

    fn state_mut(&mut self) -> Option<&mut State>;
}

slotmap::new_key_type! {
    pub struct Node;
    pub struct Region;
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct FunctionNode {
    dependencies: Vec<ValueInput>,
    output: ValueOutput,
    region: Region,
}

impl FunctionNode {
    pub fn body_region(&self) -> Region {
        self.region
    }

    pub fn dependencies(&self) -> &[ValueInput] {
        &self.dependencies
    }

    pub fn output(&self) -> &ValueOutput {
        &self.output
    }
}

impl Connectivity for FunctionNode {
    fn value_inputs(&self) -> &[ValueInput] {
        &self.dependencies
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        &mut self.dependencies
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        slice::from_ref(&self.output)
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        slice::from_mut(&mut self.output)
    }

    fn state(&self) -> Option<&State> {
        None
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        None
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct ValueInput {
    pub ty: Type,
    pub origin: ValueOrigin,
}

impl ValueInput {
    pub fn placeholder(ty: Type) -> Self {
        ValueInput {
            ty,
            origin: ValueOrigin::placeholder(),
        }
    }

    pub fn argument(ty: Type, arg: u32) -> Self {
        ValueInput {
            ty,
            origin: ValueOrigin::Argument(arg),
        }
    }

    pub fn output(ty: Type, node: Node, output: u32) -> Self {
        ValueInput {
            ty,
            origin: ValueOrigin::Output {
                producer: node,
                output,
            },
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub enum ValueOrigin {
    Argument(u32),
    Output { producer: Node, output: u32 },
}

impl ValueOrigin {
    pub fn placeholder() -> Self {
        ValueOrigin::Argument(u32::MAX)
    }

    pub fn is_placeholder(&self) -> bool {
        self == &ValueOrigin::Argument(u32::MAX)
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct ValueOutput {
    pub ty: Type,
    pub users: ThinSet<ValueUser>,
}

impl ValueOutput {
    pub fn new(ty: Type) -> Self {
        ValueOutput {
            ty,
            users: ThinSet::new(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub enum ValueUser {
    Result(u32),
    Input { consumer: Node, input: u32 },
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub enum StateOrigin {
    Argument,
    Node(Node),
}

impl StateOrigin {
    pub fn as_node(&self) -> Option<Node> {
        if let StateOrigin::Node(node) = self {
            Some(*node)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub enum StateUser {
    Result,
    Node(Node),
}

impl StateUser {
    pub fn as_node(&self) -> Option<Node> {
        if let StateUser::Node(node) = self {
            Some(*node)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct State {
    pub origin: StateOrigin,
    pub user: StateUser,
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct RegionData {
    owner: Option<Node>,
    nodes: IndexSet<Node>,
    value_arguments: Vec<ValueOutput>,
    value_results: Vec<ValueInput>,
    state_argument: StateUser,
    state_result: StateOrigin,
}

impl RegionData {
    pub fn owner(&self) -> Node {
        self.owner.expect("region not correctly initialized")
    }

    pub fn nodes(&self) -> &IndexSet<Node> {
        &self.nodes
    }

    pub fn value_arguments(&self) -> &[ValueOutput] {
        &self.value_arguments
    }

    pub fn value_results(&self) -> &[ValueInput] {
        &self.value_results
    }

    pub fn state_argument(&self) -> &StateUser {
        &self.state_argument
    }

    pub fn state_result(&self) -> &StateOrigin {
        &self.state_result
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct NodeData {
    kind: NodeKind,
    region: Option<Region>,
}

impl NodeData {
    pub fn kind(&self) -> &NodeKind {
        &self.kind
    }

    pub fn region(&self) -> Region {
        self.region
            .expect("should have a region after initialization")
    }

    pub fn value_input_for_origin(&self, origin: ValueOrigin) -> Option<u32> {
        for (i, input) in self.value_inputs().iter().enumerate() {
            if input.origin == origin {
                return Some(i as u32);
            }
        }

        None
    }

    pub fn is_function(&self) -> bool {
        matches!(self.kind, NodeKind::Function(_))
    }

    pub fn expect_function(&self) -> &FunctionNode {
        if let NodeKind::Function(n) = &self.kind {
            n
        } else {
            panic!("expected node to be a function node")
        }
    }

    fn expect_function_mut(&mut self) -> &mut FunctionNode {
        if let NodeKind::Function(n) = &mut self.kind {
            n
        } else {
            panic!("expected node to be a function node")
        }
    }

    pub fn is_uniform_binding(&self) -> bool {
        matches!(self.kind, NodeKind::UniformBinding(_))
    }

    pub fn expect_uniform_binding(&self) -> &UniformBindingNode {
        if let NodeKind::UniformBinding(n) = &self.kind {
            n
        } else {
            panic!("expected node to be a uniform binding node")
        }
    }

    pub fn is_storage_binding(&self) -> bool {
        matches!(self.kind, NodeKind::StorageBinding(_))
    }

    pub fn expect_storage_binding(&self) -> &StorageBindingNode {
        if let NodeKind::StorageBinding(n) = &self.kind {
            n
        } else {
            panic!("expected node to be a storage binding node")
        }
    }

    pub fn is_workgroup_binding(&self) -> bool {
        matches!(self.kind, NodeKind::WorkgroupBinding(_))
    }

    pub fn expect_workgroup_binding(&self) -> &WorkgroupBindingNode {
        if let NodeKind::WorkgroupBinding(n) = &self.kind {
            n
        } else {
            panic!("expected node to be a workgroup binding node")
        }
    }

    pub fn is_simple(&self) -> bool {
        matches!(self.kind, NodeKind::Simple(_))
    }

    pub fn expect_simple(&self) -> &SimpleNode {
        if let NodeKind::Simple(n) = &self.kind {
            n
        } else {
            panic!("expected node to be a simple node")
        }
    }

    pub fn is_switch(&self) -> bool {
        matches!(self.kind, NodeKind::Switch(_))
    }

    pub fn expect_switch(&self) -> &SwitchNode {
        if let NodeKind::Switch(n) = &self.kind {
            n
        } else {
            panic!("expected node to be a switch node")
        }
    }

    fn expect_switch_mut(&mut self) -> &mut SwitchNode {
        if let NodeKind::Switch(n) = &mut self.kind {
            n
        } else {
            panic!("expected node to be a switch node")
        }
    }

    pub fn is_loop(&self) -> bool {
        matches!(self.kind, NodeKind::Loop(_))
    }

    pub fn expect_loop(&self) -> &LoopNode {
        if let NodeKind::Loop(n) = &self.kind {
            n
        } else {
            panic!("expected node to be a loop node")
        }
    }

    fn expect_loop_mut(&mut self) -> &mut LoopNode {
        if let NodeKind::Loop(n) = &mut self.kind {
            n
        } else {
            panic!("expected node to be a loop node")
        }
    }

    pub fn is_const_u32(&self) -> bool {
        matches!(self.kind, NodeKind::Simple(SimpleNode::ConstU32(_)))
    }

    pub fn expect_const_u32(&self) -> &ConstU32 {
        if let NodeKind::Simple(SimpleNode::ConstU32(op)) = &self.kind {
            op
        } else {
            panic!("expected node to be a `u32` constant")
        }
    }

    pub fn is_const_i32(&self) -> bool {
        matches!(self.kind, NodeKind::Simple(SimpleNode::ConstI32(_)))
    }

    pub fn expect_const_i32(&self) -> &ConstI32 {
        if let NodeKind::Simple(SimpleNode::ConstI32(op)) = &self.kind {
            op
        } else {
            panic!("expected node to be an `i32` constant")
        }
    }

    pub fn is_const_f32(&self) -> bool {
        matches!(self.kind, NodeKind::Simple(SimpleNode::ConstF32(_)))
    }

    pub fn expect_const_f32(&self) -> &ConstF32 {
        if let NodeKind::Simple(SimpleNode::ConstF32(op)) = &self.kind {
            op
        } else {
            panic!("expected node to be an `f32` constant")
        }
    }

    pub fn is_const_bool(&self) -> bool {
        matches!(self.kind, NodeKind::Simple(SimpleNode::ConstBool(_)))
    }

    pub fn expect_const_bool(&self) -> &ConstBool {
        if let NodeKind::Simple(SimpleNode::ConstBool(op)) = &self.kind {
            op
        } else {
            panic!("expected node to be a `bool` constant")
        }
    }

    pub fn is_const_predicate(&self) -> bool {
        matches!(self.kind, NodeKind::Simple(SimpleNode::ConstPredicate(_)))
    }

    pub fn expect_const_predicate(&self) -> &ConstPredicate {
        if let NodeKind::Simple(SimpleNode::ConstPredicate(op)) = &self.kind {
            op
        } else {
            panic!("expected node to be a `predicate` constant")
        }
    }

    pub fn is_const_ptr(&self) -> bool {
        matches!(self.kind, NodeKind::Simple(SimpleNode::ConstPtr(_)))
    }

    pub fn expect_const_ptr(&self) -> &ConstPtr {
        if let NodeKind::Simple(SimpleNode::ConstPtr(op)) = &self.kind {
            op
        } else {
            panic!("expected node to be a pointer constant")
        }
    }

    pub fn is_const_fallback(&self) -> bool {
        matches!(self.kind, NodeKind::Simple(SimpleNode::ConstFallback(_)))
    }

    pub fn expect_const_fallback(&self) -> &ConstFallback {
        if let NodeKind::Simple(SimpleNode::ConstFallback(op)) = &self.kind {
            op
        } else {
            panic!("expected node to be a fallback value constant")
        }
    }

    pub fn is_op_alloca(&self) -> bool {
        matches!(self.kind, NodeKind::Simple(SimpleNode::OpAlloca(_)))
    }

    pub fn expect_op_alloca(&self) -> &OpAlloca {
        if let NodeKind::Simple(SimpleNode::OpAlloca(op)) = &self.kind {
            op
        } else {
            panic!("expected node to be an `alloca` operation")
        }
    }

    pub fn is_op_load(&self) -> bool {
        matches!(self.kind, NodeKind::Simple(SimpleNode::OpLoad(_)))
    }

    pub fn expect_op_load(&self) -> &OpLoad {
        if let NodeKind::Simple(SimpleNode::OpLoad(op)) = &self.kind {
            op
        } else {
            panic!("expected node to be a `load` operation")
        }
    }

    pub fn is_op_store(&self) -> bool {
        matches!(self.kind, NodeKind::Simple(SimpleNode::OpStore(_)))
    }

    pub fn expect_op_store(&self) -> &OpStore {
        if let NodeKind::Simple(SimpleNode::OpStore(op)) = &self.kind {
            op
        } else {
            panic!("expected node to be a `store` operation")
        }
    }

    pub fn is_op_ptr_element_ptr(&self) -> bool {
        matches!(self.kind, NodeKind::Simple(SimpleNode::OpPtrElementPtr(_)))
    }

    pub fn expect_op_ptr_element_ptr(&self) -> &OpPtrElementPtr {
        if let NodeKind::Simple(SimpleNode::OpPtrElementPtr(op)) = &self.kind {
            op
        } else {
            panic!("expected node to be a `pointer-element-pointer` operation")
        }
    }

    pub fn is_op_ptr_discriminant_ptr(&self) -> bool {
        matches!(
            self.kind,
            NodeKind::Simple(SimpleNode::OpPtrDiscriminantPtr(_))
        )
    }

    pub fn expect_op_ptr_discriminant_ptr(&self) -> &OpPtrDiscriminantPtr {
        if let NodeKind::Simple(SimpleNode::OpPtrDiscriminantPtr(op)) = &self.kind {
            op
        } else {
            panic!("expected node to be a `pointer-discriminant-pointer` operation")
        }
    }

    pub fn is_op_ptr_variant_ptr(&self) -> bool {
        matches!(self.kind, NodeKind::Simple(SimpleNode::OpPtrVariantPtr(_)))
    }

    pub fn expect_op_ptr_variant_ptr(&self) -> &OpPtrVariantPtr {
        if let NodeKind::Simple(SimpleNode::OpPtrVariantPtr(op)) = &self.kind {
            op
        } else {
            panic!("expected node to be a `pointer-variant-pointer` operation")
        }
    }

    pub fn is_op_extract_element(&self) -> bool {
        matches!(self.kind, NodeKind::Simple(SimpleNode::OpExtractElement(_)))
    }

    pub fn expect_op_extract_element(&self) -> &OpExtractElement {
        if let NodeKind::Simple(SimpleNode::OpExtractElement(op)) = &self.kind {
            op
        } else {
            panic!("expected node to be an `extract-element` operation")
        }
    }

    pub fn is_op_get_discriminant(&self) -> bool {
        matches!(
            self.kind,
            NodeKind::Simple(SimpleNode::OpGetDiscriminant(_))
        )
    }

    pub fn expect_op_get_discriminant(&self) -> &OpGetDiscriminant {
        if let NodeKind::Simple(SimpleNode::OpGetDiscriminant(op)) = &self.kind {
            op
        } else {
            panic!("expected node to be a `get-discriminant` operation")
        }
    }

    pub fn is_op_set_discriminant(&self) -> bool {
        matches!(
            self.kind,
            NodeKind::Simple(SimpleNode::OpSetDiscriminant(_))
        )
    }

    pub fn expect_op_set_discriminant(&self) -> &OpSetDiscriminant {
        if let NodeKind::Simple(SimpleNode::OpSetDiscriminant(op)) = &self.kind {
            op
        } else {
            panic!("expected node to be a `set-discriminant` operation")
        }
    }

    pub fn is_op_add_ptr_offset(&self) -> bool {
        matches!(self.kind, NodeKind::Simple(SimpleNode::OpAddPtrOffset(_)))
    }

    pub fn expect_op_add_ptr_offset(&self) -> &OpAddPtrOffset {
        if let NodeKind::Simple(SimpleNode::OpAddPtrOffset(op)) = &self.kind {
            op
        } else {
            panic!("expected node to be a `add-ptr-offset` operation")
        }
    }

    pub fn is_op_get_ptr_offset(&self) -> bool {
        matches!(self.kind, NodeKind::Simple(SimpleNode::OpGetPtrOffset(_)))
    }

    pub fn expect_op_get_ptr_offset(&self) -> &OpGetPtrOffset {
        if let NodeKind::Simple(SimpleNode::OpGetPtrOffset(op)) = &self.kind {
            op
        } else {
            panic!("expected node to be a `get-ptr-offset` operation")
        }
    }

    pub fn is_op_call(&self) -> bool {
        matches!(self.kind, NodeKind::Simple(SimpleNode::OpCall(_)))
    }

    pub fn expect_op_call(&self) -> &OpCall {
        if let NodeKind::Simple(SimpleNode::OpCall(op)) = &self.kind {
            op
        } else {
            panic!("expected node to be an `call` operation")
        }
    }

    pub fn is_op_call_builtin(&self) -> bool {
        matches!(self.kind, NodeKind::Simple(SimpleNode::OpCallBuiltin(_)))
    }

    pub fn expect_op_call_builtin(&self) -> &OpCallBuiltin {
        if let NodeKind::Simple(SimpleNode::OpCallBuiltin(op)) = &self.kind {
            op
        } else {
            panic!("expected node to be an `call-builtin` operation")
        }
    }

    pub fn is_op_unary(&self) -> bool {
        matches!(self.kind, NodeKind::Simple(SimpleNode::OpUnary(_)))
    }

    pub fn expect_op_unary(&self) -> &OpUnary {
        if let NodeKind::Simple(SimpleNode::OpUnary(op)) = &self.kind {
            op
        } else {
            panic!("expected node to be a `unary` operation")
        }
    }

    pub fn is_op_binary(&self) -> bool {
        matches!(self.kind, NodeKind::Simple(SimpleNode::OpBinary(_)))
    }

    pub fn expect_op_binary(&self) -> &OpBinary {
        if let NodeKind::Simple(SimpleNode::OpBinary(op)) = &self.kind {
            op
        } else {
            panic!("expected node to be a `binary` operation")
        }
    }

    pub fn is_op_vector(&self) -> bool {
        matches!(self.kind, NodeKind::Simple(SimpleNode::OpVector(_)))
    }

    pub fn expect_op_vector(&self) -> &OpVector {
        if let NodeKind::Simple(SimpleNode::OpVector(op)) = &self.kind {
            op
        } else {
            panic!("expected node to be a `vector` operation")
        }
    }

    pub fn is_op_matrix(&self) -> bool {
        matches!(self.kind, NodeKind::Simple(SimpleNode::OpMatrix(_)))
    }

    pub fn expect_op_matrix(&self) -> &OpMatrix {
        if let NodeKind::Simple(SimpleNode::OpMatrix(op)) = &self.kind {
            op
        } else {
            panic!("expected node to be a `matrix` operation")
        }
    }

    pub fn is_op_case_to_switch_predicate(&self) -> bool {
        matches!(
            self.kind,
            NodeKind::Simple(SimpleNode::OpCaseToSwitchPredicate(_))
        )
    }

    pub fn expect_op_case_to_switch_predicate(&self) -> &OpCaseToSwitchPredicate {
        if let NodeKind::Simple(SimpleNode::OpCaseToSwitchPredicate(proxy)) = &self.kind {
            proxy
        } else {
            panic!("expected node to be an op-case-to-switch-predicate node")
        }
    }

    fn expect_op_case_to_switch_predicate_mut(&mut self) -> &mut OpCaseToSwitchPredicate {
        if let NodeKind::Simple(SimpleNode::OpCaseToSwitchPredicate(proxy)) = &mut self.kind {
            proxy
        } else {
            panic!("expected node to be an op-case-to-switch-predicate node")
        }
    }

    pub fn is_op_bool_to_switch_predicate(&self) -> bool {
        matches!(
            self.kind,
            NodeKind::Simple(SimpleNode::OpBoolToSwitchPredicate(_))
        )
    }

    pub fn expect_op_bool_to_switch_predicate(&self) -> &OpBoolToSwitchPredicate {
        if let NodeKind::Simple(SimpleNode::OpBoolToSwitchPredicate(proxy)) = &self.kind {
            proxy
        } else {
            panic!("expected node to be an op-bool-to-switch-predicate node")
        }
    }

    pub fn is_op_u32_to_switch_predicate(&self) -> bool {
        matches!(
            self.kind,
            NodeKind::Simple(SimpleNode::OpU32ToSwitchPredicate(_))
        )
    }

    pub fn expect_op_u32_to_switch_predicate(&self) -> &OpU32ToSwitchPredicate {
        if let NodeKind::Simple(SimpleNode::OpU32ToSwitchPredicate(proxy)) = &self.kind {
            proxy
        } else {
            panic!("expected node to be an op-u32-to-switch-predicate node")
        }
    }

    pub fn is_op_switch_predicate_case(&self) -> bool {
        matches!(
            self.kind,
            NodeKind::Simple(SimpleNode::OpSwitchPredicateToCase(_))
        )
    }

    pub fn expect_op_switch_predicate_to_case(&self) -> &OpSwitchPredicateToCase {
        if let NodeKind::Simple(SimpleNode::OpSwitchPredicateToCase(proxy)) = &self.kind {
            proxy
        } else {
            panic!("expected node to be an op-switch-predicate-to-case node")
        }
    }

    pub fn is_op_convert_to_u32(&self) -> bool {
        matches!(self.kind, NodeKind::Simple(SimpleNode::OpConvertToU32(_)))
    }

    pub fn expect_op_convert_to_u32(&self) -> &OpConvertToU32 {
        if let NodeKind::Simple(SimpleNode::OpConvertToU32(op)) = &self.kind {
            op
        } else {
            panic!("expected node to be a `convert-to-u32` operation")
        }
    }

    pub fn is_op_convert_to_i32(&self) -> bool {
        matches!(self.kind, NodeKind::Simple(SimpleNode::OpConvertToI32(_)))
    }

    pub fn expect_op_convert_to_i32(&self) -> &OpConvertToI32 {
        if let NodeKind::Simple(SimpleNode::OpConvertToI32(op)) = &self.kind {
            op
        } else {
            panic!("expected node to be a `convert-to-i32` operation")
        }
    }

    pub fn is_op_convert_to_f32(&self) -> bool {
        matches!(self.kind, NodeKind::Simple(SimpleNode::OpConvertToF32(_)))
    }

    pub fn expect_op_convert_to_f32(&self) -> &OpConvertToF32 {
        if let NodeKind::Simple(SimpleNode::OpConvertToF32(op)) = &self.kind {
            op
        } else {
            panic!("expected node to be a `convert-to-f32` operation")
        }
    }

    pub fn is_op_convert_to_bool(&self) -> bool {
        matches!(self.kind, NodeKind::Simple(SimpleNode::OpConvertToBool(_)))
    }

    pub fn expect_op_convert_to_bool(&self) -> &OpConvertToBool {
        if let NodeKind::Simple(SimpleNode::OpConvertToBool(op)) = &self.kind {
            op
        } else {
            panic!("expected node to be a `convert-to-bool` operation")
        }
    }

    pub fn is_value_proxy(&self) -> bool {
        matches!(self.kind, NodeKind::Simple(SimpleNode::ValueProxy(_)))
    }

    pub fn is_switch_output_replacement_marker(&self) -> bool {
        if let NodeKind::Simple(SimpleNode::ValueProxy(proxy)) = &self.kind {
            proxy.proxy_kind().is_switch_output_replacement_marker()
        } else {
            false
        }
    }

    pub fn expect_value_proxy(&self) -> &ValueProxy {
        if let NodeKind::Simple(SimpleNode::ValueProxy(proxy)) = &self.kind {
            proxy
        } else {
            panic!("expected node to be a value-proxy node")
        }
    }

    pub fn is_reaggregation(&self) -> bool {
        matches!(self.kind, NodeKind::Simple(SimpleNode::Reaggregation(_)))
    }

    pub fn expect_reaggregation(&self) -> &Reaggregation {
        if let NodeKind::Simple(SimpleNode::Reaggregation(node)) = &self.kind {
            node
        } else {
            panic!("expected node to be a reaggregation node")
        }
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub enum NodeKind {
    Switch(SwitchNode),
    Loop(LoopNode),
    Simple(SimpleNode),
    UniformBinding(UniformBindingNode),
    StorageBinding(StorageBindingNode),
    WorkgroupBinding(WorkgroupBindingNode),
    Constant(ConstantNode),
    Function(FunctionNode),
}

impl NodeKind {
    fn expect_switch_mut(&mut self) -> &mut SwitchNode {
        if let NodeKind::Switch(n) = self {
            n
        } else {
            panic!("expected node to be switch node")
        }
    }
}

impl Connectivity for NodeData {
    fn value_inputs(&self) -> &[ValueInput] {
        match &self.kind {
            NodeKind::Switch(n) => n.value_inputs(),
            NodeKind::Loop(n) => n.value_inputs(),
            NodeKind::Simple(n) => n.value_inputs(),
            NodeKind::UniformBinding(n) => n.value_inputs(),
            NodeKind::StorageBinding(n) => n.value_inputs(),
            NodeKind::WorkgroupBinding(n) => n.value_inputs(),
            NodeKind::Constant(n) => n.value_inputs(),
            NodeKind::Function(n) => n.value_inputs(),
        }
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        match &mut self.kind {
            NodeKind::Switch(n) => n.value_inputs_mut(),
            NodeKind::Loop(n) => n.value_inputs_mut(),
            NodeKind::Simple(n) => n.value_inputs_mut(),
            NodeKind::UniformBinding(n) => n.value_inputs_mut(),
            NodeKind::StorageBinding(n) => n.value_inputs_mut(),
            NodeKind::WorkgroupBinding(n) => n.value_inputs_mut(),
            NodeKind::Constant(n) => n.value_inputs_mut(),
            NodeKind::Function(n) => n.value_inputs_mut(),
        }
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        match &self.kind {
            NodeKind::Switch(n) => n.value_outputs(),
            NodeKind::Loop(n) => n.value_outputs(),
            NodeKind::Simple(n) => n.value_outputs(),
            NodeKind::UniformBinding(n) => n.value_outputs(),
            NodeKind::StorageBinding(n) => n.value_outputs(),
            NodeKind::WorkgroupBinding(n) => n.value_outputs(),
            NodeKind::Constant(n) => n.value_outputs(),
            NodeKind::Function(n) => n.value_outputs(),
        }
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        match &mut self.kind {
            NodeKind::Switch(n) => n.value_outputs_mut(),
            NodeKind::Loop(n) => n.value_outputs_mut(),
            NodeKind::Simple(n) => n.value_outputs_mut(),
            NodeKind::UniformBinding(n) => n.value_outputs_mut(),
            NodeKind::StorageBinding(n) => n.value_outputs_mut(),
            NodeKind::WorkgroupBinding(n) => n.value_outputs_mut(),
            NodeKind::Constant(n) => n.value_outputs_mut(),
            NodeKind::Function(n) => n.value_outputs_mut(),
        }
    }

    fn state(&self) -> Option<&State> {
        match &self.kind {
            NodeKind::Switch(n) => n.state(),
            NodeKind::Loop(n) => n.state(),
            NodeKind::Simple(n) => n.state(),
            NodeKind::UniformBinding(n) => n.state(),
            NodeKind::StorageBinding(n) => n.state(),
            NodeKind::WorkgroupBinding(n) => n.state(),
            NodeKind::Constant(n) => n.state(),
            NodeKind::Function(n) => n.state(),
        }
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        match &mut self.kind {
            NodeKind::Switch(n) => n.state_mut(),
            NodeKind::Loop(n) => n.state_mut(),
            NodeKind::Simple(n) => n.state_mut(),
            NodeKind::UniformBinding(n) => n.state_mut(),
            NodeKind::StorageBinding(n) => n.state_mut(),
            NodeKind::WorkgroupBinding(n) => n.state_mut(),
            NodeKind::Constant(n) => n.state_mut(),
            NodeKind::Function(n) => n.state_mut(),
        }
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct UniformBindingNode {
    binding: UniformBinding,
    output: ValueOutput,
}

impl UniformBindingNode {
    pub fn binding(&self) -> UniformBinding {
        self.binding
    }

    pub fn output(&self) -> &ValueOutput {
        &self.output
    }
}

impl Connectivity for UniformBindingNode {
    fn value_inputs(&self) -> &[ValueInput] {
        &[]
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        &mut []
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        slice::from_ref(&self.output)
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        slice::from_mut(&mut self.output)
    }

    fn state(&self) -> Option<&State> {
        None
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        None
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct StorageBindingNode {
    binding: StorageBinding,
    output: ValueOutput,
}

impl StorageBindingNode {
    pub fn binding(&self) -> StorageBinding {
        self.binding
    }

    pub fn output(&self) -> &ValueOutput {
        &self.output
    }
}

impl Connectivity for StorageBindingNode {
    fn value_inputs(&self) -> &[ValueInput] {
        &[]
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        &mut []
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        slice::from_ref(&self.output)
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        slice::from_mut(&mut self.output)
    }

    fn state(&self) -> Option<&State> {
        None
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        None
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct WorkgroupBindingNode {
    binding: WorkgroupBinding,
    output: ValueOutput,
}

impl WorkgroupBindingNode {
    pub fn binding(&self) -> WorkgroupBinding {
        self.binding
    }

    pub fn output(&self) -> &ValueOutput {
        &self.output
    }
}

impl Connectivity for WorkgroupBindingNode {
    fn value_inputs(&self) -> &[ValueInput] {
        &[]
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        &mut []
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        slice::from_ref(&self.output)
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        slice::from_mut(&mut self.output)
    }

    fn state(&self) -> Option<&State> {
        None
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        None
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct ConstantNode {
    constant: Constant,
    output: ValueOutput,
}

impl ConstantNode {
    pub fn constant(&self) -> Constant {
        self.constant
    }

    pub fn output(&self) -> &ValueOutput {
        &self.output
    }
}

impl Connectivity for ConstantNode {
    fn value_inputs(&self) -> &[ValueInput] {
        &[]
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        &mut []
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        slice::from_ref(&self.output)
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        slice::from_mut(&mut self.output)
    }

    fn state(&self) -> Option<&State> {
        None
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        None
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct SwitchNode {
    value_inputs: Vec<ValueInput>,
    value_outputs: Vec<ValueOutput>,
    branches: Vec<Region>,
    state: Option<State>,
}

impl SwitchNode {
    pub fn predicate(&self) -> &ValueInput {
        &self.value_inputs[0]
    }

    pub fn entry_inputs(&self) -> &[ValueInput] {
        &self.value_inputs[1..]
    }

    pub fn branches(&self) -> &[Region] {
        &self.branches
    }
}

impl Connectivity for SwitchNode {
    fn value_inputs(&self) -> &[ValueInput] {
        &self.value_inputs
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        &mut self.value_inputs
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        &self.value_outputs
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        &mut self.value_outputs
    }

    fn state(&self) -> Option<&State> {
        self.state.as_ref()
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        self.state.as_mut()
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct LoopNode {
    value_inputs: Vec<ValueInput>,
    value_outputs: Vec<ValueOutput>,
    state: Option<State>,
    loop_region: Region,
}

impl LoopNode {
    pub fn loop_region(&self) -> Region {
        self.loop_region
    }
}

impl Connectivity for LoopNode {
    fn value_inputs(&self) -> &[ValueInput] {
        &self.value_inputs
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        &mut self.value_inputs
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        &self.value_outputs
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        &mut self.value_outputs
    }

    fn state(&self) -> Option<&State> {
        self.state.as_ref()
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        self.state.as_mut()
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpAlloca {
    ty: Type,
    value_output: ValueOutput,
}

impl OpAlloca {
    pub fn ty(&self) -> Type {
        self.ty
    }

    pub fn value_output(&self) -> &ValueOutput {
        &self.value_output
    }
}

impl Connectivity for OpAlloca {
    fn value_inputs(&self) -> &[ValueInput] {
        &[]
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        &mut []
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        slice::from_ref(&self.value_output)
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        slice::from_mut(&mut self.value_output)
    }

    fn state(&self) -> Option<&State> {
        None
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        None
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpLoad {
    ptr_input: ValueInput,
    value_output: ValueOutput,
    state: State,
}

impl OpLoad {
    pub fn ptr_input(&self) -> &ValueInput {
        &self.ptr_input
    }

    pub fn value_output(&self) -> &ValueOutput {
        &self.value_output
    }
}

impl Connectivity for OpLoad {
    fn value_inputs(&self) -> &[ValueInput] {
        slice::from_ref(&self.ptr_input)
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        slice::from_mut(&mut self.ptr_input)
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        slice::from_ref(&self.value_output)
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        slice::from_mut(&mut self.value_output)
    }

    fn state(&self) -> Option<&State> {
        Some(&self.state)
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        Some(&mut self.state)
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpStore {
    value_inputs: [ValueInput; 2],
    state: State,
}

impl OpStore {
    pub fn ptr_input(&self) -> &ValueInput {
        &self.value_inputs[0]
    }

    pub fn value_input(&self) -> &ValueInput {
        &self.value_inputs[1]
    }
}

impl Connectivity for OpStore {
    fn value_inputs(&self) -> &[ValueInput] {
        &self.value_inputs
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        &mut self.value_inputs
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        &[]
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        &mut []
    }

    fn state(&self) -> Option<&State> {
        Some(&self.state)
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        Some(&mut self.state)
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpPtrElementPtr {
    element_ty: Type,
    inputs: Vec<ValueInput>,
    output: ValueOutput,
}

impl OpPtrElementPtr {
    pub fn element_ty(&self) -> Type {
        self.element_ty
    }

    pub fn ptr_input(&self) -> &ValueInput {
        &self.inputs[0]
    }

    pub fn index_inputs(&self) -> &[ValueInput] {
        &self.inputs[1..]
    }

    pub fn output(&self) -> &ValueOutput {
        &self.output
    }
}

impl Connectivity for OpPtrElementPtr {
    fn value_inputs(&self) -> &[ValueInput] {
        &self.inputs
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        &mut self.inputs
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        slice::from_ref(&self.output)
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        slice::from_mut(&mut self.output)
    }

    fn state(&self) -> Option<&State> {
        None
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        None
    }
}

/// Takes a pointer to an enum as its input and outputs a pointer the the enum's discriminant.
///
/// This is a temporary node kind used by enum-replacement. To represent code that interacts with an
/// enum's discriminant, use [OpGetDiscriminant] and [OpSetDiscriminant] instead.
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpPtrDiscriminantPtr {
    input: ValueInput,
    output: ValueOutput,
}

impl OpPtrDiscriminantPtr {
    pub fn input(&self) -> &ValueInput {
        &self.input
    }

    pub fn output(&self) -> &ValueOutput {
        &self.output
    }
}

impl Connectivity for OpPtrDiscriminantPtr {
    fn value_inputs(&self) -> &[ValueInput] {
        slice::from_ref(&self.input)
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        slice::from_mut(&mut self.input)
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        slice::from_ref(&self.output)
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        slice::from_mut(&mut self.output)
    }

    fn state(&self) -> Option<&State> {
        None
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        None
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpPtrVariantPtr {
    variant_index: u32,
    input: ValueInput,
    output: ValueOutput,
}

impl OpPtrVariantPtr {
    pub fn variant_index(&self) -> u32 {
        self.variant_index
    }

    pub fn input(&self) -> &ValueInput {
        &self.input
    }

    pub fn output(&self) -> &ValueOutput {
        &self.output
    }
}

impl Connectivity for OpPtrVariantPtr {
    fn value_inputs(&self) -> &[ValueInput] {
        slice::from_ref(&self.input)
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        slice::from_mut(&mut self.input)
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        slice::from_ref(&self.output)
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        slice::from_mut(&mut self.output)
    }

    fn state(&self) -> Option<&State> {
        None
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        None
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpExtractElement {
    element_ty: Type,
    inputs: Vec<ValueInput>,
    output: ValueOutput,
}

impl OpExtractElement {
    pub fn element_ty(&self) -> Type {
        self.element_ty
    }

    pub fn aggregate(&self) -> &ValueInput {
        &self.inputs[0]
    }

    pub fn indices(&self) -> &[ValueInput] {
        &self.inputs[1..]
    }

    pub fn output(&self) -> &ValueOutput {
        &self.output
    }
}

impl Connectivity for OpExtractElement {
    fn value_inputs(&self) -> &[ValueInput] {
        &self.inputs
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        &mut self.inputs
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        slice::from_ref(&self.output)
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        slice::from_mut(&mut self.output)
    }

    fn state(&self) -> Option<&State> {
        None
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        None
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpGetDiscriminant {
    input: ValueInput,
    output: ValueOutput,
    state: State,
}

impl OpGetDiscriminant {
    pub fn input(&self) -> &ValueInput {
        &self.input
    }

    pub fn output(&self) -> &ValueOutput {
        &self.output
    }
}

impl Connectivity for OpGetDiscriminant {
    fn value_inputs(&self) -> &[ValueInput] {
        slice::from_ref(&self.input)
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        slice::from_mut(&mut self.input)
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        slice::from_ref(&self.output)
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        slice::from_mut(&mut self.output)
    }

    fn state(&self) -> Option<&State> {
        Some(&self.state)
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        Some(&mut self.state)
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpSetDiscriminant {
    variant_index: u32,
    input: ValueInput,
    state: State,
}

impl OpSetDiscriminant {
    pub fn input(&self) -> &ValueInput {
        &self.input
    }

    pub fn variant_index(&self) -> u32 {
        self.variant_index
    }
}

impl Connectivity for OpSetDiscriminant {
    fn value_inputs(&self) -> &[ValueInput] {
        slice::from_ref(&self.input)
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        slice::from_mut(&mut self.input)
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        &[]
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        &mut []
    }

    fn state(&self) -> Option<&State> {
        Some(&self.state)
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        Some(&mut self.state)
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpAddPtrOffset {
    inputs: [ValueInput; 2],
    output: ValueOutput,
}

impl OpAddPtrOffset {
    pub fn slice_ptr(&self) -> &ValueInput {
        &self.inputs[0]
    }

    pub fn offset(&self) -> &ValueInput {
        &self.inputs[1]
    }

    pub fn output(&self) -> &ValueOutput {
        &self.output
    }
}

impl Connectivity for OpAddPtrOffset {
    fn value_inputs(&self) -> &[ValueInput] {
        &self.inputs
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        &mut self.inputs
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        slice::from_ref(&self.output)
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        slice::from_mut(&mut self.output)
    }

    fn state(&self) -> Option<&State> {
        None
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        None
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpGetPtrOffset {
    slice_ptr: ValueInput,
    output: ValueOutput,
}

impl OpGetPtrOffset {
    pub fn ptr(&self) -> &ValueInput {
        &self.slice_ptr
    }

    pub fn output(&self) -> &ValueOutput {
        &self.output
    }
}

impl Connectivity for OpGetPtrOffset {
    fn value_inputs(&self) -> &[ValueInput] {
        slice::from_ref(&self.slice_ptr)
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        slice::from_mut(&mut self.slice_ptr)
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        slice::from_ref(&self.output)
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        slice::from_mut(&mut self.output)
    }

    fn state(&self) -> Option<&State> {
        None
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        None
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpCall {
    value_inputs: Vec<ValueInput>,
    value_output: Option<ValueOutput>,
    state: State,
}

impl OpCall {
    pub fn fn_input(&self) -> &ValueInput {
        &self.value_inputs[0]
    }

    pub fn argument_inputs(&self) -> &[ValueInput] {
        &self.value_inputs[1..]
    }

    pub fn resolve_fn(&self, module: &Module) -> Function {
        *module.ty.kind(self.value_inputs[0].ty).expect_fn()
    }
}

impl Connectivity for OpCall {
    fn value_inputs(&self) -> &[ValueInput] {
        &self.value_inputs
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        &mut self.value_inputs
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        self.value_output.as_slice()
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        self.value_output.as_mut_slice()
    }

    fn state(&self) -> Option<&State> {
        Some(&self.state)
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        Some(&mut self.state)
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpCallBuiltin {
    callee: BuiltinFunction,
    value_inputs: Vec<ValueInput>,
    value_output: Option<ValueOutput>,
}

impl OpCallBuiltin {
    pub fn callee(&self) -> &BuiltinFunction {
        &self.callee
    }

    pub fn argument_inputs(&self) -> &[ValueInput] {
        &self.value_inputs
    }

    pub fn value_output(&self) -> Option<&ValueOutput> {
        self.value_output.as_ref()
    }
}

impl Connectivity for OpCallBuiltin {
    fn value_inputs(&self) -> &[ValueInput] {
        &self.value_inputs
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        &mut self.value_inputs
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        self.value_output.as_slice()
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        self.value_output.as_mut_slice()
    }

    fn state(&self) -> Option<&State> {
        None
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        None
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpUnary {
    operator: UnaryOperator,
    input: ValueInput,
    output: ValueOutput,
}

impl OpUnary {
    pub fn operator(&self) -> UnaryOperator {
        self.operator
    }

    pub fn input(&self) -> &ValueInput {
        &self.input
    }

    pub fn output(&self) -> &ValueOutput {
        &self.output
    }
}

impl Connectivity for OpUnary {
    fn value_inputs(&self) -> &[ValueInput] {
        slice::from_ref(&self.input)
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        slice::from_mut(&mut self.input)
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        slice::from_ref(&self.output)
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        slice::from_mut(&mut self.output)
    }

    fn state(&self) -> Option<&State> {
        None
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        None
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpBinary {
    operator: BinaryOperator,
    inputs: [ValueInput; 2],
    output: ValueOutput,
}

impl OpBinary {
    pub fn operator(&self) -> BinaryOperator {
        self.operator
    }

    pub fn lhs_input(&self) -> &ValueInput {
        &self.inputs[0]
    }

    pub fn rhs_input(&self) -> &ValueInput {
        &self.inputs[1]
    }

    pub fn output(&self) -> &ValueOutput {
        &self.output
    }
}

impl Connectivity for OpBinary {
    fn value_inputs(&self) -> &[ValueInput] {
        &self.inputs
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        &mut self.inputs
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        slice::from_ref(&self.output)
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        slice::from_mut(&mut self.output)
    }

    fn state(&self) -> Option<&State> {
        None
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        None
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpVector {
    vector_ty: ty::Vector,
    inputs: Vec<ValueInput>,
    output: ValueOutput,
}

impl OpVector {
    pub fn vector_ty(&self) -> &ty::Vector {
        &self.vector_ty
    }

    pub fn inputs(&self) -> &[ValueInput] {
        &self.inputs
    }

    pub fn output(&self) -> &ValueOutput {
        &self.output
    }
}

impl Connectivity for OpVector {
    fn value_inputs(&self) -> &[ValueInput] {
        &self.inputs
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        &mut self.inputs
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        slice::from_ref(&self.output)
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        slice::from_mut(&mut self.output)
    }

    fn state(&self) -> Option<&State> {
        None
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        None
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpMatrix {
    matrix_ty: ty::Matrix,
    inputs: Vec<ValueInput>,
    output: ValueOutput,
}

impl OpMatrix {
    pub fn matrix_ty(&self) -> &ty::Matrix {
        &self.matrix_ty
    }

    pub fn inputs(&self) -> &[ValueInput] {
        &self.inputs
    }

    pub fn output(&self) -> &ValueOutput {
        &self.output
    }
}

impl Connectivity for OpMatrix {
    fn value_inputs(&self) -> &[ValueInput] {
        &self.inputs
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        &mut self.inputs
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        slice::from_ref(&self.output)
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        slice::from_mut(&mut self.output)
    }

    fn state(&self) -> Option<&State> {
        None
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        None
    }
}

macro_rules! gen_const_nodes {
    ($($name:ident: $ty:ident,)*) => {
        $(
            #[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
            pub struct $name {
                value: $ty,
                output: ValueOutput,
            }

            impl $name {
                pub fn value(&self) -> $ty {
                    self.value
                }

                pub fn output(&self) -> &ValueOutput {
                    &self.output
                }
            }

            impl Connectivity for $name {
                fn value_inputs(&self) -> &[ValueInput] {
                    &[]
                }

                fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
                    &mut []
                }

                fn value_outputs(&self) -> &[ValueOutput] {
                    slice::from_ref(&self.output)
                }

                fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
                    slice::from_mut(&mut self.output)
                }

                fn state(&self) -> Option<&State> {
                    None
                }

                fn state_mut(&mut self) -> Option<&mut State> {
                    None
                }
            }
        )*
    };
}

gen_const_nodes! {
    ConstU32: u32,
    ConstI32: i32,
    ConstF32: f32,
    ConstBool: bool,
    ConstPredicate: u32,
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct ConstPtr {
    base: ValueInput,
    output: ValueOutput,
    pointee_ty: Type,
}

impl ConstPtr {
    pub fn base(&self) -> &ValueInput {
        &self.base
    }

    pub fn pointee_ty(&self) -> Type {
        self.pointee_ty
    }
}

impl Connectivity for ConstPtr {
    fn value_inputs(&self) -> &[ValueInput] {
        slice::from_ref(&self.base)
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        slice::from_mut(&mut self.base)
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        slice::from_ref(&self.output)
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        slice::from_mut(&mut self.output)
    }

    fn state(&self) -> Option<&State> {
        None
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        None
    }
}

/// Represents a "fallback" value of a given type that is used when a determinate (initialized)
/// value is unknown.
///
/// The actual value this will represent depends on the type.
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct ConstFallback {
    output: ValueOutput,
}

impl ConstFallback {
    pub fn ty(&self) -> Type {
        self.output.ty
    }

    pub fn output(&self) -> &ValueOutput {
        &self.output
    }
}

impl Connectivity for ConstFallback {
    fn value_inputs(&self) -> &[ValueInput] {
        &[]
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        &mut []
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        slice::from_ref(&self.output)
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        slice::from_mut(&mut self.output)
    }

    fn state(&self) -> Option<&State> {
        None
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        None
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpCaseToSwitchPredicate {
    cases: Vec<u32>,
    input: ValueInput,
    output: ValueOutput,
}

impl OpCaseToSwitchPredicate {
    pub fn cases(&self) -> &[u32] {
        &self.cases
    }

    pub fn input(&self) -> &ValueInput {
        &self.input
    }

    pub fn output(&self) -> &ValueOutput {
        &self.output
    }
}

impl Connectivity for OpCaseToSwitchPredicate {
    fn value_inputs(&self) -> &[ValueInput] {
        slice::from_ref(&self.input)
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        slice::from_mut(&mut self.input)
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        slice::from_ref(&self.output)
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        slice::from_mut(&mut self.output)
    }

    fn state(&self) -> Option<&State> {
        None
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        None
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpBoolToSwitchPredicate {
    input: ValueInput,
    output: ValueOutput,
}

impl OpBoolToSwitchPredicate {
    pub fn input(&self) -> &ValueInput {
        &self.input
    }

    pub fn output(&self) -> &ValueOutput {
        &self.output
    }
}

impl Connectivity for OpBoolToSwitchPredicate {
    fn value_inputs(&self) -> &[ValueInput] {
        slice::from_ref(&self.input)
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        slice::from_mut(&mut self.input)
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        slice::from_ref(&self.output)
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        slice::from_mut(&mut self.output)
    }

    fn state(&self) -> Option<&State> {
        None
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        None
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpU32ToSwitchPredicate {
    branch_count: u32,
    input: ValueInput,
    output: ValueOutput,
}

impl OpU32ToSwitchPredicate {
    pub fn branch_count(&self) -> u32 {
        self.branch_count
    }

    pub fn input(&self) -> &ValueInput {
        &self.input
    }

    pub fn output(&self) -> &ValueOutput {
        &self.output
    }
}

impl Connectivity for OpU32ToSwitchPredicate {
    fn value_inputs(&self) -> &[ValueInput] {
        slice::from_ref(&self.input)
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        slice::from_mut(&mut self.input)
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        slice::from_ref(&self.output)
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        slice::from_mut(&mut self.output)
    }

    fn state(&self) -> Option<&State> {
        None
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        None
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpSwitchPredicateToCase {
    cases: Vec<u32>,
    input: ValueInput,
    output: ValueOutput,
}

impl OpSwitchPredicateToCase {
    pub fn cases(&self) -> &[u32] {
        &self.cases
    }

    pub fn input(&self) -> &ValueInput {
        &self.input
    }

    pub fn output(&self) -> &ValueOutput {
        &self.output
    }
}

impl Connectivity for OpSwitchPredicateToCase {
    fn value_inputs(&self) -> &[ValueInput] {
        slice::from_ref(&self.input)
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        slice::from_mut(&mut self.input)
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        slice::from_ref(&self.output)
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        slice::from_mut(&mut self.output)
    }

    fn state(&self) -> Option<&State> {
        None
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        None
    }
}

macro_rules! gen_conversion_nodes {
    ($($name:ident),*) => {
        $(
            #[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
            pub struct $name {
                input: ValueInput,
                output: ValueOutput,
            }

            impl $name {
                pub fn input(&self) -> &ValueInput {
                    &self.input
                }

                pub fn output(&self) -> &ValueOutput {
                    &self.output
                }
            }

            impl Connectivity for $name {
                fn value_inputs(&self) -> &[ValueInput] {
                    slice::from_ref(&self.input)
                }

                fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
                    slice::from_mut(&mut self.input)
                }

                fn value_outputs(&self) -> &[ValueOutput] {
                    slice::from_ref(&self.output)
                }

                fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
                    slice::from_mut(&mut self.output)
                }

                fn state(&self) -> Option<&State> {
                    None
                }

                fn state_mut(&mut self) -> Option<&mut State> {
                    None
                }
            }
        )*
    };
}

gen_conversion_nodes!(
    OpConvertToU32,
    OpConvertToI32,
    OpConvertToF32,
    OpConvertToBool
);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default, Debug)]
pub enum ProxyKind {
    #[default]
    Generic,
    SwitchOutputReplacementMarker,
}

impl ProxyKind {
    pub fn is_switch_output_replacement_marker(&self) -> bool {
        *self == ProxyKind::SwitchOutputReplacementMarker
    }
}

#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct ValueProxy {
    proxy_kind: ProxyKind,
    input: ValueInput,
    output: ValueOutput,
}

impl ValueProxy {
    pub fn proxy_kind(&self) -> ProxyKind {
        self.proxy_kind
    }

    pub fn input(&self) -> &ValueInput {
        &self.input
    }

    pub fn output(&self) -> &ValueOutput {
        &self.output
    }
}

impl Connectivity for ValueProxy {
    fn value_inputs(&self) -> &[ValueInput] {
        slice::from_ref(&self.input)
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        slice::from_mut(&mut self.input)
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        slice::from_ref(&self.output)
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        slice::from_mut(&mut self.output)
    }

    fn state(&self) -> Option<&State> {
        None
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        None
    }
}

/// Re-aggregates a set of pointers or values to the individual parts of an aggregate into a single
/// pointer or value.
///
/// This is a temporary node that acts as a pause/continue point for aggregate-replacement; it
/// allows us to run e.g. a memory-promotion-and-legalization pass midway through the replacement
/// pass. We also use this when an aggregate pointer is the result of a [Switch] node to pause at
/// the end of each branch, until all other branches have also split the corresponding result.
///
/// This "operation" is not implementable on any back-end; it is only to be used as a temporary node
/// during RVSDG transformation, and no nodes of this kind should be left in the graph when
/// transformation is complete.
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct Reaggregation {
    inputs: Vec<ValueInput>,
    output: ValueOutput,
}

impl Reaggregation {
    pub fn original(&self) -> &ValueInput {
        &self.inputs[0]
    }

    pub fn parts(&self) -> &[ValueInput] {
        &self.inputs[1..]
    }

    pub fn output(&self) -> &ValueOutput {
        &self.output
    }
}

impl Connectivity for Reaggregation {
    fn value_inputs(&self) -> &[ValueInput] {
        &self.inputs
    }

    fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
        &mut self.inputs
    }

    fn value_outputs(&self) -> &[ValueOutput] {
        slice::from_ref(&self.output)
    }

    fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
        slice::from_mut(&mut self.output)
    }

    fn state(&self) -> Option<&State> {
        None
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        None
    }
}

macro_rules! gen_simple_node {
    ($($ty:ident,)*) => {
        #[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
        pub enum SimpleNode {
            $($ty($ty)),*
        }

        impl Connectivity for SimpleNode {
            fn value_inputs(&self) -> &[ValueInput] {
                match self {
                    $(SimpleNode::$ty(n) => n.value_inputs()),*
                }
            }

            fn value_inputs_mut(&mut self) -> &mut [ValueInput] {
                match self {
                    $(SimpleNode::$ty(n) => n.value_inputs_mut()),*
                }
            }

            fn value_outputs(&self) -> &[ValueOutput] {
                match self {
                    $(SimpleNode::$ty(n) => n.value_outputs()),*
                }
            }

            fn value_outputs_mut(&mut self) -> &mut [ValueOutput] {
                match self {
                    $(SimpleNode::$ty(n) => n.value_outputs_mut()),*
                }
            }

            fn state(&self) -> Option<&State> {
                match self {
                    $(SimpleNode::$ty(n) => n.state()),*
                }
            }

            fn state_mut(&mut self) -> Option<&mut State> {
                match self {
                    $(SimpleNode::$ty(n) => n.state_mut()),*
                }
            }
        }

        $(impl From<$ty> for SimpleNode {
            fn from(v: $ty) -> Self {
                SimpleNode::$ty(v)
            }
        })*
    };
}

gen_simple_node! {
    ConstU32,
    ConstI32,
    ConstF32,
    ConstBool,
    ConstPredicate,
    ConstPtr,
    ConstFallback,
    OpAlloca,
    OpLoad,
    OpStore,
    OpPtrElementPtr,
    OpPtrDiscriminantPtr,
    OpPtrVariantPtr,
    OpExtractElement,
    OpGetDiscriminant,
    OpSetDiscriminant,
    OpAddPtrOffset,
    OpGetPtrOffset,
    OpCall,
    OpCallBuiltin,
    OpUnary,
    OpBinary,
    OpVector,
    OpMatrix,
    OpCaseToSwitchPredicate,
    OpBoolToSwitchPredicate,
    OpU32ToSwitchPredicate,
    OpSwitchPredicateToCase,
    OpConvertToU32,
    OpConvertToI32,
    OpConvertToF32,
    OpConvertToBool,
    ValueProxy,
    Reaggregation,
}

macro_rules! add_const_methods {
    ($($method:ident $node:ident $ty:ident $ty_handle:ident,)*) => {
        $(
            pub fn $method(&mut self, region: Region, value: $ty) -> Node {
                let node = self.nodes.insert(NodeData {
                    kind: NodeKind::Simple(
                        $node {
                            value,
                            output: ValueOutput::new($ty_handle),
                        }
                        .into(),
                    ),
                    region: Some(region),
                });

                self.regions[region].nodes.insert(node);

                node
            }
        )*
    };
}

#[derive(Clone, Deserialize, Debug)]
pub struct RvsdgData {
    regions: SlotMap<Region, RegionData>,
    nodes: SlotMap<Node, NodeData>,
    global_region: Region,
    function_node: FxHashMap<Function, Node>,
}

#[derive(Clone, Serialize, Debug)]
pub struct Rvsdg {
    #[serde(skip_serializing)]
    ty: TypeRegistry,
    regions: SlotMap<Region, RegionData>,
    nodes: SlotMap<Node, NodeData>,
    global_region: Region,
    function_node: FxHashMap<Function, Node>,
}

impl Rvsdg {
    pub fn new(type_registry: TypeRegistry) -> Self {
        let mut regions = SlotMap::default();
        let global_region = regions.insert(RegionData {
            owner: None,
            nodes: Default::default(),
            value_arguments: vec![],
            value_results: vec![],
            state_argument: StateUser::Result,
            state_result: StateOrigin::Argument,
        });

        Rvsdg {
            ty: type_registry,
            regions,
            nodes: Default::default(),
            global_region,
            function_node: Default::default(),
        }
    }

    pub fn from_ty_and_data(ty: TypeRegistry, data: RvsdgData) -> Self {
        let RvsdgData {
            regions,
            nodes,
            global_region,
            function_node,
        } = data;

        Rvsdg {
            ty,
            regions,
            nodes,
            global_region,
            function_node,
        }
    }

    pub fn ty(&self) -> &TypeRegistry {
        &self.ty
    }

    pub fn global_region(&self) -> Region {
        self.global_region
    }

    pub fn register_uniform_binding(&mut self, module: &Module, binding: UniformBinding) -> Node {
        let ty = module.uniform_bindings[binding].ty;

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::UniformBinding(UniformBindingNode {
                binding,
                output: ValueOutput::new(ty),
            }),
            region: Some(self.global_region),
        });

        self.regions[self.global_region].nodes.insert(node);

        node
    }

    pub fn register_storage_binding(&mut self, module: &Module, binding: StorageBinding) -> Node {
        let ty = module.storage_bindings[binding].ty;

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::StorageBinding(StorageBindingNode {
                binding,
                output: ValueOutput::new(ty),
            }),
            region: Some(self.global_region),
        });

        self.regions[self.global_region].nodes.insert(node);

        node
    }

    pub fn register_workgroup_binding(
        &mut self,
        module: &Module,
        binding: WorkgroupBinding,
    ) -> Node {
        let ty = module.workgroup_bindings[binding].ty;

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::WorkgroupBinding(WorkgroupBindingNode {
                binding,
                output: ValueOutput::new(ty),
            }),
            region: Some(self.global_region),
        });

        self.regions[self.global_region].nodes.insert(node);

        node
    }

    pub fn register_constant(&mut self, module: &Module, constant: Constant) -> Node {
        let ty = module.constants[constant].ty();

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Constant(ConstantNode {
                constant,
                output: ValueOutput::new(ty),
            }),
            region: Some(self.global_region),
        });

        self.regions[self.global_region].nodes.insert(node);

        node
    }

    pub fn register_function(
        &mut self,
        module: &Module,
        function: Function,
        dependencies: impl IntoIterator<Item = Node>,
    ) -> (Node, Region) {
        let sig = &module.fn_sigs[function];

        let dep_nodes = dependencies.into_iter().collect::<Vec<_>>();
        let dependencies: Vec<ValueInput> = dep_nodes
            .iter()
            .copied()
            .map(|dep| {
                let ty = self[dep].value_outputs()[0].ty;

                ValueInput {
                    ty,
                    origin: ValueOrigin::Output {
                        producer: dep,
                        output: 0,
                    },
                }
            })
            .collect();

        let mut region_arguments: Vec<ValueOutput> = dependencies
            .iter()
            .map(|d| ValueOutput::new(d.ty))
            .collect();

        region_arguments.extend(sig.args.iter().map(|a| ValueOutput::new(a.ty)));

        let region_results = sig
            .ret_ty
            .iter()
            .map(|ty| ValueInput::placeholder(*ty))
            .collect();
        let region = self.regions.insert(RegionData {
            owner: None,
            nodes: Default::default(),
            value_arguments: region_arguments,
            value_results: region_results,
            state_argument: StateUser::Result,
            state_result: StateOrigin::Argument,
        });

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Function(FunctionNode {
                dependencies: dependencies.clone(),
                output: ValueOutput::new(sig.ty),
                region,
            }),
            region: Some(self.global_region),
        });

        self.regions[region].owner = Some(node);
        self.function_node.insert(function, node);
        self.regions[self.global_region].nodes.insert(node);

        for (i, dep) in dep_nodes.into_iter().enumerate() {
            self.nodes[dep].value_outputs_mut()[0]
                .users
                .insert(ValueUser::Input {
                    consumer: node,
                    input: i as u32,
                })
        }

        (node, region)
    }

    pub fn get_function_node(&self, function: Function) -> Option<Node> {
        self.function_node.get(&function).copied()
    }

    pub fn is_live_region(&self, region: Region) -> bool {
        self.regions.contains_key(region)
    }

    pub fn is_live_node(&self, node: Node) -> bool {
        self.nodes.contains_key(node)
    }

    /// Adds a switch node to the given `region`.
    ///
    /// Must supply the `value_inputs` and `value_outputs` for the node at creation. May optionally
    /// supply a `state_origin`: if supplied, then the node will be inserted into the state chain
    /// between the origin and the origin's prior user; if `None` the switch node will not be part
    /// of the state chain.
    ///
    /// The first of the `value_inputs` must be the predicate that selects which branch will be
    /// taken.
    ///
    /// The branch regions for the switch node are added after the creation of the switch node,
    /// by calling [add_switch_branch] with the [Node] handle return from this [add_switch]
    /// operation.
    pub fn add_switch(
        &mut self,
        region: Region,
        value_inputs: Vec<ValueInput>,
        value_outputs: Vec<ValueOutput>,
        state_origin: Option<StateOrigin>,
    ) -> Node {
        assert!(
            value_inputs.len() > 0,
            "a switch node must specify at least 1 value input that acts as the branch selector \
            predicate"
        );
        assert_eq!(
            value_inputs[0].ty, TY_PREDICATE,
            "first input must be a `predicate` type value"
        );

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Switch(SwitchNode {
                value_inputs,
                value_outputs,
                branches: vec![],
                state: state_origin.map(|origin| State {
                    origin,
                    user: StateUser::Result, // Temporary value
                }),
            }),
            region: Some(region),
        });

        if let Some(state_origin) = state_origin {
            self.link_state(region, node, state_origin);
        }

        self.regions[region].nodes.insert(node);
        self.connect_node_value_inputs(node);

        node
    }

    /// Adds a branch region to the given [switch_node].
    pub fn add_switch_branch(&mut self, switch_node: Node) -> Region {
        let data = self.nodes[switch_node].kind.expect_switch_mut();

        let region = self.regions.insert(RegionData {
            owner: Some(switch_node),
            nodes: Default::default(),
            value_arguments: data
                .value_inputs
                .iter()
                // The first input is the predicate that selects which branch will be taken; this
                // input is not to be passed on to the region as an argument, so we skip it.
                .skip(1)
                .map(|input| ValueOutput {
                    ty: input.ty,
                    users: Default::default(),
                })
                .collect(),
            value_results: data
                .value_outputs
                .iter()
                .map(|output| ValueInput {
                    ty: output.ty,
                    origin: ValueOrigin::placeholder(),
                })
                .collect(),
            state_argument: StateUser::Result,
            state_result: StateOrigin::Argument,
        });

        data.branches.push(region);

        region
    }

    pub fn permute_switch_branches(&mut self, switch_node: Node, permutation: &[usize]) {
        let mut branches_new = Vec::with_capacity(permutation.len());

        let data = self.nodes[switch_node].kind.expect_switch_mut();

        for index in permutation {
            branches_new.push(data.branches[*index as usize]);
        }

        data.branches = branches_new;
    }

    pub fn add_switch_input(&mut self, switch_node: Node, input: ValueInput) -> u32 {
        let region = self.nodes[switch_node].region();

        self.validate_node_value_input(region, &input);

        let node_data = self.nodes[switch_node].expect_switch_mut();
        let input_index = node_data.value_inputs.len();

        node_data.value_inputs.push(input);

        for branch in node_data.branches.iter().copied() {
            self.regions[branch]
                .value_arguments
                .push(ValueOutput::new(input.ty));
        }

        self.connect_node_value_input(region, switch_node, input_index);

        input_index as u32
    }

    pub fn remove_switch_input(&mut self, switch_node: Node, input: u32) {
        assert_ne!(input, 0, "cannot remove the branch selector input");

        let index = input as usize;
        let arg = index - 1;
        let region = self.nodes[switch_node].region();
        let node_data = self.nodes[switch_node].expect_switch_mut();
        let input_origin = node_data.value_inputs[input as usize].origin;
        let branch_count = node_data.branches.len();

        node_data.value_inputs.remove(index);

        // Correct the connections for any value_inputs after the input that was removed
        self.correct_value_input_connections(switch_node, index, -1);

        // Remove the corresponding argument in each of the branch regions
        for branch_index in 0..branch_count {
            let branch = self.nodes[switch_node].expect_switch().branches()[branch_index];
            let arguments = &mut self.regions[branch].value_arguments;

            if !arguments[arg].users.is_empty() {
                panic!(
                    "cannot remove an input if one of the corresponding arguments still has users"
                );
            }

            arguments.remove(arg);
            self.correct_region_argument_connections(branch, arg);
        }

        // Remove the input as a user from its origin
        self.remove_user(
            region,
            input_origin,
            ValueUser::Input {
                consumer: switch_node,
                input,
            },
        );
    }

    pub fn add_switch_output(&mut self, switch_node: Node, ty: Type) -> u32 {
        let node_data = self.nodes[switch_node].expect_switch_mut();
        let index = node_data.value_outputs.len();

        node_data.value_outputs.push(ValueOutput::new(ty));

        for branch in node_data.branches.iter().copied() {
            self.regions[branch]
                .value_results
                .push(ValueInput::placeholder(ty));
        }

        index as u32
    }

    pub fn remove_switch_output(&mut self, switch_node: Node, output: u32) {
        let index = output as usize;
        let node_data = self.nodes[switch_node].expect_switch_mut();
        let branch_count = node_data.branches.len();

        if !node_data.value_outputs[index].users.is_empty() {
            panic!("cannot remove an output if it still has users")
        }

        node_data.value_outputs.remove(index);

        // Correct the connections for any value_inputs to the right of the input that was removed
        self.correct_value_output_connections(switch_node, index);

        // Remove the corresponding result in each of the branch regions
        for branch_index in 0..branch_count {
            let branch = self.nodes[switch_node].expect_switch().branches()[branch_index];
            let origin = self.regions[branch].value_results[index].origin;

            self.remove_user(branch, origin, ValueUser::Result(index as u32));
            self.regions[branch].value_results.remove(index);
            self.correct_region_result_connections(branch, index, -1);
        }
    }

    pub fn link_switch_state(&mut self, switch_node: Node, state_origin: StateOrigin) {
        if self.nodes[switch_node].state().is_some() {
            panic!("switch node is already linked into the state chain")
        }

        let region = self.nodes[switch_node].region();

        self.link_state(region, switch_node, state_origin);
    }

    pub fn unlink_switch_state(&mut self, switch_node: Node, state_origin: StateOrigin) {
        if self.nodes[switch_node].state().is_none() {
            panic!("switch node is not linked into the state chain")
        }

        let region = self.nodes[switch_node].region();

        self.unlink_state(switch_node);
    }

    pub fn add_loop(
        &mut self,
        region: Region,
        value_inputs: Vec<ValueInput>,
        state_origin: Option<StateOrigin>,
    ) -> (Node, Region) {
        for input in &value_inputs {
            self.validate_node_value_input(region, input);
        }

        // The output signature and the contained region arguments and results all match the
        // signature of a loop node's inputs, so we can derive all of these from the `value_inputs`.

        let value_outputs = value_inputs
            .iter()
            .map(|input| ValueOutput::new(input.ty))
            .collect::<Vec<_>>();

        let mut loop_region_results = vec![ValueInput {
            // First result of the region is the re-entry predicate
            ty: TY_BOOL,
            origin: ValueOrigin::placeholder(),
        }];

        // The remaining results match the input signature
        loop_region_results.extend(value_inputs.iter().map(|input| ValueInput {
            ty: input.ty,
            origin: ValueOrigin::placeholder(),
        }));

        let loop_region = self.regions.insert(RegionData {
            owner: None,
            nodes: Default::default(),
            value_arguments: value_outputs.clone(),
            value_results: loop_region_results,
            state_argument: StateUser::Result,
            state_result: StateOrigin::Argument,
        });

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Loop(LoopNode {
                value_inputs,
                value_outputs,
                state: state_origin.map(|origin| State {
                    origin,
                    user: StateUser::Result, // Temporary value
                }),
                loop_region,
            }),
            region: Some(region),
        });

        self.regions[loop_region].owner = Some(node);

        if let Some(state_origin) = state_origin {
            self.link_state(region, node, state_origin);
        }

        self.regions[region].nodes.insert(node);
        self.connect_node_value_inputs(node);

        (node, loop_region)
    }

    pub fn add_loop_input(&mut self, loop_node: Node, input: ValueInput) -> u32 {
        let region = self.nodes[loop_node].region();

        self.validate_node_value_input(region, &input);

        let node_data = self.nodes[loop_node].expect_loop_mut();
        let input_index = node_data.value_inputs.len();

        node_data.value_inputs.push(input);
        node_data.value_outputs.push(ValueOutput::new(input.ty));

        let region_data = &mut self.regions[node_data.loop_region];

        region_data.value_arguments.push(ValueOutput::new(input.ty));
        region_data
            .value_results
            .push(ValueInput::placeholder(input.ty));

        self.connect_node_value_input(region, loop_node, input_index);

        input_index as u32
    }

    pub fn remove_loop_input(&mut self, loop_node: Node, input: u32) {
        let index = input as usize;

        // Loop nodes have an equal number of inputs outputs and arguments, but have 1 more result
        // (where the first result decides loop reentry), so we add one for get the index for the
        // result we're going to remove.
        let result_index = index + 1;

        let region = self.nodes[loop_node].region();
        let node_data = self.nodes[loop_node].expect_loop_mut();
        let input_origin = node_data.value_inputs[index].origin;
        let loop_region = node_data.loop_region;

        if !node_data.value_outputs[index].users.is_empty() {
            panic!("cannot remove an input if the corresponding output still has users");
        }

        if !self.regions[loop_region].value_arguments[index]
            .users
            .is_empty()
        {
            panic!("cannot remove an input if the corresponding region argument still has users");
        }

        // Remove the input the corresponding output and correct their connections
        node_data.value_inputs.remove(index);
        node_data.value_outputs.remove(index);
        self.correct_value_input_connections(loop_node, index, -1);
        self.correct_value_output_connections(loop_node, index);

        // Remove the corresponding argument and correct its connections. Its important that we
        // correct the argument's connections before we remove the result, because the argument may
        // connect directly to results that succeed the result we're removing.
        self.regions[loop_region].value_arguments.remove(index);
        self.correct_region_argument_connections(loop_region, index);

        // Remove the corresponding result and correct its connections
        self.regions[loop_region].value_results.remove(result_index);
        self.correct_region_result_connections(loop_region, result_index, -1);

        // Remove the input as a user from its origin
        self.remove_user(
            region,
            input_origin,
            ValueUser::Input {
                consumer: loop_node,
                input,
            },
        );
    }

    add_const_methods! {
        add_const_u32 ConstU32 u32 TY_U32,
        add_const_i32 ConstI32 i32 TY_I32,
        add_const_f32 ConstF32 f32 TY_F32,
        add_const_bool ConstBool bool TY_BOOL,
    }

    pub fn add_const_predicate(&mut self, region: Region, value: u32) -> Node {
        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                ConstPredicate {
                    value,
                    output: ValueOutput::new(TY_PREDICATE),
                }
                .into(),
            ),
            region: Some(region),
        });

        self.regions[region].nodes.insert(node);

        node
    }

    pub fn add_const_ptr(&mut self, region: Region, pointee_ty: Type, base: ValueInput) -> Node {
        let ptr_ty = self.ty.register(TypeKind::Ptr(pointee_ty));

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                ConstPtr {
                    base,
                    output: ValueOutput::new(ptr_ty),
                    pointee_ty,
                }
                .into(),
            ),
            region: Some(region),
        });

        self.regions[region].nodes.insert(node);
        self.connect_node_value_inputs(node);

        node
    }

    pub fn add_const_fallback(&mut self, region: Region, ty: Type) -> Node {
        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                ConstFallback {
                    output: ValueOutput::new(ty),
                }
                .into(),
            ),
            region: Some(region),
        });

        self.regions[region].nodes.insert(node);
        self.connect_node_value_inputs(node);

        node
    }

    pub fn add_op_alloca(&mut self, region: Region, ty: Type) -> Node {
        let ptr_ty = self.ty.register(TypeKind::Ptr(ty));

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                OpAlloca {
                    ty,
                    value_output: ValueOutput::new(ptr_ty),
                }
                .into(),
            ),
            region: Some(region),
        });

        self.regions[region].nodes.insert(node);
        self.connect_node_value_inputs(node);

        node
    }

    pub fn add_op_load(
        &mut self,
        region: Region,
        ptr_input: ValueInput,
        state_origin: StateOrigin,
    ) -> Node {
        self.validate_node_value_input(region, &ptr_input);

        let TypeKind::Ptr(pointee_ty) = *self.ty.kind(ptr_input.ty) else {
            panic!("`ptr_input` must be a pointer type");
        };

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                OpLoad {
                    ptr_input,
                    value_output: ValueOutput::new(pointee_ty),
                    state: State {
                        origin: state_origin,
                        user: StateUser::Result, // Temp value
                    },
                }
                .into(),
            ),
            region: Some(region),
        });

        self.link_state(region, node, state_origin);
        self.regions[region].nodes.insert(node);
        self.connect_node_value_inputs(node);

        node
    }

    pub fn add_op_store(
        &mut self,
        region: Region,
        ptr_input: ValueInput,
        value_input: ValueInput,
        state_origin: StateOrigin,
    ) -> Node {
        self.validate_node_value_input(region, &ptr_input);
        self.validate_node_value_input(region, &value_input);

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                OpStore {
                    value_inputs: [ptr_input, value_input],
                    state: State {
                        origin: state_origin,
                        user: StateUser::Result, // Temp value
                    },
                }
                .into(),
            ),
            region: Some(region),
        });

        self.link_state(region, node, state_origin);
        self.regions[region].nodes.insert(node);
        self.connect_node_value_inputs(node);

        node
    }

    pub fn add_op_ptr_element_ptr(
        &mut self,
        region: Region,
        ptr_input: ValueInput,
        index_inputs: impl IntoIterator<Item = ValueInput>,
    ) -> Node {
        self.validate_node_value_input(region, &ptr_input);

        let TypeKind::Ptr(mut element_ty) = *self.ty.kind(ptr_input.ty) else {
            panic!("`ptr_input` must be a pointer type");
        };

        let mut inputs = vec![ptr_input];

        for (i, input) in index_inputs.into_iter().enumerate() {
            self.validate_node_value_input(region, &input);

            element_ty = self.project_index((i, &input), element_ty);

            inputs.push(input);
        }

        let ptr_ty = self.ty.register(TypeKind::Ptr(element_ty));

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                OpPtrElementPtr {
                    element_ty,
                    inputs,
                    output: ValueOutput::new(ptr_ty),
                }
                .into(),
            ),
            region: Some(region),
        });

        self.regions[region].nodes.insert(node);
        self.connect_node_value_inputs(node);

        node
    }

    pub fn add_op_ptr_discriminant_ptr(&mut self, region: Region, ptr_input: ValueInput) -> Node {
        self.validate_node_value_input(region, &ptr_input);

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                OpPtrDiscriminantPtr {
                    input: ptr_input,
                    output: ValueOutput::new(TY_PTR_U32),
                }
                .into(),
            ),
            region: Some(region),
        });

        self.regions[region].nodes.insert(node);
        self.connect_node_value_inputs(node);

        node
    }

    pub fn add_op_ptr_variant_ptr(
        &mut self,
        region: Region,
        input: ValueInput,
        variant_index: u32,
    ) -> Node {
        self.validate_node_value_input(region, &input);

        let TypeKind::Ptr(pointee_ty) = *self.ty.kind(input.ty) else {
            panic!("`ptr_input` must be of a pointer type")
        };
        let pointee_kind = self.ty.kind(pointee_ty);
        let TypeKind::Enum(enum_data) = &*pointee_kind else {
            panic!("`ptr_input` must point to an enum type")
        };

        let variants = &enum_data.variants;

        let Some(variant_ty) = variants.get(variant_index as usize).copied() else {
            panic!(
                "tried to select variant `{}` on an enum that only has {} variants",
                variant_index,
                variants.len()
            )
        };

        let output_ty = self.ty.register(TypeKind::Ptr(variant_ty));

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                OpPtrVariantPtr {
                    variant_index,
                    input,
                    output: ValueOutput::new(output_ty),
                }
                .into(),
            ),
            region: Some(region),
        });

        self.regions[region].nodes.insert(node);
        self.connect_node_value_inputs(node);

        node
    }

    pub fn add_op_extract_element(
        &mut self,
        region: Region,
        aggregate_input: ValueInput,
        index_inputs: impl IntoIterator<Item = ValueInput>,
    ) -> Node {
        self.validate_node_value_input(region, &aggregate_input);

        let mut element_ty = aggregate_input.ty;

        let mut inputs = vec![aggregate_input];

        for input in index_inputs {
            self.validate_node_value_input(region, &input);

            element_ty = self.project_index((0, &input), element_ty);

            inputs.push(input);
        }

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                OpExtractElement {
                    element_ty,
                    inputs,
                    output: ValueOutput::new(element_ty),
                }
                .into(),
            ),
            region: Some(region),
        });

        self.regions[region].nodes.insert(node);
        self.connect_node_value_inputs(node);

        node
    }

    pub fn add_op_get_discriminant(
        &mut self,
        region: Region,
        ptr_input: ValueInput,
        state_origin: StateOrigin,
    ) -> Node {
        self.validate_node_value_input(region, &ptr_input);

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                OpGetDiscriminant {
                    input: ptr_input,
                    output: ValueOutput::new(TY_U32),
                    state: State {
                        origin: state_origin,
                        user: StateUser::Result, // Temp value
                    },
                }
                .into(),
            ),
            region: Some(region),
        });

        self.link_state(region, node, state_origin);
        self.regions[region].nodes.insert(node);
        self.connect_node_value_inputs(node);

        node
    }

    pub fn add_op_set_discriminant(
        &mut self,
        region: Region,
        ptr_input: ValueInput,
        variant_index: u32,
        state_origin: StateOrigin,
    ) -> Node {
        self.validate_node_value_input(region, &ptr_input);

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                OpSetDiscriminant {
                    input: ptr_input,
                    variant_index,
                    state: State {
                        origin: state_origin,
                        user: StateUser::Result, // Temp value
                    },
                }
                .into(),
            ),
            region: Some(region),
        });

        self.link_state(region, node, state_origin);
        self.regions[region].nodes.insert(node);
        self.connect_node_value_inputs(node);

        node
    }

    pub fn add_op_add_ptr_offset(
        &mut self,
        region: Region,
        slice_ptr: ValueInput,
        offset: ValueInput,
    ) -> Node {
        self.validate_node_value_input(region, &slice_ptr);
        self.validate_node_value_input(region, &offset);

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                OpAddPtrOffset {
                    inputs: [slice_ptr, offset],
                    output: ValueOutput::new(slice_ptr.ty),
                }
                .into(),
            ),
            region: Some(region),
        });

        self.regions[region].nodes.insert(node);
        self.connect_node_value_inputs(node);

        node
    }

    pub fn add_op_get_ptr_offset(&mut self, region: Region, slice_ptr: ValueInput) -> Node {
        self.validate_node_value_input(region, &slice_ptr);

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                OpGetPtrOffset {
                    slice_ptr,
                    output: ValueOutput::new(TY_U32),
                }
                .into(),
            ),
            region: Some(region),
        });

        self.regions[region].nodes.insert(node);
        self.connect_node_value_inputs(node);

        node
    }

    pub fn add_op_call(
        &mut self,
        module: &Module,
        region: Region,
        fn_input: ValueInput,
        argument_inputs: impl IntoIterator<Item = ValueInput>,
        state_origin: StateOrigin,
    ) -> Node {
        let ty_kind = module.ty.kind(fn_input.ty);
        let function = ty_kind.expect_fn();
        let sig = &module.fn_sigs[*function];
        let ret_ty = sig.ret_ty;

        let mut value_inputs = vec![fn_input];

        value_inputs.extend(argument_inputs);

        // The total length of the value_inputs also includes the function input, so subtract `1`.
        let arg_count = value_inputs.len() - 1;

        // Validate the value input arguments
        assert_eq!(
            sig.args.len(),
            arg_count,
            "function expects {} arguments, but {} were provided",
            sig.args.len(),
            arg_count
        );
        for i in 0..arg_count {
            let sig_arg_ty = sig.args[i].ty;
            let value_input_ty = value_inputs[i + 1].ty;
            //TODO
            // assert_eq!(
            //     sig_arg_ty,
            //     value_input_ty,
            //     "argument `{}` expects a value of type `{}`, but a value input of type `{}` was provided",
            //     i,
            //     sig_arg_ty.to_string(module),
            //     value_input_ty.to_string(module)
            // );
        }

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                OpCall {
                    value_inputs,
                    value_output: ret_ty.map(|ty| ValueOutput::new(ty)),
                    state: State {
                        origin: state_origin,
                        user: StateUser::Result, // Temp value
                    },
                }
                .into(),
            ),
            region: Some(region),
        });

        self.link_state(region, node, state_origin);
        self.regions[region].nodes.insert(node);
        self.connect_node_value_inputs(node);

        node
    }

    pub fn add_op_call_builtin(
        &mut self,
        module: &Module,
        region: Region,
        callee: BuiltinFunction,
        argument_inputs: impl IntoIterator<Item = ValueInput>,
    ) -> Node {
        let mut value_inputs = argument_inputs.into_iter().collect::<Vec<_>>();

        // The total length of the value_inputs also includes the function input, so subtract `1`.
        let arg_count = value_inputs.len();

        // Validate the value input arguments
        assert_eq!(
            callee.arguments().len(),
            arg_count,
            "function expects {} arguments, but {} were provided",
            callee.arguments().len(),
            arg_count
        );
        for i in 0..arg_count {
            let sig_arg_ty = callee.arguments()[i];
            let value_input_ty = value_inputs[i].ty;

            assert_eq!(
                sig_arg_ty,
                value_input_ty,
                "argument `{}` expects a value of type `{}`, but a value input of type `{}` was provided",
                i,
                sig_arg_ty.to_string(self.ty()),
                value_input_ty.to_string(self.ty())
            );
        }

        let value_output = callee.return_type().map(|ty| ValueOutput::new(ty));

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                OpCallBuiltin {
                    callee,
                    value_inputs,
                    value_output,
                }
                .into(),
            ),
            region: Some(region),
        });

        self.regions[region].nodes.insert(node);
        self.connect_node_value_inputs(node);

        node
    }

    pub fn add_op_unary(
        &mut self,
        region: Region,
        operator: UnaryOperator,
        input: ValueInput,
    ) -> Node {
        self.validate_node_value_input(region, &input);

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                OpUnary {
                    operator,
                    input,
                    output: ValueOutput::new(input.ty),
                }
                .into(),
            ),
            region: Some(region),
        });

        self.regions[region].nodes.insert(node);
        self.connect_node_value_inputs(node);

        node
    }

    pub fn add_op_binary(
        &mut self,
        region: Region,
        operator: BinaryOperator,
        lhs_input: ValueInput,
        rhs_input: ValueInput,
    ) -> Node {
        self.validate_node_value_input(region, &lhs_input);
        self.validate_node_value_input(region, &rhs_input);

        let output_ty = match self
            .ty()
            .check_binary_op(operator, lhs_input.ty, rhs_input.ty)
        {
            Ok(ty) => ty,
            Err(err) => panic!("invalid operation: {}", err),
        };

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                OpBinary {
                    operator,
                    inputs: [lhs_input, rhs_input],
                    output: ValueOutput::new(output_ty),
                }
                .into(),
            ),
            region: Some(region),
        });

        self.regions[region].nodes.insert(node);
        self.connect_node_value_inputs(node);

        node
    }

    pub fn add_op_vector(
        &mut self,
        region: Region,
        vector_ty: ty::Vector,
        inputs: impl IntoIterator<Item = ValueInput>,
    ) -> Node {
        let size = vector_ty.size.to_usize();
        let mut elements = Vec::with_capacity(size);
        let mut iter = inputs.into_iter();

        for i in 0..size {
            let Some(input) = iter.next() else {
                panic!(
                    "expected at least {} elements for a vector of type `{}` (found only {})",
                    size, vector_ty, i
                );
            };

            self.validate_node_value_input(region, &input);

            let TypeKind::Scalar(s) = *self.ty().kind(input.ty) else {
                panic!(
                    "expected all vector element inputs to be `{}` values (element `{}` was of \
                    type `{}`)",
                    vector_ty.scalar,
                    i,
                    input.ty.to_string(self.ty())
                );
            };

            if s != vector_ty.scalar {
                panic!(
                    "expected all vector element inputs to be `{}` values (element `{}` was of \
                    type `{}`)",
                    vector_ty.scalar,
                    i,
                    input.ty.to_string(self.ty())
                );
            }

            elements.push(input);
        }

        if let Some(_) = iter.next() {
            panic!(
                "expected only {} elements for a vector of type `{}`, but more were provided",
                size, vector_ty
            );
        }

        let output_ty = self.ty().register(TypeKind::Vector(vector_ty));

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                OpVector {
                    vector_ty,
                    inputs: elements,
                    output: ValueOutput::new(output_ty),
                }
                .into(),
            ),
            region: Some(region),
        });

        self.regions[region].nodes.insert(node);
        self.connect_node_value_inputs(node);

        node
    }

    pub fn add_op_matrix(
        &mut self,
        region: Region,
        matrix_ty: ty::Matrix,
        inputs: impl IntoIterator<Item = ValueInput>,
    ) -> Node {
        let columns = matrix_ty.columns.to_usize();
        let expected_vector_ty = ty::Vector {
            scalar: matrix_ty.scalar,
            size: matrix_ty.rows,
        };

        let mut collected_inputs = Vec::with_capacity(columns);
        let mut iter = inputs.into_iter();

        for i in 0..columns {
            let Some(input) = iter.next() else {
                panic!(
                    "expected at least {} columns for a matrix of type `{}` (found only {})",
                    columns, matrix_ty, i
                );
            };

            self.validate_node_value_input(region, &input);

            let TypeKind::Vector(v) = *self.ty().kind(input.ty) else {
                panic!(
                    "expected all column inputs to be `{}` values (element `{}` was of type `{}`)",
                    expected_vector_ty,
                    i,
                    input.ty.to_string(self.ty())
                );
            };

            if v != expected_vector_ty {
                panic!(
                    "expected all column inputs to be `{}` values (element `{}` was of type `{}`)",
                    expected_vector_ty,
                    i,
                    input.ty.to_string(self.ty())
                );
            }

            collected_inputs.push(input);
        }

        if let Some(_) = iter.next() {
            panic!(
                "expected only {} columns for a matrix of type `{}`, but more were provided",
                columns, matrix_ty
            );
        }

        let output_ty = self.ty().register(TypeKind::Matrix(matrix_ty));

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                OpMatrix {
                    matrix_ty,
                    inputs: collected_inputs,
                    output: ValueOutput::new(output_ty),
                }
                .into(),
            ),
            region: Some(region),
        });

        self.regions[region].nodes.insert(node);
        self.connect_node_value_inputs(node);

        node
    }

    pub fn add_op_case_to_switch_predicate(
        &mut self,
        region: Region,
        input: ValueInput,
        cases: impl IntoIterator<Item = u32>,
    ) -> Node {
        self.validate_node_value_input(region, &input);
        assert_eq!(input.ty, TY_U32, "input must by a `u32` value");

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                OpCaseToSwitchPredicate {
                    cases: cases.into_iter().collect(),
                    input,
                    output: ValueOutput::new(TY_PREDICATE),
                }
                .into(),
            ),
            region: Some(region),
        });

        self.regions[region].nodes.insert(node);
        self.connect_node_value_inputs(node);

        node
    }

    pub fn permute_op_case_to_switch_predicate_cases(&mut self, node: Node, permutation: &[usize]) {
        let mut new_cases = Vec::with_capacity(permutation.len());

        let data = self.nodes[node].expect_op_case_to_switch_predicate_mut();
        let cases = data.cases();

        for index in permutation {
            new_cases.push(cases[*index]);
        }

        data.cases = new_cases;
    }

    pub fn add_op_bool_to_switch_predicate(&mut self, region: Region, input: ValueInput) -> Node {
        self.validate_node_value_input(region, &input);
        assert_eq!(input.ty, TY_BOOL, "input must by a `bool` value");

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                OpBoolToSwitchPredicate {
                    input,
                    output: ValueOutput::new(TY_PREDICATE),
                }
                .into(),
            ),
            region: Some(region),
        });

        self.regions[region].nodes.insert(node);
        self.connect_node_value_inputs(node);

        node
    }

    pub fn add_op_u32_to_switch_predicate(
        &mut self,
        region: Region,
        branch_count: u32,
        input: ValueInput,
    ) -> Node {
        self.validate_node_value_input(region, &input);
        assert_eq!(input.ty, TY_U32, "input must by a `u32` value");

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                OpU32ToSwitchPredicate {
                    branch_count,
                    input,
                    output: ValueOutput::new(TY_PREDICATE),
                }
                .into(),
            ),
            region: Some(region),
        });

        self.regions[region].nodes.insert(node);
        self.connect_node_value_inputs(node);

        node
    }

    pub fn add_op_switch_predicate_to_case(
        &mut self,
        region: Region,
        input: ValueInput,
        cases: impl IntoIterator<Item = u32>,
    ) -> Node {
        self.validate_node_value_input(region, &input);
        assert_eq!(input.ty, TY_PREDICATE, "input must by a `predicate` value");

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                OpSwitchPredicateToCase {
                    cases: cases.into_iter().collect(),
                    input,
                    output: ValueOutput::new(TY_U32),
                }
                .into(),
            ),
            region: Some(region),
        });

        self.regions[region].nodes.insert(node);
        self.connect_node_value_inputs(node);

        node
    }

    pub fn add_op_convert_to_u32(&mut self, region: Region, input: ValueInput) -> Node {
        self.validate_node_value_input(region, &input);

        if !input.ty.is_scalar() {
            panic!(
                "expected input to be a `u32`, `i32`, `f32` or `bool` value, but found `{}`",
                input.ty.to_string(self.ty())
            );
        }

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                OpConvertToU32 {
                    input,
                    output: ValueOutput::new(input.ty),
                }
                .into(),
            ),
            region: Some(region),
        });

        self.regions[region].nodes.insert(node);
        self.connect_node_value_inputs(node);

        node
    }

    pub fn add_op_convert_to_i32(&mut self, region: Region, input: ValueInput) -> Node {
        self.validate_node_value_input(region, &input);

        if !input.ty.is_scalar() {
            panic!(
                "expected input to be a `u32`, `i32`, `f32` or `bool` value, but found `{}`",
                input.ty.to_string(self.ty())
            );
        }

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                OpConvertToI32 {
                    input,
                    output: ValueOutput::new(input.ty),
                }
                .into(),
            ),
            region: Some(region),
        });

        self.regions[region].nodes.insert(node);
        self.connect_node_value_inputs(node);

        node
    }

    pub fn add_op_convert_to_f32(&mut self, region: Region, input: ValueInput) -> Node {
        self.validate_node_value_input(region, &input);

        if !input.ty.is_numeric_scalar() {
            panic!(
                "expected input to be a `u32`, `i32`, or `f32` value, but found `{}`",
                input.ty.to_string(self.ty())
            );
        }

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                OpConvertToF32 {
                    input,
                    output: ValueOutput::new(input.ty),
                }
                .into(),
            ),
            region: Some(region),
        });

        self.regions[region].nodes.insert(node);
        self.connect_node_value_inputs(node);

        node
    }

    pub fn add_op_convert_to_bool(&mut self, region: Region, input: ValueInput) -> Node {
        self.validate_node_value_input(region, &input);

        if !input.ty.is_scalar() {
            panic!(
                "expected input to be a `u32`, `i32`, `f32` or `bool` value, but found `{}`",
                input.ty.to_string(self.ty())
            );
        }

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                OpConvertToBool {
                    input,
                    output: ValueOutput::new(input.ty),
                }
                .into(),
            ),
            region: Some(region),
        });

        self.regions[region].nodes.insert(node);
        self.connect_node_value_inputs(node);

        node
    }

    pub fn add_value_proxy(
        &mut self,
        region: Region,
        input: ValueInput,
        proxy_kind: ProxyKind,
    ) -> Node {
        self.validate_node_value_input(region, &input);

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                ValueProxy {
                    proxy_kind,
                    input,
                    output: ValueOutput::new(input.ty),
                }
                .into(),
            ),
            region: Some(region),
        });

        self.regions[region].nodes.insert(node);
        self.connect_node_value_inputs(node);

        node
    }

    pub fn add_reaggregation(
        &mut self,
        region: Region,
        original: ValueInput,
        parts: impl IntoIterator<Item = ValueInput>,
    ) -> Node {
        let mut inputs = vec![original];

        inputs.extend(parts);

        for input in &inputs {
            self.validate_node_value_input(region, input);
        }

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                Reaggregation {
                    inputs,
                    output: ValueOutput::new(original.ty),
                }
                .into(),
            ),
            region: Some(region),
        });

        self.regions[region].nodes.insert(node);
        self.connect_node_value_inputs(node);

        node
    }

    pub fn reconnect_value_input(&mut self, node: Node, value_input: u32, origin: ValueOrigin) {
        let old_input = self.nodes[node].value_inputs()[value_input as usize];
        let old_origin = old_input.origin;
        let region = self.nodes[node].region();

        self.validate_node_value_input(
            region,
            &ValueInput {
                ty: old_input.ty,
                origin,
            },
        );

        let user = ValueUser::Input {
            consumer: node,
            input: value_input,
        };

        // Remove the input as a user from the old origin (if any)
        if !old_origin.is_placeholder() {
            self.remove_user(region, old_origin, user);
        }

        // Add the input as a user to the new origin
        self.add_user(region, origin, user);

        // Update the input's origin
        self.nodes[node].value_inputs_mut()[value_input as usize].origin = origin;
    }

    pub fn reconnect_region_result(&mut self, region: Region, result: u32, origin: ValueOrigin) {
        let old_input = self[region].value_results[result as usize];

        self.validate_node_value_input(
            region,
            &ValueInput {
                ty: old_input.ty,
                origin,
            },
        );

        let user = ValueUser::Result(result);

        // Remove the result as a user from the old origin (if any)
        if !old_input.origin.is_placeholder() {
            self.remove_user(region, old_input.origin, user);
        }

        // Add the result as a user to the new origin
        self.add_user(region, origin, user);

        // Update the result's origin
        self.regions[region].value_results[result as usize].origin = origin;
    }

    pub fn disconnect_region_result(&mut self, region: Region, result: u32) {
        let origin = self.regions[region].value_results[result as usize].origin;
        let user = ValueUser::Result(result);

        if !origin.is_placeholder() {
            self.remove_user(region, origin, user);
            self.regions[region].value_results[result as usize].origin = ValueOrigin::placeholder();
        }
    }

    pub fn reconnect_value_user(
        &mut self,
        region: Region,
        value_user: ValueUser,
        origin: ValueOrigin,
    ) {
        match value_user {
            ValueUser::Result(i) => self.reconnect_region_result(region, i, origin),
            ValueUser::Input { consumer, input } => {
                self.reconnect_value_input(consumer, input, origin)
            }
        }
    }

    pub fn reconnect_value_users(
        &mut self,
        region: Region,
        original_origin: ValueOrigin,
        new_origin: ValueOrigin,
    ) {
        let user_count = match original_origin {
            ValueOrigin::Argument(a) => self.regions[region].value_arguments()[a as usize]
                .users
                .len(),
            ValueOrigin::Output { producer, output } => self.nodes[producer].value_outputs()
                [output as usize]
                .users
                .len(),
        };

        for i in (0..user_count).rev() {
            let user = match original_origin {
                ValueOrigin::Argument(a) => {
                    self.regions[region].value_arguments()[a as usize].users[i]
                }
                ValueOrigin::Output { producer, output } => {
                    self.nodes[producer].value_outputs()[output as usize].users[i]
                }
            };

            self.reconnect_value_user(region, user, new_origin);
        }
    }

    /// Removes the given `node` from the RVSDG.
    ///
    /// The node should not have any users for any of its value outputs, will panic otherwise.
    pub fn remove_node(&mut self, node: Node) {
        self.try_remove_node(node).unwrap()
    }

    /// Removes the given `node` from the RVSDG if it has no value-users, or returns an error
    /// otherwise.
    pub fn try_remove_node(&mut self, node: Node) -> Result<(), RemoveNodeError> {
        let data = &self.nodes[node];

        for (i, output) in data.value_outputs().iter().enumerate() {
            if !output.users.is_empty() {
                return Err(RemoveNodeError {
                    output_with_users: i as u32,
                });
            }
        }

        let region = data.region();

        for i in 0..data.value_inputs().len() {
            let origin = self.nodes[node].value_inputs()[i].origin;

            self.remove_user(
                region,
                origin,
                ValueUser::Input {
                    consumer: node,
                    input: i as u32,
                },
            );
        }

        self.unlink_state(node);
        self.regions[region].nodes.shift_remove(&node);

        match self.nodes[node].kind() {
            NodeKind::Switch(switch_node) => {
                let branch_count = switch_node.branches().len();

                for i in 0..branch_count {
                    let branch = self.nodes[node].expect_switch().branches()[i];

                    self.remove_region_with_nodes(branch);
                }
            }
            NodeKind::Loop(loop_node) => {
                let loop_region = loop_node.loop_region();

                self.remove_region_with_nodes(loop_region);
            }
            _ => {}
        }

        self.nodes.remove(node);

        Ok(())
    }

    /// Inserts a proxy node into the given `region` in between the `origin` and the `user`.
    ///
    /// A proxy node simply passes its input on to its output unmodified. This serves no semantic
    /// function in the program, but can be useful as an intermediate construct during a transform.
    /// For example, transformations may want to concurrently transform a graph during traversal.
    /// However, removing/adding users from/to an output during concurrent iteration over that same
    /// output's users may result in double visits or skipped visits. When replacing one node with
    /// multiple nodes, rather than removing the node and adding the replacements as additional
    /// users to the original producer, you might insert a proxy and instead modify the users of the
    /// proxy.
    ///
    /// # Panics
    ///
    /// Panics if the `origin` or `user` is a node output resp. node input of a node that does not
    /// belong to the same [Region] as the given `region. Panics of the `origin` or `user` type does
    /// not match the given `ty`.
    pub fn proxy_origin_user(
        &mut self,
        region: Region,
        ty: Type,
        origin: ValueOrigin,
        user: ValueUser,
        proxy_kind: ProxyKind,
    ) -> Node {
        let proxy = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                ValueProxy {
                    proxy_kind,
                    input: ValueInput { ty, origin },
                    output: ValueOutput {
                        ty,
                        users: thin_set![user],
                    },
                }
                .into(),
            ),
            region: Some(region),
        });

        let output = match origin {
            ValueOrigin::Argument(arg) => &mut self.regions[region].value_arguments[arg as usize],
            ValueOrigin::Output { producer, output } => {
                let producer = &mut self.nodes[producer];

                assert_eq!(
                    producer.region(),
                    region,
                    "origin must be in the specified region"
                );

                &mut producer.value_outputs_mut()[output as usize]
            }
        };

        let mut found_user = false;
        for candidate_user in output.users.iter_mut() {
            if *candidate_user == user {
                *candidate_user = ValueUser::Input {
                    consumer: proxy,
                    input: 0,
                };

                found_user = true;
            }
        }

        assert!(
            found_user,
            "could not find a matching user for origin output"
        );

        let input = match user {
            ValueUser::Result(res) => &mut self.regions[region].value_results[res as usize],
            ValueUser::Input { consumer, input } => {
                let consumer = &mut self.nodes[consumer];

                assert_eq!(
                    consumer.region(),
                    region,
                    "origin must be in the specified region"
                );

                &mut consumer.value_inputs_mut()[input as usize]
            }
        };

        assert_eq!(input.ty, ty, "the user type must match the specified type");

        input.origin = ValueOrigin::Output {
            producer: proxy,
            output: 0,
        };

        self.regions[region].nodes.insert(proxy);

        proxy
    }

    pub fn proxy_origin_users(
        &mut self,
        region: Region,
        ty: Type,
        origin: ValueOrigin,
        proxy_kind: ProxyKind,
    ) -> Node {
        let output = match origin {
            ValueOrigin::Argument(arg) => &mut self.regions[region].value_arguments[arg as usize],
            ValueOrigin::Output { producer, output } => {
                let producer = &mut self.nodes[producer];

                assert_eq!(
                    producer.region(),
                    region,
                    "origin must be in the specified region"
                );

                &mut producer.value_outputs_mut()[output as usize]
            }
        };

        let users = output.users.clone();

        let proxy = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                ValueProxy {
                    proxy_kind,
                    input: ValueInput { ty, origin },
                    output: ValueOutput {
                        ty,
                        users: thin_set![],
                    },
                }
                .into(),
            ),
            region: Some(region),
        });

        for user in users.iter().copied() {
            let input = match user {
                ValueUser::Result(res) => &mut self.regions[region].value_results[res as usize],
                ValueUser::Input { consumer, input } => {
                    let consumer = &mut self.nodes[consumer];

                    assert_eq!(
                        consumer.region(),
                        region,
                        "origin must be in the specified region"
                    );

                    &mut consumer.value_inputs_mut()[input as usize]
                }
            };

            assert_eq!(input.ty, ty, "the user type must match the specified type");

            input.origin = ValueOrigin::Output {
                producer: proxy,
                output: 0,
            };
        }

        self.nodes[proxy].value_outputs_mut()[0].users = users;
        self.regions[region].nodes.insert(proxy);

        proxy
    }

    pub fn dissolve_value_proxy(&mut self, proxy_node: Node) {
        let region = self.nodes[proxy_node].region();
        let data = self.nodes[proxy_node].expect_value_proxy();
        let proxied_origin = data.input().origin;
        let user_count = data.output().users.len();

        for i in (0..user_count).rev() {
            let user = self.nodes[proxy_node].expect_value_proxy().output().users[i];

            self.reconnect_value_user(region, user, proxied_origin);
        }

        self.remove_node(proxy_node);
    }

    /// Adds a dependency on the given `dependency` node to the `function_node`.
    ///
    /// If the `function_node` was already dependent on the `dependency`, then this operation does
    /// nothing and returns the region argument index for the `dependency`. If the function was not
    /// already dependent on the `dependency`, a dependency is added at the end of the dependency
    /// list, and all call arguments are shifted to the right; the RVSDG is updated so that all
    /// users of the body region's call arguments remain valid.
    pub fn add_function_dependency(&mut self, function_node: Node, dependency: Node) -> u32 {
        let ty = self[dependency].value_outputs()[0].ty;
        let fn_data = self.nodes[function_node].expect_function_mut();

        for (i, dep_input) in fn_data.dependencies().iter().enumerate() {
            if let ValueOrigin::Output {
                producer,
                output: 0,
            } = dep_input.origin
            {
                if producer == dependency {
                    // The function already has a dependency on the node, so return the argument
                    // position for that dependency.
                    return i as u32;
                }
            }
        }

        let index = fn_data.dependencies().len();

        fn_data.dependencies.push(ValueInput {
            ty,
            origin: ValueOrigin::Output {
                producer: dependency,
                output: 0,
            },
        });

        let body_region = fn_data.body_region();

        self.insert_region_argument(body_region, ty, index);

        // Add the function as a user of the dependency's output
        self.nodes[dependency].value_outputs_mut()[0]
            .users
            .insert(ValueUser::Input {
                consumer: function_node,
                input: index as u32,
            });

        index as u32
    }

    /// For the given `function_node`, removes any the dependencies (and their corresponding region
    /// arguments) that don't have any users.
    pub fn remove_unused_dependencies(&mut self, function_node: Node) {
        let node_data = self.nodes[function_node].expect_function_mut();
        let body_region = node_data.body_region();
        let mut correction_start = node_data.dependencies.len();

        // Iterate over the dependencies in reverse order, so that when we remove dependencies
        // during iteration, we don't skip nodes
        for i in (0..node_data.dependencies.len()).rev() {
            if self.regions[body_region].value_arguments()[i]
                .users
                .is_empty()
            {
                let input = self.nodes[function_node]
                    .expect_function_mut()
                    .dependencies
                    .remove(i);

                self.regions[body_region].value_arguments.remove(i);

                let ValueOrigin::Output {
                    producer,
                    output: 0,
                } = input.origin
                else {
                    panic!("dependency origin should be the first output of a node");
                };

                self.nodes[producer].value_outputs_mut()[0]
                    .users
                    .remove(&ValueUser::Input {
                        consumer: function_node,
                        input: i as u32,
                    });

                // We won't have to correct all region arguments, only those that came after the
                // left-most dependency removed. Since we're iterating in reverse, simply setting
                // `correction_start` to the current index should ensure it ends up being the
                // correct index at the end of this loop.
                correction_start = i;
            }
        }

        // Correct the region argument back-edge connections.
        self.correct_region_argument_connections(body_region, correction_start);

        let new_dep_count = self.nodes[function_node]
            .expect_function()
            .dependencies
            .len();

        // Correct the edges from dependency nodes to the remaining dependency inputs.
        for i in 0..new_dep_count {
            let ValueOrigin::Output {
                producer: dep_node,
                output: 0,
            } = self.nodes[function_node].expect_function().dependencies()[i].origin
            else {
                panic!("dependency origin should be the first output of a node");
            };

            for user in self.nodes[dep_node].value_outputs_mut()[0].users.iter_mut() {
                if let ValueUser::Input { consumer, input } = user
                    && *consumer == function_node
                {
                    *input = i as u32;
                }
            }
        }
    }

    pub fn value_origin_ty(&self, region: Region, origin: ValueOrigin) -> Type {
        match origin {
            ValueOrigin::Argument(arg) => self.regions[region].value_arguments[arg as usize].ty,
            ValueOrigin::Output { producer, output } => {
                self.nodes[producer].value_outputs()[output as usize].ty
            }
        }
    }

    pub fn get_input_index(&self, node: Node, origin: ValueOrigin) -> Option<u32> {
        for (i, input) in self.nodes[node].value_inputs().iter().enumerate() {
            if input.origin == origin {
                return Some(i as u32);
            }
        }

        None
    }

    fn remove_region_with_nodes(&mut self, region: Region) {
        let node_count = self.regions[region].nodes().len();

        for i in (0..node_count).rev() {
            let node = self.regions[region].nodes()[i];

            // Note that we don't need to worry about removing/correcting any connections: nodes
            // can only connect to nodes inside the region or region arguments and results, all of
            // which are getting removed as well.

            match self.nodes[node].kind() {
                NodeKind::Switch(switch_node) => {
                    let branch_count = switch_node.branches().len();

                    for b in 0..branch_count {
                        let branch = self.nodes[node].expect_switch().branches()[b];

                        self.remove_region_with_nodes(branch);
                    }
                }
                NodeKind::Loop(loop_node) => {
                    let loop_region = loop_node.loop_region();

                    self.remove_region_with_nodes(loop_region);
                }
                _ => {}
            }

            self.nodes.remove(node);
        }

        self.regions.remove(region);
    }

    /// Helper function that inserts an argument of the given `ty` into the given `region` at the
    /// given `index`.
    fn insert_region_argument(&mut self, region: Region, ty: Type, at: usize) {
        let region_data = &mut self.regions[region];

        // Insert the new argument at the given position, shifting all arguments after it to the
        // right.
        region_data.value_arguments.insert(
            at,
            ValueOutput {
                ty,
                users: Default::default(),
            },
        );

        // All back-edges connecting to arguments to the right of the insertion index will now be
        // incorrect, so we visit all their consumers and adjust their value origins to reflect the
        // new argument position.
        let correction_start = at + 1;

        self.correct_region_argument_connections(region, correction_start);
    }

    /// A helper function that validates that the origin of the given `value_input` exists, belongs
    /// to the given `region`, and matches the `value_input`'s expected typed.
    fn validate_node_value_input(&mut self, region: Region, value_input: &ValueInput) {
        match &value_input.origin {
            ValueOrigin::Argument(i) => {
                let region = &self[region];

                if let Some(a) = region.value_arguments().get(*i as usize) {
                    if value_input.ty != a.ty {
                        panic!(
                            "cannot connect a node input of type `{:?}` to a region argument of type `{:?}",
                            value_input.ty, a.ty
                        );
                    }
                } else {
                    panic!(
                        "tried to connect to region argument `{}`, but region only has {} arguments",
                        i,
                        region.value_arguments().len()
                    );
                }
            }
            ValueOrigin::Output { producer, output } => {
                let producer = &self[*producer];

                if producer.region != Some(region) {
                    panic!("cannot connect a node input to a node output in a different region");
                }

                if let Some(output) = producer.value_outputs().get(*output as usize) {
                    if !self.ty.is_compatible(value_input.ty, output.ty) {
                        panic!(
                            "cannot connect a node input of type `{:?}` to an output of type `{:?}",
                            value_input.ty, output.ty
                        );
                    }
                } else {
                    panic!(
                        "tried to connect to node output `{}`, but the target only has {} outputs",
                        output,
                        producer.value_outputs().len()
                    );
                }
            }
        }
    }

    /// Helper that adds all of a node's value inputs to the corresponding value outputs as users.
    fn connect_node_value_inputs(&mut self, node: Node) {
        let region = self.nodes[node]
            .region
            .expect("node region should be set before connecting inputs");
        let input_count = self.nodes[node].value_inputs().len();

        for input_index in 0..input_count {
            self.connect_node_value_input(region, node, input_index);
        }
    }

    fn connect_node_value_input(&mut self, region: Region, node: Node, input_index: usize) {
        let user = ValueUser::Input {
            consumer: node,
            input: input_index as u32,
        };

        match self.nodes[node].value_inputs()[input_index].origin {
            ValueOrigin::Argument(arg_index) => {
                self.regions[region].value_arguments[arg_index as usize]
                    .users
                    .insert(user);
            }
            ValueOrigin::Output { producer, output } => {
                assert_ne!(
                    producer, node,
                    "cannot connect a node input to one of its own outputs"
                );

                self.nodes[producer].value_outputs_mut()[output as usize]
                    .users
                    .insert(user);
            }
        }
    }

    /// Helper function that for each value input with an index of `start` or greater, updates the
    /// edge from that input's origin to use the current index of that input.
    ///
    /// Needs to know the `adjustment` that happened to cause the potentially invalid edges. For
    /// example: if an input was removed, then all inputs starting at `start` will have had their
    /// indices reduced by `1`, so the value passed as the `adjustment` argument should be `-1`.
    fn correct_value_input_connections(&mut self, node: Node, start: usize, adjustment: i32) {
        let node_data = &self.nodes[node];
        let region = node_data.region();
        let end = node_data.value_inputs().len();

        for i in start..end {
            let origin = self.nodes[node].value_inputs()[i].origin;

            // Resolve what the `input` value was before the adjustment by undoing (subtracting) the
            // adjustment.
            let old_input = (i as i32 - adjustment) as u32;

            match origin {
                ValueOrigin::Argument(arg) => {
                    for user in self.regions[region].value_arguments[arg as usize]
                        .users
                        .iter_mut()
                    {
                        if let ValueUser::Input { consumer, input } = user
                            && *consumer == node
                            && *input == old_input
                        {
                            *input = i as u32;
                        }
                    }
                }
                ValueOrigin::Output { producer, output } => {
                    for user in self.nodes[producer].value_outputs_mut()[output as usize]
                        .users
                        .iter_mut()
                    {
                        if let ValueUser::Input { consumer, input } = user
                            && *consumer == node
                            && *input == old_input
                        {
                            *input = i as u32;
                        }
                    }
                }
            }
        }
    }

    /// Helper function that for each value output with an index of `start` or greater, updates all
    /// back-edges from that outputs's users to use the current index of that output.
    fn correct_value_output_connections(&mut self, node: Node, start: usize) {
        let end = self.nodes[node].value_outputs().len();
        let region = self.nodes[node].region();

        for i in start..end {
            let user_count = self.nodes[node].value_outputs()[i].users.len();
            let new_origin = ValueOrigin::Output {
                producer: node,
                output: i as u32,
            };

            for j in 0..user_count {
                match self.nodes[node].value_outputs_mut()[i].users[j] {
                    ValueUser::Result(result_index) => {
                        self.regions[region].value_results[result_index as usize].origin =
                            new_origin;
                    }
                    ValueUser::Input { consumer, input } => {
                        self.nodes[consumer].value_inputs_mut()[input as usize].origin = new_origin;
                    }
                }
            }
        }
    }

    /// Helper function that for each argument with an index of `start` or greater, updates all
    /// back-edges from that argument's users to use the current index of that argument.
    fn correct_region_argument_connections(&mut self, region: Region, start: usize) {
        let end = self.regions[region].value_arguments.len();

        for i in start..end {
            let user_count = self.regions[region].value_arguments[i].users.len();
            let new_origin = ValueOrigin::Argument(i as u32);

            for j in 0..user_count {
                match self.regions[region].value_arguments[i].users[j] {
                    ValueUser::Result(result_index) => {
                        self.regions[region].value_results[result_index as usize].origin =
                            new_origin;
                    }
                    ValueUser::Input { consumer, input } => {
                        self.nodes[consumer].value_inputs_mut()[input as usize].origin = new_origin;
                    }
                }
            }
        }
    }

    /// Helper function that for each result with an index of `start` or greater, updates the edge
    /// from that result's origin to use the current index of that result.
    ///
    /// Needs to know the `adjustment` that happened to cause the potentially invalid edges. For
    /// example: if a result was removed, then all results starting at `start` will have had their
    /// indices reduced by `1`, so the value passed as the `adjustment` argument should be `-1`.
    fn correct_region_result_connections(&mut self, region: Region, start: usize, adjustment: i32) {
        let end = self.regions[region].value_results.len();

        for i in start..end {
            let origin = self.regions[region].value_results[i].origin;

            // Resolve what the result's index was before the adjustment by undoing (subtracting)
            // the adjustment.
            let old_index = (i as i32 - adjustment) as u32;

            match origin {
                ValueOrigin::Argument(arg) => {
                    for user in self.regions[region].value_arguments[arg as usize]
                        .users
                        .iter_mut()
                    {
                        if let ValueUser::Result(res) = user
                            && *res == old_index
                        {
                            *res = i as u32;
                        }
                    }
                }
                ValueOrigin::Output { producer, output } => {
                    for user in self.nodes[producer].value_outputs_mut()[output as usize]
                        .users
                        .iter_mut()
                    {
                        if let ValueUser::Result(res) = user
                            && *res == old_index
                        {
                            *res = i as u32;
                        }
                    }
                }
            }
        }
    }

    /// A helper function that links the given `node` into the state chain after the `state_origin`.
    fn link_state(&mut self, region: Region, node: Node, state_origin: StateOrigin) {
        fn adjust_user_origin(
            rvsdg: &mut Rvsdg,
            region: Region,
            node: Node,
            state_user: StateUser,
        ) {
            match state_user {
                StateUser::Result => rvsdg.regions[region].state_result = StateOrigin::Node(node),
                StateUser::Node(n) => {
                    rvsdg.nodes[n].state_mut().unwrap().origin = StateOrigin::Node(node)
                }
            }
        }

        let state_user = match state_origin {
            StateOrigin::Argument => {
                let state_user = self[region].state_argument;

                self.regions[region].state_argument = StateUser::Node(node);
                adjust_user_origin(self, region, node, state_user);

                state_user
            }
            StateOrigin::Node(n) => {
                let state_user = self[n].state().unwrap().user;

                self.nodes[n].state_mut().unwrap().user = StateUser::Node(node);
                adjust_user_origin(self, region, node, state_user);

                state_user
            }
        };

        self.nodes[node].state_mut().unwrap().user = state_user;
    }

    /// Helper function that unlinks the given `node` from its region's state chain.
    fn unlink_state(&mut self, node: Node) {
        let Some(state) = self.nodes[node].state().copied() else {
            // The node has no state information, there's nothing to unlink.
            return;
        };

        let region = self.nodes[node].region();

        match state.origin {
            StateOrigin::Argument => {
                self.regions[region].state_argument = state.user;
            }
            StateOrigin::Node(producer) => {
                self.nodes[producer]
                    .state_mut()
                    .expect("a node that is part of the state chain should have state information")
                    .user = state.user;
            }
        }

        match state.user {
            StateUser::Result => {
                self.regions[region].state_result = state.origin;
            }
            StateUser::Node(consumer) => {
                self.nodes[consumer]
                    .state_mut()
                    .expect("a node that is part of the state chain should have state information")
                    .origin = state.origin;
            }
        }
    }

    /// Helper function for adding a user to an origin.
    ///
    /// Does not modify the user side; this needs to be updated separately to ensure a valid two-way
    /// connection.
    fn add_user(&mut self, region: Region, origin: ValueOrigin, user: ValueUser) {
        match origin {
            ValueOrigin::Argument(i) => self.regions[region].value_arguments[i as usize]
                .users
                .insert(user),
            ValueOrigin::Output { producer, output } => self.nodes[producer].value_outputs_mut()
                [output as usize]
                .users
                .insert(user),
        }
    }

    /// Helper function for removing a user from an origin.
    ///
    /// Does not modify the user side; this needs to be updated separately to ensure a valid two-way
    /// connection.
    fn remove_user(&mut self, region: Region, origin: ValueOrigin, user: ValueUser) {
        match origin {
            ValueOrigin::Argument(i) => self.regions[region].value_arguments[i as usize]
                .users
                .remove(&user),
            ValueOrigin::Output { producer, output } => self.nodes[producer].value_outputs_mut()
                [output as usize]
                .users
                .remove(&user),
        }
    }

    fn project_index(&self, (i, input): (usize, &ValueInput), base_ty: Type) -> Type {
        match &*self.ty.kind(base_ty) {
            TypeKind::Struct(s) => {
                if let ValueOrigin::Output {
                    producer,
                    output: 0,
                } = input.origin
                    && let NodeKind::Simple(SimpleNode::ConstU32(n)) = self[producer].kind()
                {
                    let index = n.value as usize;

                    s.fields[index].ty
                } else {
                    panic!(
                        "index `{}` tried to project into a struct field with a non-constant index",
                        i
                    );
                }
            }
            TypeKind::Vector(v) => v.scalar.ty(),
            TypeKind::Matrix(m) => m.column_ty(),
            TypeKind::Array { element_ty, .. } | TypeKind::Slice { element_ty, .. } => *element_ty,
            _ => {
                panic!("index `{}` tried to index into a non-aggregate type", i);
            }
        }
    }
}

impl Index<Region> for Rvsdg {
    type Output = RegionData;

    fn index(&self, region: Region) -> &Self::Output {
        &self.regions[region]
    }
}

impl Index<Node> for Rvsdg {
    type Output = NodeData;

    fn index(&self, node: Node) -> &Self::Output {
        &self.nodes[node]
    }
}

impl PartialEq for Rvsdg {
    fn eq(&self, other: &Self) -> bool {
        if self.function_node != other.function_node {
            return false;
        }

        if self.regions.len() != other.regions.len() {
            return false;
        }

        for (region, data) in other.regions.iter() {
            if self.regions.get(region) != Some(data) {
                return false;
            }
        }

        if self.nodes.len() != other.nodes.len() {
            return false;
        }

        for (node, data) in other.nodes.iter() {
            if self.nodes.get(node) != Some(data) {
                return false;
            }
        }

        true
    }
}

#[derive(Error, Debug)]
#[error("cannot remove a node that still has users (output {output_with_users} has users)")]
pub struct RemoveNodeError {
    output_with_users: u32,
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::ty::TY_DUMMY;
    use crate::{FnArg, FnSig, Symbol, thin_set};

    #[test]
    fn test_rvsdg_single_simple_node() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref(""),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Default::default(),
                ty: TY_DUMMY,
                args: vec![
                    FnArg {
                        ty: TY_U32,
                        shader_io_binding: None,
                    },
                    FnArg {
                        ty: TY_U32,
                        shader_io_binding: None,
                    },
                ],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let node = rvsdg.add_op_binary(
            region,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 0),
            ValueInput::argument(TY_U32, 1),
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: node,
                output: 0,
            },
        );

        // Check the region arguments
        assert_eq!(
            rvsdg[region].value_arguments()[0],
            ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Input {
                    consumer: node,
                    input: 0,
                }],
            }
        );
        assert_eq!(
            rvsdg[region].value_arguments()[1],
            ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Input {
                    consumer: node,
                    input: 1,
                }],
            }
        );

        // Check the node inputs and outputs
        assert_eq!(
            rvsdg[node].value_inputs()[0],
            ValueInput::argument(TY_U32, 0),
        );
        assert_eq!(
            rvsdg[node].value_inputs()[1],
            ValueInput::argument(TY_U32, 1),
        );
        assert_eq!(
            rvsdg[node].value_outputs()[0],
            ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Result(0)],
            }
        );

        // Check the region results
        assert_eq!(
            rvsdg[region].value_results()[0],
            ValueInput {
                ty: TY_U32,
                origin: ValueOrigin::Output {
                    producer: node,
                    output: 0,
                },
            }
        );
    }

    #[test]
    fn test_rvsdg_dependent_simple_nodes() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref(""),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Default::default(),
                ty: TY_DUMMY,
                args: vec![
                    FnArg {
                        ty: TY_U32,
                        shader_io_binding: None,
                    },
                    FnArg {
                        ty: TY_U32,
                        shader_io_binding: None,
                    },
                ],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let node_0 = rvsdg.add_const_u32(region, 5);

        let node_1 = rvsdg.add_op_binary(
            region,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 0),
            ValueInput::output(TY_U32, node_0, 0),
        );

        let node_2 = rvsdg.add_op_binary(
            region,
            BinaryOperator::Add,
            ValueInput::output(TY_U32, node_1, 0),
            ValueInput::argument(TY_U32, 1),
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: node_2,
                output: 0,
            },
        );

        // Check the region arguments
        assert_eq!(
            rvsdg[region].value_arguments()[0],
            ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Input {
                    consumer: node_1,
                    input: 0,
                }],
            }
        );
        assert_eq!(
            rvsdg[region].value_arguments()[1],
            ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Input {
                    consumer: node_2,
                    input: 1,
                }],
            }
        );

        // Check node_0 inputs and outputs
        assert!(rvsdg[node_0].value_inputs().is_empty());
        assert_eq!(
            rvsdg[node_0].value_outputs()[0],
            ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Input {
                    consumer: node_1,
                    input: 1,
                }],
            }
        );

        // Check node_1 inputs and outputs
        assert_eq!(
            rvsdg[node_1].value_inputs()[0],
            ValueInput::argument(TY_U32, 0),
        );
        assert_eq!(
            rvsdg[node_1].value_inputs()[1],
            ValueInput::output(TY_U32, node_0, 0)
        );
        assert_eq!(
            rvsdg[node_1].value_outputs()[0],
            ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Input {
                    consumer: node_2,
                    input: 0,
                }],
            }
        );

        // Check node_1 inputs and outputs
        assert_eq!(
            rvsdg[node_2].value_inputs()[0],
            ValueInput::output(TY_U32, node_1, 0),
        );
        assert_eq!(
            rvsdg[node_2].value_inputs()[1],
            ValueInput::argument(TY_U32, 1),
        );
        assert_eq!(
            rvsdg[node_2].value_outputs()[0],
            ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Result(0)],
            }
        );

        // Check the region results
        assert_eq!(
            rvsdg[region].value_results()[0],
            ValueInput::output(TY_U32, node_2, 0)
        );
    }

    #[test]
    fn test_rvsdg_switch_node() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref(""),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Default::default(),
                ty: TY_DUMMY,
                args: vec![
                    FnArg {
                        ty: TY_PREDICATE,
                        shader_io_binding: None,
                    },
                    FnArg {
                        ty: TY_U32,
                        shader_io_binding: None,
                    },
                ],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::argument(TY_U32, 1),
            ],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);

        let branch_0_node_0 = rvsdg.add_const_u32(branch_0, 1);
        let branch_0_node_1 = rvsdg.add_op_binary(
            branch_0,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 0),
            ValueInput::output(TY_U32, branch_0_node_0, 0),
        );

        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: branch_0_node_1,
                output: 0,
            },
        );

        let branch_1 = rvsdg.add_switch_branch(switch_node);

        let branch_1_node_0 = rvsdg.add_const_u32(branch_1, 0);

        rvsdg.reconnect_region_result(
            branch_1,
            0,
            ValueOrigin::Output {
                producer: branch_1_node_0,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: switch_node,
                output: 0,
            },
        );

        // Check the base region arguments
        assert_eq!(
            rvsdg[region].value_arguments()[0],
            ValueOutput {
                ty: TY_PREDICATE,
                users: thin_set![ValueUser::Input {
                    consumer: switch_node,
                    input: 0,
                }],
            }
        );
        assert_eq!(
            rvsdg[region].value_arguments()[1],
            ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Input {
                    consumer: switch_node,
                    input: 1,
                }],
            }
        );

        // Check the switch node inputs and outputs
        assert_eq!(
            rvsdg[switch_node].value_inputs()[0],
            ValueInput::argument(TY_PREDICATE, 0),
        );
        assert_eq!(
            rvsdg[switch_node].value_inputs()[1],
            ValueInput::argument(TY_U32, 1),
        );
        assert_eq!(
            rvsdg[switch_node].value_outputs()[0],
            ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Result(0)],
            }
        );

        // Check region branch_0 arguments
        assert_eq!(rvsdg[branch_0].value_arguments().len(), 1);
        assert_eq!(
            rvsdg[branch_0].value_arguments()[0],
            ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Input {
                    consumer: branch_0_node_1,
                    input: 0,
                }],
            }
        );

        // Check branch_0_node_0 inputs and outputs
        assert_eq!(
            rvsdg[branch_0_node_0].value_outputs()[0],
            ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Input {
                    consumer: branch_0_node_1,
                    input: 1,
                }],
            }
        );

        // Check branch_0_node_1 inputs and outputs
        assert_eq!(
            rvsdg[branch_0_node_1].value_inputs()[0],
            ValueInput::argument(TY_U32, 0),
        );
        assert_eq!(
            rvsdg[branch_0_node_1].value_inputs()[1],
            ValueInput::output(TY_U32, branch_0_node_0, 0),
        );
        assert_eq!(
            rvsdg[branch_0_node_1].value_outputs()[0],
            ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Result(0)],
            }
        );

        // Check region branch_0 results
        assert_eq!(rvsdg[branch_0].value_results().len(), 1);
        assert_eq!(
            rvsdg[branch_0].value_results()[0],
            ValueInput::output(TY_U32, branch_0_node_1, 0),
        );

        // Check region branch_1 arguments
        assert_eq!(rvsdg[branch_1].value_arguments().len(), 1);
        assert_eq!(
            rvsdg[branch_1].value_arguments()[0],
            ValueOutput {
                ty: TY_U32,
                users: thin_set![],
            }
        );

        // Check branch_1_node_0 inputs and outputs
        assert_eq!(
            rvsdg[branch_1_node_0].value_outputs()[0],
            ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Result(0)],
            }
        );

        // Check branch_1 region results
        assert_eq!(rvsdg[branch_1].value_results().len(), 1);
        assert_eq!(
            rvsdg[branch_1].value_results()[0],
            ValueInput::output(TY_U32, branch_1_node_0, 0),
        );

        // Check base region results
        assert_eq!(rvsdg[region].value_results().len(), 1);
        assert_eq!(
            rvsdg[region].value_results()[0],
            ValueInput::output(TY_U32, switch_node, 0),
        );
    }

    #[test]
    fn test_rvsdg_loop_node() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref(""),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Default::default(),
                ty: TY_DUMMY,
                args: vec![FnArg {
                    ty: TY_U32,
                    shader_io_binding: None,
                }],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let (loop_node, loop_region) =
            rvsdg.add_loop(region, vec![ValueInput::argument(TY_U32, 0)], None);

        let loop_node_0 = rvsdg.add_const_u32(loop_region, 1);
        let loop_node_1 = rvsdg.add_op_binary(
            loop_region,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 0),
            ValueInput::output(TY_U32, loop_node_0, 0),
        );
        let loop_node_2 = rvsdg.add_const_u32(loop_region, 10);
        let loop_node_3 = rvsdg.add_op_binary(
            loop_region,
            BinaryOperator::Lt,
            ValueInput::output(TY_U32, loop_node_1, 0),
            ValueInput::output(TY_U32, loop_node_2, 0),
        );

        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: loop_node_3,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            loop_region,
            1,
            ValueOrigin::Output {
                producer: loop_node_1,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: loop_node,
                output: 0,
            },
        );

        // Check the base region arguments
        assert_eq!(
            rvsdg[region].value_arguments()[0],
            ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Input {
                    consumer: loop_node,
                    input: 0,
                }],
            }
        );

        // Check loop node inputs and outputs
        assert_eq!(
            rvsdg[loop_node].value_inputs()[0],
            ValueInput::argument(TY_U32, 0),
        );
        assert_eq!(
            rvsdg[loop_node].value_outputs()[0],
            ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Result(0)],
            }
        );

        // Check loop region arguments
        assert_eq!(
            rvsdg[loop_region].value_arguments()[0],
            ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Input {
                    consumer: loop_node_1,
                    input: 0,
                }],
            }
        );

        // Check loop_node_0 inputs and outputs
        assert_eq!(
            rvsdg[loop_node_0].value_outputs()[0],
            ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Input {
                    consumer: loop_node_1,
                    input: 1,
                }],
            }
        );

        // Check loop_node_1 inputs and outputs
        assert_eq!(
            rvsdg[loop_node_1].value_inputs()[0],
            ValueInput::argument(TY_U32, 0),
        );
        assert_eq!(
            rvsdg[loop_node_1].value_inputs()[1],
            ValueInput::output(TY_U32, loop_node_0, 0)
        );
        assert_eq!(
            rvsdg[loop_node_1].value_outputs()[0],
            ValueOutput {
                ty: TY_U32,
                users: thin_set![
                    ValueUser::Input {
                        consumer: loop_node_3,
                        input: 0,
                    },
                    ValueUser::Result(1)
                ],
            }
        );

        // Check loop_node_2 inputs and outputs
        assert_eq!(
            rvsdg[loop_node_2].value_outputs()[0],
            ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Input {
                    consumer: loop_node_3,
                    input: 1,
                }],
            }
        );

        // Check loop_node_3 inputs and outputs
        assert_eq!(
            rvsdg[loop_node_3].value_inputs()[0],
            ValueInput::output(TY_U32, loop_node_1, 0),
        );
        assert_eq!(
            rvsdg[loop_node_3].value_inputs()[1],
            ValueInput::output(TY_U32, loop_node_2, 0)
        );
        assert_eq!(
            rvsdg[loop_node_3].value_outputs()[0],
            ValueOutput {
                ty: TY_BOOL,
                users: thin_set![ValueUser::Result(0)],
            }
        );

        // Check loop region results
        assert_eq!(rvsdg[loop_region].value_results().len(), 2);
        assert_eq!(
            rvsdg[loop_region].value_results()[0],
            ValueInput::output(TY_BOOL, loop_node_3, 0),
        );
        assert_eq!(
            rvsdg[loop_region].value_results()[1],
            ValueInput::output(TY_U32, loop_node_1, 0),
        );

        // Check base region results
        assert_eq!(rvsdg[region].value_results().len(), 1);
        assert_eq!(
            rvsdg[region].value_results()[0],
            ValueInput::output(TY_U32, loop_node, 0),
        );
    }

    #[test]
    fn test_rvsdg_stateful() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref(""),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Default::default(),
                ty: TY_DUMMY,
                args: vec![
                    FnArg {
                        ty: module.ty.register(TypeKind::Ptr(TY_U32)),
                        shader_io_binding: None,
                    },
                    FnArg {
                        ty: TY_U32,
                        shader_io_binding: None,
                    },
                ],
                ret_ty: None,
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let node_0 = rvsdg.add_const_u32(region, 1);
        let node_1 = rvsdg.add_op_load(
            region,
            ValueInput::argument(TY_PTR_U32, 0),
            StateOrigin::Argument,
        );
        let node_2 = rvsdg.add_op_binary(
            region,
            BinaryOperator::Add,
            ValueInput::output(TY_U32, node_1, 0),
            ValueInput::output(TY_U32, node_0, 0),
        );
        let node_3 = rvsdg.add_op_store(
            region,
            ValueInput::argument(TY_PTR_U32, 0),
            ValueInput::output(TY_U32, node_2, 0),
            StateOrigin::Node(node_1),
        );

        // Check the region arguments
        assert_eq!(
            rvsdg[region].value_arguments()[0],
            ValueOutput {
                ty: TY_PTR_U32,
                users: thin_set![
                    ValueUser::Input {
                        consumer: node_1,
                        input: 0,
                    },
                    ValueUser::Input {
                        consumer: node_3,
                        input: 0,
                    }
                ],
            }
        );
        assert_eq!(rvsdg[region].state_argument(), &StateUser::Node(node_1));

        // Check node_0 inputs and outputs
        assert_eq!(
            rvsdg[node_0].value_outputs()[0],
            ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Input {
                    consumer: node_2,
                    input: 1,
                }],
            }
        );

        // Check node_1 inputs and outputs
        assert_eq!(
            rvsdg[node_1].value_inputs()[0],
            ValueInput::argument(TY_PTR_U32, 0),
        );
        assert_eq!(
            rvsdg[node_1].state(),
            Some(&State {
                origin: StateOrigin::Argument,
                user: StateUser::Node(node_3),
            })
        );

        // Check node_2 inputs and outputs
        assert_eq!(
            rvsdg[node_2].value_inputs()[0],
            ValueInput::output(TY_U32, node_1, 0),
        );
        assert_eq!(
            rvsdg[node_2].value_inputs()[1],
            ValueInput::output(TY_U32, node_0, 0),
        );
        assert_eq!(
            rvsdg[node_2].value_outputs()[0],
            ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Input {
                    consumer: node_3,
                    input: 1
                }],
            }
        );

        // Check node_3 inputs and outputs
        assert_eq!(
            rvsdg[node_3].value_inputs()[0],
            ValueInput::argument(TY_PTR_U32, 0),
        );
        assert_eq!(
            rvsdg[node_3].value_inputs()[1],
            ValueInput::output(TY_U32, node_2, 0),
        );
        assert_eq!(
            rvsdg[node_3].state(),
            Some(&State {
                origin: StateOrigin::Node(node_1),
                user: StateUser::Result,
            })
        );

        // Check region results
        assert!(rvsdg[region].value_results().is_empty());
        assert_eq!(rvsdg[region].state_result(), &StateOrigin::Node(node_3),);
    }

    #[test]
    fn test_remove_unused_dependencies() {
        let mut module = Module::new(Symbol::from_ref(""));

        let dependency_0 = Function {
            name: Symbol::from_ref("dependency_0"),
            module: Symbol::from_ref(""),
        };
        let dependency_0_ty = module.ty.register(TypeKind::Function(dependency_0));

        module.fn_sigs.register(
            dependency_0,
            FnSig {
                name: Default::default(),
                ty: dependency_0_ty,
                args: vec![],
                ret_ty: None,
            },
        );

        let dependency_1 = Function {
            name: Symbol::from_ref("dependency_1"),
            module: Symbol::from_ref(""),
        };
        let dependency_1_ty = module.ty.register(TypeKind::Function(dependency_1));

        module.fn_sigs.register(
            dependency_1,
            FnSig {
                name: Default::default(),
                ty: dependency_1_ty,
                args: vec![],
                ret_ty: None,
            },
        );

        let dependency_2 = Function {
            name: Symbol::from_ref("dependency_2"),
            module: Symbol::from_ref(""),
        };
        let dependency_2_ty = module.ty.register(TypeKind::Function(dependency_2));

        module.fn_sigs.register(
            dependency_2,
            FnSig {
                name: Default::default(),
                ty: dependency_2_ty,
                args: vec![],
                ret_ty: None,
            },
        );

        let dependent = Function {
            name: Symbol::from_ref("dependent"),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            dependent,
            FnSig {
                name: Default::default(),
                ty: TY_DUMMY,
                args: vec![FnArg {
                    ty: TY_U32,
                    shader_io_binding: None,
                }],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (dep_0_node, _) = rvsdg.register_function(&module, dependency_0, iter::empty());
        let (dep_1_node, _) = rvsdg.register_function(&module, dependency_1, iter::empty());
        let (dep_2_node, _) = rvsdg.register_function(&module, dependency_2, iter::empty());

        let (dependent_node, dependent_region) =
            rvsdg.register_function(&module, dependent, [dep_0_node, dep_1_node, dep_2_node]);

        rvsdg.add_op_call(
            &module,
            dependent_region,
            ValueInput::argument(dependency_1_ty, 1),
            iter::empty(),
            StateOrigin::Argument,
        );
        rvsdg.reconnect_region_result(dependent_region, 0, ValueOrigin::Argument(3));

        rvsdg.remove_unused_dependencies(dependent_node);

        // The dependent function should now only depend on dependency `1`.
        assert_eq!(
            rvsdg[dependent_node].expect_function().dependencies(),
            &[ValueInput::output(dependency_1_ty, dep_1_node, 0)]
        );

        // Dependency `0` should not have any users.
        assert!(rvsdg[dep_0_node].value_outputs()[0].users.is_empty());

        // Dependency `1` should still have the dependent node as its user.
        assert_eq!(rvsdg[dep_1_node].value_outputs()[0].users.len(), 1);
        assert_eq!(
            rvsdg[dep_1_node].value_outputs()[0].users[0],
            ValueUser::Input {
                consumer: dependent_node,
                input: 0
            }
        );

        // Dependency `2` should not have any users.
        assert!(rvsdg[dep_2_node].value_outputs()[0].users.is_empty());

        // The region result's origin should have been updated to point the shifted argument
        // position.
        assert_eq!(
            rvsdg[dependent_region].value_results()[0].origin,
            ValueOrigin::Argument(1)
        );
    }
}
