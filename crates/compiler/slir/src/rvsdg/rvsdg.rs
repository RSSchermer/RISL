use std::fs::File;
use std::io::Write;
use std::ops::Index;
use std::path::Path;
use std::slice;

use indexmap::IndexSet;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use slotmap::SlotMap;
use smallvec::SmallVec;
use thiserror::Error;

use crate::intrinsic::Intrinsic;
use crate::ty::{TY_BOOL, TY_F32, TY_I32, TY_PREDICATE, TY_U32, Type, TypeKind, TypeRegistry};
use crate::util::thin_set::ThinSet;
use crate::{
    BinaryOperator, Constant, Function, Module, StorageBinding, UnaryOperator, UniformBinding,
    WorkgroupBinding, intrinsic, thin_set, ty,
};

/// Common interface that all nodes implement to describe how they are connected to other elements
/// of the RVSDG.
pub trait Connectivity {
    /// List the [ValueInput]s the node consumes.
    fn value_inputs(&self) -> &[ValueInput];

    #[doc(hidden)]
    /// This is only used internally, we don't ever return `mut` references to nodes in the RVSDG;
    /// modifications to nodes in RVSDG are done via method calls on the [Rvsdg] data structure.
    fn value_inputs_mut(&mut self) -> &mut [ValueInput];

    /// List the [ValueOutput]s the node produces.
    fn value_outputs(&self) -> &[ValueOutput];

    #[doc(hidden)]
    /// This is only used internally, we don't ever return `mut` references to nodes in the RVSDG;
    /// modifications to nodes in RVSDG are done via method calls on the [Rvsdg] data structure.
    fn value_outputs_mut(&mut self) -> &mut [ValueOutput];

    /// Describes how the node connects to the state chain if the node is part of the state chain.
    ///
    /// A `None` value indicates that the node is not part of the state chain.
    fn state(&self) -> Option<&State>;

    #[doc(hidden)]
    /// This is only used internally, we don't ever return `mut` references to nodes in the RVSDG;
    /// modifications to nodes in RVSDG are done via method calls on the [Rvsdg] data structure.
    fn state_mut(&mut self) -> Option<&mut State>;
}

slotmap::new_key_type! {
    /// Identifies a node in a [Rvsdg].
    ///
    /// Can be used to resolve [NodeData] through the `Index<Node>` implementation on the [Rvsdg]
    /// with which it is associated:
    ///
    /// ```
    /// # fn f(node: Node) {
    /// let node_data = &rvsdg[node];
    /// # }
    /// ```
    pub struct Node;

    /// Identifies a region in a [Rvsdg].
    ///
    /// Can be used to resolve [RegionData] through the `Index<Region>` implementation on the
    /// [Rvsdg] with which it is associated:
    ///
    /// ```
    /// # fn f(region: Region) {
    /// let node_data = &rvsdg[region];
    /// # }
    /// ```
    pub struct Region;
}

/// [Rvsdg] node that models a function.
///
/// Function nodes may only appear in the [Rvsdg::global_region] region.
///
/// # Dependency Inputs
///
/// A function node may have any number of value-inputs. Since a function may only appear in the
/// global region, the origins of these inputs can only originate from the following node kinds:
///
/// - Uniform Binding nodes (see [UniformBindingNode] and [NodeKind::UniformBinding]).
/// - Storage Binding nodes (see [StorageBindingNode] and [NodeKind::StorageBinding]).
/// - Workgroup Binding nodes (see [WorkgroupBindingNode] and [NodeKind::WorkgroupBinding]).
/// - Constant nodes (see [ConstantNode] and [NodeKind::Constant]).
/// - Other function nodes.
///
/// The inputs to a function node represent its dependencies. These dependencies are made available
/// as argument values inside the function node's body region, see the `Body Region` section below
/// for details.
///
/// # Output
///
/// A function node has a single output value that represents the node's function itself as a
/// value. Other function nodes can use this [ValueOutput] as a dependency input. This makes this
/// function value available as an argument in that dependent function's body region. [OpCall]
/// nodes may then use this value as their first input to invoke the function.
///
/// # Body Region
///
/// A function node has an associated [Region] that represents the function body; this region
/// contains the nodes that model the function's instructions.
///
/// The body region receives a set of arguments. These region arguments represent first the
/// function node's dependencies, then the function's call arguments. A function with 2 dependencies
/// and 3 call arguments will have a body region with 5 arguments: region arguments `0..2` will
/// represent the dependencies, and region arguments `2..5` will represent the call arguments.
///
/// For functions that have a return value, the body region will have a single value result that
/// represents the function's return value; for functions that have no return value, the body region
/// will not have any value results.
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct FunctionNode {
    function: Function,
    dependencies: Vec<ValueInput>,
    output: ValueOutput,
    region: Region,
}

impl FunctionNode {
    /// The function this node represents.
    pub fn function(&self) -> Function {
        self.function
    }

    /// The function node's body region.
    ///
    /// See the `Body Region` section of the documentation for the [FunctionNode] struct for
    /// details.
    pub fn body_region(&self) -> Region {
        self.region
    }

    /// The function's dependency inputs.
    ///
    /// See the `Dependency Inputs` section of the documentation for the [FunctionNode] struct for
    /// details.
    pub fn dependencies(&self) -> &[ValueInput] {
        &self.dependencies
    }

    /// The output value of this function node.
    ///
    /// See the `Output` section of the documentation for the [FunctionNode] struct for details.
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

/// Models the consuming end of a value-flow edge in the RVSDG.
///
/// A combination of a [Type] and a [ValueOrigin]. The origin must resolve to a [ValueOutput] in the
/// same [Region]. The type of that [ValueOutput] must be compatible with the [Type] of the
/// [ValueInput], as defined by [TypeRegistry::can_coerce].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct ValueInput {
    /// The type of the value.
    pub ty: Type,

    /// The origin of the value.
    pub origin: ValueOrigin,
}

impl ValueInput {
    /// Shorthand function for creating a new value-input for which the [origin] is a "placeholder".
    ///
    /// See [ValueOrigin::placeholder] for details.
    pub fn placeholder(ty: Type) -> Self {
        ValueInput {
            ty,
            origin: ValueOrigin::placeholder(),
        }
    }

    /// Shorthand function for creating a new value-input for which the [origin] is a region
    /// argument.
    pub fn argument(ty: Type, arg: u32) -> Self {
        ValueInput {
            ty,
            origin: ValueOrigin::Argument(arg),
        }
    }

    /// Shorthand function for creating a new value-input for which the [origin] is a node output.
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

/// Enumerates the possible origins of a [ValueInput].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub enum ValueOrigin {
    /// The origin is a region argument in the same region as the [ValueInput].
    Argument(u32),

    /// The origin is a node output in the same region as the [ValueInput].
    Output {
        /// The node producing the output.
        producer: Node,

        /// The specific index of the output on the [producer] node.
        output: u32,
    },
}

impl ValueOrigin {
    /// Creates a new "placeholder" value-origin.
    ///
    /// A placeholder value-origin can be used for value-inputs that cannot immediately be assigned
    /// a valid origin. For example, when adding a new branch region to a [SwitchNode] (see
    /// [Rvsdg::add_switch_branch]), this branch region will initially be empty with placeholder
    /// origins for each of the region results. These placeholder origins must then be replaced with
    /// valid origins, for example, by connecting them to a region argument or by connecting them
    /// to node outputs as nodes are added to the region. Placeholder value-origins are only meant
    /// to be a temporary part of an RVSDG during construction or a transformation; there should be
    /// no placeholder value-inputs in the "final" RVSDG.
    pub fn placeholder() -> Self {
        ValueOrigin::Argument(u32::MAX)
    }

    /// Whether or not this origin is a "placeholder".
    ///
    /// See [placeholder] for details.
    pub fn is_placeholder(&self) -> bool {
        self == &ValueOrigin::Argument(u32::MAX)
    }
}

/// Models the producing end of a value-flow edge in the RVSDG.
///
/// A combination of a [Type] and a set of [ValueUser]s. Each user origin must resolve to a
/// [ValueInput] in the same [Region]. The type of that [ValueInput] must be compatible with the
/// [Type] of the [ValueOutput], as defined by [TypeRegistry::can_coerce]. It is valid for the
/// set of users to be empty.
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct ValueOutput {
    /// The type of the value.
    pub ty: Type,

    /// The users of the value.
    pub users: ThinSet<ValueUser>,
}

impl ValueOutput {
    /// Create a new [ValueOutput] with the given [Type] and empty set of users.
    pub fn new(ty: Type) -> Self {
        ValueOutput {
            ty,
            users: ThinSet::new(),
        }
    }
}

/// Enumerates the possible users of a [ValueOutput].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub enum ValueUser {
    /// The user is a region argument in the same region as the [ValueOutput].
    Result(u32),

    /// The user is a node input in the same region as the [ValueInput].
    Input {
        /// The node consuming the output.
        consumer: Node,

        /// The specific index of the input on the [consumer] node.
        input: u32,
    },
}

/// Describes how a node is linked into a region's state chain.
///
/// While we model a node's value-inputs and value-outputs separately, we use this single data
/// structure to model both a node's state-input and state-output at once. This is because, while it
/// is perfectly valid for a node to have a value-input but no value-output (or multiple
/// value-inputs, or multiple value-outputs), a state-input and state-output are always paired
/// together; it is not possible for a node to have a state-input but no state-output. Also, while
/// it's valid for a value-output to have no users, a state-output must always be used; a region's
/// state chain must always flow from the region's state argument to the region's state result, the
/// chain may not have any "breaks". The public modification interface of the [Rvsdg] type is
/// designed to ensure that this invariant is always maintained.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub struct State {
    /// The origin of the state.
    ///
    /// Must resolve to a state output in the same region as the node with which this [State] is
    /// associated.
    pub origin: StateOrigin,

    /// The user of the state.
    ///
    /// Must resolve to a state input in the same region as the node with which this [State] is
    /// associated.
    pub user: StateUser,
}

/// Enumerates the possible origins of a node's state-input.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub enum StateOrigin {
    /// The origin is the region's state argument.
    Argument,

    /// The origin is the state-output of the given [Node].
    ///
    /// The node must be in the same region as the region of the node with which this [State] is
    /// associated.
    Node(Node),
}

/// Enumerates the possible users of a node's state-output.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub enum StateUser {
    /// The user is the region's state result.
    Result,

    /// The user is the state-input of the given [Node].
    ///
    /// The node must be in the same region as the region of the node with which this [State] is
    /// associated.
    Node(Node),
}

/// Describes a region in an [Rvsdg].
///
/// See also the [Regions and Region Data](crate::rvsdg#regions-and-region-data) section of the
/// module-level documentation.
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
    /// The [Node] that owns this region.
    ///
    /// All regions have an owner node, except for the [Rvsdg::global_region].
    ///
    /// # Panics
    ///
    /// Panics when called on the [Rvsdg::global_region].
    pub fn owner(&self) -> Node {
        self.owner.expect("region not correctly initialized")
    }

    /// An ordered set of all nodes in this region.
    ///
    /// Only includes "direct children" of the region: if the region contains nested regions (e.g.,
    /// the branch regions of a [SwitchNode]), then only the owner nodes of those nested regions
    /// are included in this set, not any of the nodes inside the nested regions.
    pub fn nodes(&self) -> &IndexSet<Node> {
        &self.nodes
    }

    /// The list of [ValueOutput]s that represent this region's arguments.
    pub fn value_arguments(&self) -> &[ValueOutput] {
        &self.value_arguments
    }

    /// The list of [ValueInput]s that represent this region's results.
    pub fn value_results(&self) -> &[ValueInput] {
        &self.value_results
    }

    /// The user of the region's state-argument.
    ///
    /// If the region does not contain any nodes that link into the state chain, then the user will
    /// be the region's state-result.
    pub fn state_argument(&self) -> &StateUser {
        &self.state_argument
    }

    /// The origin of the region's state-result.
    ///
    /// If the region does not contain any nodes that link into the state chain, then the origin
    /// will be the region's state-argument.
    pub fn state_result(&self) -> &StateOrigin {
        &self.state_result
    }
}

/// Describes a node in an [Rvsdg].
///
/// See also the [Nodes and Node Data](crate::rvsdg#nodes-and-node-data) section of the
/// module-level documentation.
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct NodeData {
    kind: NodeKind,
    region: Option<Region>,
}

macro_rules! gen_node_data_is_and_expect_simple_kind {
    ($($kind:ident $is_name:ident $expect_name:ident $label:literal,)*) => {
        $(
            pub fn $is_name(&self) -> bool {
                matches!(self.kind, NodeKind::Simple(SimpleNode::$kind(_)))
            }

            pub fn $expect_name(&self) -> &$kind {
                if let NodeKind::Simple(SimpleNode::$kind(op)) = &self.kind {
                    op
                } else {
                    panic!("expected node to be `{}`", $label)
                }
            }
        )*
    };
}

impl NodeData {
    /// The region to which this node belongs.
    pub fn region(&self) -> Region {
        self.region
            .expect("should have a region after initialization")
    }

    /// A reference to the data specific to this node's kind.
    ///
    /// See the [NodeKind] enum for an overview of the different node kinds.
    pub fn kind(&self) -> &NodeKind {
        &self.kind
    }

    /// Searches through the node's [value_inputs] for the first input that has an origin that
    /// matches the given `origin`.
    ///
    /// Returns the index of the value-input for which a matching origin is found, or `None`
    /// otherwise.
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

    gen_node_data_is_and_expect_simple_kind! {
        ConstU32 is_const_u32 expect_const_u32 "a `u32` constant",
        ConstI32 is_const_i32 expect_const_i32 "a `i32` constant",
        ConstF32 is_const_f32 expect_const_f32 "a `f32` constant",
        ConstBool is_const_bool expect_const_bool "a `bool` constant",
        ConstPredicate is_const_predicate expect_const_predicate "a `predicate` constant",
        ConstPtr is_const_ptr expect_const_ptr "a pointer constant",
        ConstFallback is_const_fallback expect_const_fallback "a fallback-value constant",
        OpAlloca is_op_alloca expect_op_alloca "an `alloca` operation",
        OpLoad is_op_load expect_op_load "a `load` operation",
        OpStore is_op_store expect_op_store "a `store` operation",
        OpFieldPtr is_op_field_ptr expect_op_field_ptr "a `field-pointer` operation",
        OpElementPtr is_op_element_ptr expect_op_element_ptr "a `element-pointer` operation",
        OpExtractField is_op_extract_field expect_op_extract_field "an `extract-field` operation",
        OpExtractElement is_op_extract_element expect_op_extract_element "an `extract-element` operation",
        OpDiscriminantPtr is_op_discriminant_ptr expect_op_discriminant_ptr "a `discriminant-pointer` operation",
        OpVariantPtr is_op_variant_ptr expect_op_variant_ptr "a `variant-pointer` operation",
        OpGetDiscriminant is_op_get_discriminant expect_op_get_discriminant "a `get-discriminant` operation",
        OpSetDiscriminant is_op_set_discriminant expect_op_set_discriminant "a `set-discriminant` operation",
        OpOffsetSlice is_op_offset_slice expect_op_offset_slice "an `offset-slice` operation",
        OpGetSliceOffset is_op_get_slice_offset expect_op_get_slice_offset "a `get-slice-offset` operation",
        OpUnary is_op_unary expect_op_unary "a `unary` operation",
        OpBinary is_op_binary expect_op_binary "a `binary` operation",
        OpVector is_op_vector expect_op_vector "a `vector` operation",
        OpMatrix is_op_matrix expect_op_matrix "a `matrix` operation",
        OpCaseToBranchSelector is_op_case_to_branch_selector expect_op_case_to_branch_selector "an `op-case-to-branch-selector` operation",
        OpBoolToBranchSelector is_op_bool_to_branch_selector expect_op_bool_to_branch_selector "an `op-bool-to-branch-selector` operation",
        OpU32ToBranchSelector is_op_u32_to_branch_selector expect_op_u32_to_branch_selector "an `op-u32-to-branch-selector` operation",
        OpBranchSelectorToCase is_op_branch_selector_to_case expect_op_branch_selector_to_case "an `op-branch-selector-to-case` operation",
        OpConvertToU32 is_op_convert_to_u32 expect_op_convert_to_u32 "a `convert-to-u32` operation",
        OpConvertToI32 is_op_convert_to_i32 expect_op_convert_to_i32 "a `convert-to-i32` operation",
        OpConvertToF32 is_op_convert_to_f32 expect_op_convert_to_f32 "a `convert-to-f32` operation",
        OpConvertToBool is_op_convert_to_bool expect_op_convert_to_bool "a `convert-to-bool` operation",
        OpArrayLength is_op_array_length expect_op_array_length "an `array-length` operation",
        OpCall is_op_call expect_op_call "a `call` operation",
        ValueProxy is_value_proxy expect_value_proxy "a value-proxy node",
        Reaggregation is_reaggregation expect_reaggregation "a reaggregation node",
    }

    fn expect_op_case_to_branch_selector_mut(&mut self) -> &mut OpCaseToBranchSelector {
        if let NodeKind::Simple(SimpleNode::OpCaseToBranchSelector(op)) = &mut self.kind {
            op
        } else {
            panic!("expected node to be a `OpCaseToBranchSelector` node")
        }
    }
}

/// Enumerates the different kinds of nodes that can be contained in an [Rvsdg].
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

/// [Rvsdg] node that models a uniform-binding global value.
///
/// A uniform-binding node must always belong to the [Rvsdg::global_region] region. It has a single
/// value-output that may serve as the origin of a dependency input for a [FunctionNode].
///
/// See also [UniformBindingData](crate::core::UniformBindingData) and the
/// [UniformBindingRegistry](crate::core::UniformBindingRegistry).
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct UniformBindingNode {
    binding: UniformBinding,
    output: ValueOutput,
}

impl UniformBindingNode {
    /// Returns the [UniformBinding] key for this node's uniform binding.
    ///
    /// May be resolved to a [UniformBindingData] reference with the
    /// [Module::uniform_bindings](crate::core::Module::uniform_bindings) registry for the
    /// [Module] with which this node's [Rvsdg] is associated.
    pub fn binding(&self) -> UniformBinding {
        self.binding
    }

    /// The output value of this node.
    ///
    /// Represents the uniform-binding's value. May serve as the origin of a dependency input for
    /// a [FunctionNode], which makes the value available inside the [FunctionNode]'s body region.
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

/// [Rvsdg] node that models a storage-binding global value.
///
/// A storage-binding node must always belong to the [Rvsdg::global_region] region. It has a single
/// value-output that may serve as the origin of a dependency input for a [FunctionNode].
///
/// See also [StorageBindingData](crate::core::StorageBindingData) and the
/// [StorageBindingRegistry](crate::core::StorageBindingRegistry).
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct StorageBindingNode {
    binding: StorageBinding,
    output: ValueOutput,
}

impl StorageBindingNode {
    /// Returns the [StorageBinding] key for this node's storage binding.
    ///
    /// May be resolved to a [StorageBindingData] reference with the
    /// [Module::storage_bindings](crate::core::Module::storage_bindings) registry for the
    /// [Module] with which this node's [Rvsdg] is associated.
    pub fn binding(&self) -> StorageBinding {
        self.binding
    }

    /// The output value of this node.
    ///
    /// Represents the storage-binding's value. May serve as the origin of a dependency input for
    /// a [FunctionNode], which makes the value available inside the [FunctionNode]'s body region.
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

/// [Rvsdg] node that models a workgroup-binding global value.
///
/// A workgroup-binding node must always belong to the [Rvsdg::global_region] region. It has a
/// single value-output that may serve as the origin of a dependency input for a [FunctionNode].
///
/// See also [WorkgroupBindingData](crate::core::WorkgroupBindingData) and the
/// [WorkgroupBindingRegistry](crate::core::WorkgroupBindingRegistry).
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct WorkgroupBindingNode {
    binding: WorkgroupBinding,
    output: ValueOutput,
}

impl WorkgroupBindingNode {
    /// Returns the [WorkgroupBinding] key for this node's workgroup binding.
    ///
    /// May be resolved to a [WorkgroupBindingData] reference with the
    /// [Module::workgroup_bindings](crate::core::Module::workgroup_bindings) registry for the
    /// [Module] with which this node's [Rvsdg] is associated.
    pub fn binding(&self) -> WorkgroupBinding {
        self.binding
    }

    /// The output value of this node.
    ///
    /// Represents the workgroup-binding's value. May serve as the origin of a dependency input for
    /// a [FunctionNode], which makes the value available inside the [FunctionNode]'s body region.
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

/// [Rvsdg] node that models a constant value.
///
/// A constant node must always belong to the [Rvsdg::global_region] region. It has a single
/// value-output that may serve as the origin of a dependency input for a [FunctionNode].
///
/// See also [Constant] and the [ConstantRegistry](crate::core::ConstantRegistry).
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct ConstantNode {
    constant: Constant,
    output: ValueOutput,
}

impl ConstantNode {
    /// Returns the [Constant] key for this node's constant value.
    ///
    /// May be resolved to a [ConstantData](crate::core::ConstantData) reference with the
    /// [Module::constants](crate::core::Module::constants) registry for the
    /// [Module] with which this node's [Rvsdg] is associated.
    pub fn constant(&self) -> Constant {
        self.constant
    }

    /// The output value of this node.
    ///
    /// Represents the constant value. May serve as the origin of a dependency input for
    /// a [FunctionNode], which makes the value available inside the [FunctionNode]'s body region.
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

/// Models the symmetric splitting-then-merging of control-flow in an [Rvsdg].
///
/// A switch node owns one or more "branch" regions. Based on a "branch selector" value-input,
/// control-flow is directed to one specific branch region. After this branch region completes,
/// control-flow is directed back to the region that contains the switch node.
///
/// # Value-Inputs and Branch Arguments
///
/// The first [ValueInput] to a switch node is always the branch-selector value. Additionally, it
/// may take any number of "entry" input-values. Any such entry-values are made available as region
/// arguments in each of the switch node's branch regions. The `N`th argument of a branch region
/// corresponds to the `N`th entry-value of the switch node, which corresponds to the `N+1`th
/// value-input of the switch node (as the first value-input is the branch-selector value).
///
/// A set of entry input-values may be declared when first creating the switch node with
/// [Rvsdg::add_switch_node]. An entry input-value may also be added to an existing switch node
/// with [Rvsdg::add_switch_input], or removed from an existing switch node with
/// [Rvsdg::remove_switch_input].
///
/// Branch region arguments may be unused (in fact, it is typical for each branch to only use a
/// subset of its arguments). An entry input-value is considered used if the corresponding branch
/// argument is used in *any* of the switch node's branch regions; an entry input-value is
/// considered unused if the corresponding branch argument is unused in *all* of the switch node's
/// branch regions. Only unused entry input-values may be removed from a switch node.
///
/// # Value Outputs and Branch Results
///
/// Switch nodes have value-outputs. For every value-output, each branch region has a corresponding
/// result. A switch node's value-outputs may be specified when first creating the switch node with
/// [Rvsdg::add_switch_node]. A value-output may also be added to an existing switch node with
/// [Rvsdg::add_switch_output], or removed from an existing switch node with
/// [Rvsdg::remove_switch_output].
///
/// If a switch node is created with value-outputs, or value outputs are later added, then their
/// corresponding branch results will have been initialized with "placeholder" origins (see
/// [ValueOrigin::placeholder]). These placeholder origins must then be replaced with valid origins,
/// for example, by reconnecting the region results (see [Rvsdg::reconnect_result]) to region
/// arguments or to node outputs (after nodes have been added to the branch). Placeholder
/// value-origins are only meant to be a temporary part of an RVSDG during construction or a
/// transformation; in the "final" RVSDG, all results for all branches must have valid origins.
///
/// # State
///
/// A region is said to "not use state" if the region's state argument connects directly to its
/// state result; conversely, a region is said to "use state" if the region's state argument does
/// not connect to the region's state result, but instead flows through one or more nodes.
///
/// A switch node is said to "use state" if any of its branch regions "use state". If a switch node
/// uses state, then it must be linked into its region's state chain (see
/// [Rvsdg::link_switch_state]).
///
/// It is not invalid for a switch node that does *not* use state to be linked into its region's
/// state chain. However, this creates a reordering constraint that may restrict optimization
/// opportunities. Therefore, switch nodes that no longer use state should be unlinked from the
/// state chain (see [Rvsdg::unlink_switch_state]).
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct SwitchNode {
    value_inputs: Vec<ValueInput>,
    value_outputs: Vec<ValueOutput>,
    branches: Vec<Region>,
    state: Option<State>,
}

impl SwitchNode {
    /// The branch-selector value-input for this switch node.
    ///
    /// This is always the first value-input of the switch node.
    pub fn branch_selector(&self) -> &ValueInput {
        &self.value_inputs[0]
    }

    /// The entry value-inputs for this switch node.
    ///
    /// For a [SwitchNode] with `N` total value-inputs, these are inputs `1..N` (the first
    /// value-input is the branch-selector input).
    ///
    /// See the [Value-Inputs and Branch Arguments](SwitchNode#value-inputs-and-branch-arguments)
    /// section of the documentation for the [SwitchNode] struct for details.
    pub fn entry_inputs(&self) -> &[ValueInput] {
        &self.value_inputs[1..]
    }

    /// List the branch regions for this switch node.
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

/// Models looping control-flow in an [Rvsdg].
///
/// Owns a "loop region". When control-flow reaches the loop node, control-flow is passed into this
/// loop region. The loop region's first result is the "reentry condition" value. Once the loop
/// region completes, if the reentry-condition is `true`, then control-flow is redirected back to
/// the start of the loop region. This repeats until the reentry-condition is `false`, at which
/// point control-flow is passed back to the loop node's outer region.
///
/// Note that the reentry-condition does not ever need to become false; this creates an infinite
/// loop. Also note that the loop region will always run at least once; a loop-node models a
/// "tail-controlled" loop.
///
/// # Loop Values
///
/// For loop nodes, value-inputs and value-outputs (and consequently loop region arguments and
/// loop region results) are closely tied together. A loop node with `N` value-inputs will have
/// `N` value-outputs, `N` loop region arguments, and `N+1` loop region results. Additionally,
/// if the value-input with index `I` is of type `T`, the value-output `I` will be of type `T`,
/// loop region argument `I` will be of type `T`, and loop region result `I+1` will be of type `T`.
///
/// When control-flow first reaches the loop node and is passed into the loop region, the loop
/// region's argument values will flow from the loop node's value-inputs. The first loop region
/// result will be the loop node's reentry-condition. If the reentry-condition is `true`, then loop
/// region results `1..N+1` will be "reentry" values: control-flow will be redirected to the start
/// of the loop region, but on this iteration the loop region's argument values will flow from the
/// region's result values from the previous iteration. If the reentry-condition is `false`, then
/// the loop region's result will flow to the loop node's output-values.
///
/// A set of loop values may be declared when first creating the loop node with
/// [Rvsdg::add_loop_node]. This method takes as an argument a set of value-inputs for the loop
/// node, and because of the strong coupling described above, this then also completely defines
/// the loop node's value-outputs, and its loop region's arguments and results. Loop values may
/// also be added to existing loop nodes with [Rvsdg::add_loop_input] or removed from existing
/// loop nodes with [Rvsdg::remove_loop_input].
///
/// When a loop-value with value-input index `I` is first created/added, the loop region result
/// `I+1` for that loop-value will be connected to the loop region argument `I` for that loop-value.
///
/// # State
///
/// A region is said to "not use state" if the region's state argument connects directly to its
/// state result; conversely, a region is said to "use state" if the region's state argument does
/// not connect to the region's state result, but instead flows through one or more nodes.
///
/// A loop node is said to "use state" if its loop region uses state. If a loop node uses state,
/// then it must be linked into its region's state chain (see [Rvsdg::link_loop_state]).
///
/// It is not invalid for a loop node that does *not* use state to be linked into its region's state
/// chain. However, this creates a reordering constraint that may restrict optimization
/// opportunities. Therefore, loop nodes that no longer use state should be unlinked from the state
/// chain (see [Rvsdg::unlink_loop_state]).
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct LoopNode {
    value_inputs: Vec<ValueInput>,
    value_outputs: Vec<ValueOutput>,
    state: Option<State>,
    loop_region: Region,
}

impl LoopNode {
    /// This loop node's loop region.
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

/// Represents an SLIR intrinsic operation as a node in an [Rvsdg].
///
/// SLIR consists of three different intermediate representations: the [cfg](crate::cfg)
/// representation, the [rvsdg](crate::rvsdg) representation, and the [scf](crate::scf)
/// representation. Across all representations, SLIR's intrinsic operations share common behavior
/// (e.g.: argument type validation, result type inference, etc.). Rather than reimplement this
/// common behavior for each of the IRs, we implement the common logic in the [intrinsic] module.
/// This node kind represents such intrinsic operations for the RVSDG representation.
///
/// We define a set of type-aliases to allow for concise type names. A type-alias (e.g., [OpBinary])
/// should typically be preferred over the underlying [IntrinsicNode] type (e.g.,
/// `IntrinsicNode<OpBinary>`).
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct IntrinsicNode<T> {
    intrinsic: T,
    value_inputs: SmallVec<[ValueInput; 2]>,
    value_output: Option<ValueOutput>,
    state: Option<State>,
}

impl<T: Intrinsic> IntrinsicNode<T> {
    /// The intrinsic operation represented by this node.
    pub fn intrinsic(&self) -> &T {
        &self.intrinsic
    }
}

impl<T> Connectivity for IntrinsicNode<T> {
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
        self.state.as_ref()
    }

    fn state_mut(&mut self) -> Option<&mut State> {
        self.state.as_mut()
    }
}

macro_rules! gen_intrinsic_value_input {
    ($name:ident, $index:literal) => {
        pub fn $name(&self) -> &ValueInput {
            &self.value_inputs[$index]
        }
    };
}

macro_rules! gen_intrinsic_value_output {
    () => {
        pub fn value_output(&self) -> &ValueOutput {
            self.value_output.as_ref().unwrap()
        }
    };
}

pub type OpAlloca = IntrinsicNode<intrinsic::OpAlloca>;

impl OpAlloca {
    pub fn ty(&self) -> Type {
        self.intrinsic.ty
    }

    gen_intrinsic_value_output!();
}

pub type OpLoad = IntrinsicNode<intrinsic::OpLoad>;

impl OpLoad {
    gen_intrinsic_value_input!(ptr_input, 0);
    gen_intrinsic_value_output!();
}

pub type OpStore = IntrinsicNode<intrinsic::OpStore>;

impl OpStore {
    gen_intrinsic_value_input!(ptr_input, 0);
    gen_intrinsic_value_input!(value_input, 1);
}

pub type OpExtractField = IntrinsicNode<intrinsic::OpExtractField>;

impl OpExtractField {
    pub fn field_index(&self) -> u32 {
        self.intrinsic.field_index
    }

    gen_intrinsic_value_input!(value_input, 0);
    gen_intrinsic_value_output!();
}

pub type OpExtractElement = IntrinsicNode<intrinsic::OpExtractElement>;

impl OpExtractElement {
    gen_intrinsic_value_input!(value_input, 0);
    gen_intrinsic_value_input!(index_input, 1);
    gen_intrinsic_value_output!();
}

pub type OpFieldPtr = IntrinsicNode<intrinsic::OpFieldPtr>;

impl OpFieldPtr {
    pub fn field_index(&self) -> u32 {
        self.intrinsic.field_index
    }

    gen_intrinsic_value_input!(ptr_input, 0);
    gen_intrinsic_value_output!();
}

pub type OpElementPtr = IntrinsicNode<intrinsic::OpElementPtr>;

impl OpElementPtr {
    gen_intrinsic_value_input!(ptr_input, 0);
    gen_intrinsic_value_input!(index_input, 1);
    gen_intrinsic_value_output!();
}

pub type OpDiscriminantPtr = IntrinsicNode<intrinsic::OpDiscriminantPtr>;

impl OpDiscriminantPtr {
    gen_intrinsic_value_input!(ptr_input, 0);
    gen_intrinsic_value_output!();
}

pub type OpVariantPtr = IntrinsicNode<intrinsic::OpVariantPtr>;

impl OpVariantPtr {
    pub fn variant_index(&self) -> u32 {
        self.intrinsic.variant_index
    }

    gen_intrinsic_value_input!(ptr_input, 0);
    gen_intrinsic_value_output!();
}

pub type OpGetDiscriminant = IntrinsicNode<intrinsic::OpGetDiscriminant>;

impl OpGetDiscriminant {
    gen_intrinsic_value_input!(ptr_input, 0);
    gen_intrinsic_value_output!();
}

pub type OpSetDiscriminant = IntrinsicNode<intrinsic::OpSetDiscriminant>;

impl OpSetDiscriminant {
    pub fn variant_index(&self) -> u32 {
        self.intrinsic.variant_index
    }

    gen_intrinsic_value_input!(ptr_input, 0);
}

pub type OpOffsetSlice = IntrinsicNode<intrinsic::OpOffsetSlice>;

impl OpOffsetSlice {
    gen_intrinsic_value_input!(ptr_input, 0);
    gen_intrinsic_value_input!(offset_input, 1);
    gen_intrinsic_value_output!();
}

pub type OpGetSliceOffset = IntrinsicNode<intrinsic::OpGetSliceOffset>;

impl OpGetSliceOffset {
    gen_intrinsic_value_input!(ptr_input, 0);
    gen_intrinsic_value_output!();
}

pub type OpUnary = IntrinsicNode<intrinsic::OpUnary>;

impl OpUnary {
    pub fn operator(&self) -> UnaryOperator {
        self.intrinsic.operator
    }

    gen_intrinsic_value_input!(value_input, 0);
    gen_intrinsic_value_output!();
}

pub type OpBinary = IntrinsicNode<intrinsic::OpBinary>;

impl OpBinary {
    pub fn operator(&self) -> BinaryOperator {
        self.intrinsic.operator
    }

    gen_intrinsic_value_input!(lhs_input, 0);
    gen_intrinsic_value_input!(rhs_input, 1);
    gen_intrinsic_value_output!();
}

pub type OpVector = IntrinsicNode<intrinsic::OpVector>;

impl OpVector {
    pub fn ty(&self) -> &ty::Vector {
        &self.intrinsic.ty
    }

    gen_intrinsic_value_output!();
}

pub type OpMatrix = IntrinsicNode<intrinsic::OpMatrix>;

impl OpMatrix {
    pub fn ty(&self) -> &ty::Matrix {
        &self.intrinsic.ty
    }

    gen_intrinsic_value_output!();
}

pub type OpCaseToBranchSelector = IntrinsicNode<intrinsic::OpCaseToBranchSelector>;

impl OpCaseToBranchSelector {
    pub fn cases(&self) -> &[u32] {
        &self.intrinsic.cases
    }

    gen_intrinsic_value_input!(value_input, 0);
    gen_intrinsic_value_output!();
}

pub type OpBoolToBranchSelector = IntrinsicNode<intrinsic::OpBoolToBranchSelector>;

impl OpBoolToBranchSelector {
    gen_intrinsic_value_input!(value_input, 0);
    gen_intrinsic_value_output!();
}

pub type OpU32ToBranchSelector = IntrinsicNode<intrinsic::OpU32ToBranchSelector>;

impl OpU32ToBranchSelector {
    pub fn branch_count(&self) -> u32 {
        self.intrinsic.branch_count
    }

    gen_intrinsic_value_input!(value_input, 0);
    gen_intrinsic_value_output!();
}

pub type OpBranchSelectorToCase = IntrinsicNode<intrinsic::OpBranchSelectorToCase>;

impl OpBranchSelectorToCase {
    pub fn cases(&self) -> &[u32] {
        &self.intrinsic.cases
    }

    gen_intrinsic_value_input!(value_input, 0);
    gen_intrinsic_value_output!();
}

pub type OpConvertToU32 = IntrinsicNode<intrinsic::OpConvertToU32>;

impl OpConvertToU32 {
    gen_intrinsic_value_input!(value_input, 0);
    gen_intrinsic_value_output!();
}

pub type OpConvertToI32 = IntrinsicNode<intrinsic::OpConvertToI32>;

impl OpConvertToI32 {
    gen_intrinsic_value_input!(value_input, 0);
    gen_intrinsic_value_output!();
}

pub type OpConvertToF32 = IntrinsicNode<intrinsic::OpConvertToF32>;

impl OpConvertToF32 {
    gen_intrinsic_value_input!(value_input, 0);
    gen_intrinsic_value_output!();
}

pub type OpConvertToBool = IntrinsicNode<intrinsic::OpConvertToBool>;

impl OpConvertToBool {
    gen_intrinsic_value_input!(value_input, 0);
    gen_intrinsic_value_output!();
}

pub type OpArrayLength = IntrinsicNode<intrinsic::OpArrayLength>;

impl OpArrayLength {
    gen_intrinsic_value_input!(ptr_input, 0);
    gen_intrinsic_value_output!();
}

/// Node that models a function call in an [Rvsdg].
///
/// # Value Inputs
///
/// The first input to an op-call node represents the function that is being called. This function
/// value will flow from a dependency argument of the body region that contains this node, see the
/// [Dependency Inputs](FunctionNode#depenency-inputs) section of the [FunctionNode] documentation.
/// Note that this dependency value may be passed into [SwitchNode]s and [LoopNode] for function
/// calls inside nested branch or loop regions.
///
/// The remaining value-inputs represent the call arguments. These value-inputs will flow into the
/// callee's body region as "call arguments", see the [Body Region](FunctionNode#body-region)
/// section of the [FunctionNode] documentation.
///
/// # Value Output
///
/// If an op-call node's callee function has a return value, then the op-call node has a single
/// value-output that represents the return value; otherwise the op-call node does not have any
/// value-outputs.
///
/// # State
///
/// An op-call function must currently always be linked into its region's state chain, even if the
/// callee function's body region does not "use state" (its state argument connects directly to its
/// state result, without passing through any nodes). This is because currently, we immediately
/// exhaustively inline all function calls before performing any optimizations, and therefore this
/// does not end up constraining any optimizations.
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct OpCall {
    value_inputs: Vec<ValueInput>,
    value_output: Option<ValueOutput>,
    state: State,
}

impl OpCall {
    /// The value-input that represents the callee function.
    pub fn fn_input(&self) -> &ValueInput {
        &self.value_inputs[0]
    }

    /// List the value-inputs that will be passed as call arguments to the callee.
    pub fn argument_inputs(&self) -> &[ValueInput] {
        &self.value_inputs[1..]
    }

    /// Resolves the [Function] identifier from the [fn_input]'s type using the `ty_registry`.
    pub fn resolve_fn(&self, ty_registry: &TypeRegistry) -> Function {
        *ty_registry.kind(self.value_inputs[0].ty).expect_fn()
    }

    /// Returns the value-output that represents the callee function's return value if the callee
    /// function has a return value, or `None` otherwise.
    pub fn value_output(&self) -> Option<&ValueOutput> {
        self.value_output.as_ref()
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
    /// The type of the value.
    pub fn ty(&self) -> Type {
        self.output.ty
    }

    /// The origin for users of the value.
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

/// A node that proxies a value origin.
///
/// This is a helper node used by transformation passes. The main purpose for this node kind is to
/// allow the modification of a node's value-outputs, while the RVSDG graph is being visited.
/// Without proxy nodes, adding value-users to a value-output grows that value-output's user-set. If
/// that value-output's user-set is currently being iterated over, this may disturb that iteration
/// and lead to unexpected results.
///
/// Proxying the value-user with [Rvsdg::proxy_origin_user] can avoid such problems. This replaces
/// a single user with the new proxy node without changing the user count or user order. A proxy
/// node has a single value-output, and the original value-user is added as the only initial user of
/// the proxy node's value-output. Additional value-users can now be added to the proxy node's
/// value-output. Because the proxy node was newly added, we can be assured that the proxy node's
/// value-users are not already being visited or iterated over.
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct ValueProxy {
    input: ValueInput,
    output: ValueOutput,
}

impl ValueProxy {
    /// The value that is being proxied.
    pub fn input(&self) -> &ValueInput {
        &self.input
    }

    /// The proxy value.
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
/// This is a temporary helper node used by the scalar-replacement pass. The first input represents
/// the original (pointer to an) aggregate value. The remaining inputs represent the part values
/// the original value is being split into.
///
/// This node acts like a "save-state" for the scalar-replacement pass. If we can't currently split
/// a particular value-flow path, then a reaggregation node allows us to pause, split other alloca
/// nodes or perform different transforms, then continue from the aggregation node later. Currently,
/// this is used for switch node output splitting when the switch node output is a slice type and
/// we won't know how many parts to split into until all branches have been processed.
///
/// This "operation" is not implementable on any back-end; it is only to be used as a temporary node
/// during RVSDG transformation, and no nodes of this kind should be left in the graph when
/// scalar-replacement is complete.
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct Reaggregation {
    inputs: Vec<ValueInput>,
    output: ValueOutput,
}

impl Reaggregation {
    /// Value-input for the original aggregate value.
    pub fn original(&self) -> &ValueInput {
        &self.inputs[0]
    }

    /// The value-inputs for the part-values the [original] aggregate value is being split into.
    pub fn parts(&self) -> &[ValueInput] {
        &self.inputs[1..]
    }

    /// The reaggregated output-value.
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
        /// Enumerates all "simple" node kinds that can be contained in an [Rvsdg].
        ///
        /// These are node kinds that cannot be part of the global region and do not themselves
        /// contain a region.
        ///
        /// See also [NodeKind].
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
    OpExtractField,
    OpExtractElement,
    OpFieldPtr,
    OpElementPtr,
    OpDiscriminantPtr,
    OpVariantPtr,
    OpGetDiscriminant,
    OpSetDiscriminant,
    OpOffsetSlice,
    OpGetSliceOffset,
    OpUnary,
    OpBinary,
    OpVector,
    OpMatrix,
    OpCaseToBranchSelector,
    OpBoolToBranchSelector,
    OpU32ToBranchSelector,
    OpBranchSelectorToCase,
    OpConvertToU32,
    OpConvertToI32,
    OpConvertToF32,
    OpConvertToBool,
    OpArrayLength,
    OpCall,
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

/// Type that helps facilitate serializing an [Rvsdg] instance without its [TypeRegistry].
///
/// An [Rvsdg] instance contains a shared reference to its [TypeRegistry]. However, when serializing
/// the [Rvsdg] alongside its [Module] - which shares a reference to the same [TypeRegistry] - this
/// would produce duplicate data in the serialization artifact and - after deserialization - would
/// result in the [Rvsdg]'s [TypeRegistry] being decoupled from the [Module]'s [TypeRegistry], which
/// is not desired.
///
/// Instead [Rvsdg::as_data] may be used to serialize the [Rvsdg] without its [TypeRegistry]. It may
/// then be deserialized into an [RvsdgData] instance, which may in turn be used to reconstruct the
/// [Rvsdg] instance via [Rvsdg::from_ty_and_data] where a shared reference to the [TypeRegistry] is
/// obtained from the accompanying [Module] instance (e.g., by cloning [Module::ty]).
#[derive(Clone, Serialize, Debug)]
pub struct RvsdgAsData<'a> {
    regions: &'a SlotMap<Region, RegionData>,
    nodes: &'a SlotMap<Node, NodeData>,
    global_region: &'a Region,
    function_node: &'a FxHashMap<Function, Node>,
}

/// Type that helps facilitate deserializing an [Rvsdg] instance without its [TypeRegistry].
///
/// For details, see the documentation for [RvsdgAsData].
#[derive(Clone, Deserialize, Debug)]
pub struct RvsdgData {
    regions: SlotMap<Region, RegionData>,
    nodes: SlotMap<Node, NodeData>,
    global_region: Region,
    function_node: FxHashMap<Function, Node>,
}

/// A Regionalized Value-State Dependency Graph.
///
/// For details, refer to the [module-level documentation](crate::rvsdg).
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Rvsdg {
    ty: TypeRegistry,
    regions: SlotMap<Region, RegionData>,
    nodes: SlotMap<Node, NodeData>,
    global_region: Region,
    function_node: FxHashMap<Function, Node>,
}

impl Rvsdg {
    /// Creates a new [Rvsdg] instance with the given `type_registry`.
    ///
    /// The `type_registry` should be a clone of the [Module::ty] type-registry for the [Module]
    /// with which this [Rvsdg] is associated.
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

    /// Reconstructs an [Rvsdg] instance from its data and a shared [TypeRegistry].
    ///
    /// See the documentation for [RvsdgAsData] for details.
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

    /// Returns a representation of the [Rvsdg]'s data without its [TypeRegistry].
    ///
    /// See the documentation for [RvsdgAsData] for details.
    pub fn as_data(&self) -> RvsdgAsData<'_> {
        RvsdgAsData {
            regions: &self.regions,
            nodes: &self.nodes,
            global_region: &self.global_region,
            function_node: &self.function_node,
        }
    }

    /// Serializes the RVSDG to a bincode format and writes it to the given writer.
    ///
    /// This is useful for debugging purposes, as the resulting bincode file can be loaded and
    /// visualized by the `slir-explorer` to view a rendered SVG version of the RVSDG graph.
    pub fn dump<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        bincode::serde::encode_into_std_write(self, writer, bincode::config::standard()).map_err(
            |err: bincode::error::EncodeError| {
                std::io::Error::new(std::io::ErrorKind::Other, err.to_string())
            },
        )?;

        Ok(())
    }

    /// Serializes the RVSDG to a bincode format and writes it to a file with the given path.
    ///
    /// Shorthand for [Rvsdg::dump] where the `writer` is a [File]. Creates the file if it does not
    /// yet exist, or truncates an existing file before writing the RVSDG dump.
    ///
    /// Depending on the platform, this function may fail if the full directory path does not exist.
    pub fn dump_to_file<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let mut file = File::create(path)?;

        self.dump(&mut file)?;

        Ok(())
    }

    /// The [TypeRegistry] that stores the type information for all types used by this [Rvsdg].
    pub fn ty(&self) -> &TypeRegistry {
        &self.ty
    }

    /// The RVSDG's global region,
    ///
    /// Contains [FunctionNode]s and global value nodes ([ConstantNode], [UniformBindingNode],
    /// [StorageBindingNode], [WorkgroupBindingNode]). The global region has no arguments and no
    /// results. This is also the only region in an RVSDG that does not have an owner node.
    pub fn global_region(&self) -> Region {
        self.global_region
    }

    /// Creates a new [UniformBindingNode] in this [Rvsdg]'s global region for the given `binding`.
    ///
    /// The `module` should be the [Module] with which this [Rvsdg] is associated (this [Rvsdg]'s
    /// [Rvsdg::ty] type-registry should be the same type-registry as the `module`'s [Module::ty]
    /// type-registry).
    ///
    /// # Panics
    ///
    /// Panics of the binding cannot be resolved with the `module`'s [Module::uniform_bindings]
    /// registry.
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

    /// Creates a new [StorageBindingNode] in this [Rvsdg]'s global region for the given `binding`.
    ///
    /// The `module` should be the [Module] with which this [Rvsdg] is associated (this [Rvsdg]'s
    /// [Rvsdg::ty] type-registry should be the same type-registry as the `module`'s [Module::ty]
    /// type-registry).
    ///
    /// # Panics
    ///
    /// Panics if the binding cannot be resolved with the `module`'s [Module::storage_bindings]
    /// registry.
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

    /// Creates a new [WorkgroupBindingNode] in this [Rvsdg]'s global region for the given `binding`.
    ///
    /// The `module` should be the [Module] with which this [Rvsdg] is associated (this [Rvsdg]'s
    /// [Rvsdg::ty] type-registry should be the same type-registry as the `module`'s [Module::ty]
    /// type-registry).
    ///
    /// # Panics
    ///
    /// Panics if the binding cannot be resolved with the `module`'s [Module::workgroup_bindings]
    /// registry.
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

    /// Creates a new [ConstantNode] in this [Rvsdg]'s global region for the given `constant`.
    ///
    /// The `module` should be the [Module] with which this [Rvsdg] is associated (this [Rvsdg]'s
    /// [Rvsdg::ty] type-registry should be the same type-registry as the `module`'s [Module::ty]
    /// type-registry).
    ///
    /// # Panics
    ///
    /// Panics if the constant cannot be resolved with the `module`'s [Module::constants] registry.
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

    /// Registers the given `function` with this [Rvsdg].
    ///
    /// Creates a new [FunctionNode] in this [Rvsdg]'s global region. The `module` should be the
    /// [Module] with which this [Rvsdg] is associated (this [Rvsdg]'s [Rvsdg::ty] type-registry
    /// should be the same type-registry as the `module`'s [Module::ty] type-registry). The
    /// [FunctionNode]'s value-inputs will connect to the provided set of `dependencies`. For more
    /// information on a [FunctionNode]'s dependencies, see the
    /// [Dependency Inputs](FunctionNode#depdendency-inputs) section of the [FunctionNode]
    /// documentation.
    ///
    /// Will look up the [FnSig] for the `function` in the `module`'s
    /// [FnSigRegistry](crate::core::FnSigRegistry). The [FnSig] is used to derive the set of
    /// call arguments for the [FunctionNode]'s body region. If the [FnSig] specifies a return type,
    /// then the [FunctionNode] will have a single value-output that represents the function's
    /// return value, and the body region has a single result that flows to that value-output; if
    /// the [FnSig] does not specify a return type, then the [FunctionNode] will have no
    /// value-outputs and its body region will have no results. If the body region has a result,
    /// then this result will initially be connected to a "placeholder" origin (see
    /// [ValueOrigin::placeholder]). The region result must later be adjusted to connect to a valid
    /// origin with [Rvsdg::reconnect_region_result]. For details on a [FunctionNode]'s body region,
    /// refer to the [Body Region](FunctionNode#body-region) section of the [FunctionNode]
    /// documentation.
    ///
    /// Returns a `(Node, Region)` pair where the [Node] is the identifier for the [FunctionNode]
    /// that was added to the global region, and the [Region] is the identifier for that
    /// [FunctionNode]'s body region.
    ///
    /// # Panics
    ///
    /// Panics if the `function` does not resolve to an [FnSig] in the `module`'s
    /// [FnSigRegistry](crate::core::FnSigRegistry).
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
                function,
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

    /// Returns an iterator over all functions registered with this [Rvsdg].
    ///
    /// The iterator produces `(Function, Node)` pairs where the [Function] is the identifier for
    /// the registered function and the [Node] identifies the [FunctionNode] that was added to this
    /// [Rvsdg]'s body region.
    ///
    /// See also [Rvsdg::register_function].
    pub fn registered_functions(&self) -> impl Iterator<Item = (Function, Node)> {
        self.function_node.iter().map(|(f, n)| (*f, *n))
    }

    /// Returns the [FunctionNode] for the `function` if the `function` is registered with this
    /// [Rvsdg], or `None` otherwise.
    ///
    /// See also [Rvsdg::register_function].
    pub fn get_function_node(&self, function: Function) -> Option<Node> {
        self.function_node.get(&function).copied()
    }

    /// Whether this [Rvsdg] still contains [RegionData] for the given `region`.
    pub fn is_live_region(&self, region: Region) -> bool {
        self.regions.contains_key(region)
    }

    /// Whether this [Rvsdg] still contains [NodeData] for the given `node`.
    pub fn is_live_node(&self, node: Node) -> bool {
        self.nodes.contains_key(node)
    }

    /// Adds a [SwitchNode] to the given `region`.
    ///
    /// May supply a set of `value_inputs` and `value_outputs` for the node at creation.
    /// [ValueInput]s may also be added later with [Rvsdg::add_switch_input], or removed with
    /// [Rvsdg::remove_switch_input]; [ValueOutput]s may also be added later with
    /// [Rvsdg::add_switch_output], or removed with [Rvsdg::remove_switch_output].
    ///
    /// May optionally supply a `state_origin`: if supplied, then the node will be inserted into the
    /// state chain between the origin and the origin's prior user; if `None` the switch node will
    /// not be part of the state chain. The [SwitchNode] may also be linked into the state chain
    /// later with [Rvsdg::link_switch_node], or unlinked with [Rvsdg::unlink_switch_node].
    ///
    /// The first of the `value_inputs` must be the [SwitchNode]'s branch-selector value.
    ///
    /// Returns a [Node] handle for the newly created switch node.
    ///
    /// The branch regions for the switch node are added after the creation of the switch node
    /// by calling [add_switch_branch] with the [Node] handle returned from this operation.
    ///
    /// See the [SwitchNode] documentation for more information.
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

    /// Reorder's a [SwitchNode]'s branches based on the given `permutation`.
    ///
    /// The `permutation` must be a list of valid (in bounds) branch indices; it should not contain
    /// duplicates. Given a [SwitchNode] with three branches `A`, `B` and `C`, with the initial
    /// order being `A -> B -> C`, the permutation `[2, 0, 1]` would result in the branches being
    /// reordered to the order `C -> A -> B`.
    ///
    /// # Panics
    ///
    /// Panics if the `permutations` list contains duplicate indices. Panics if one of the indices
    /// in the `permutations` list is out of bounds.
    pub fn permute_switch_branches(&mut self, switch_node: Node, permutation: &[usize]) {
        let mut uniques = FxHashSet::default();

        uniques.extend(permutation.iter().copied());

        assert_eq!(
            uniques.len(),
            permutation.len(),
            "permutation indices must be unique"
        );

        let mut branches_new = Vec::with_capacity(permutation.len());

        let data = self.nodes[switch_node].kind.expect_switch_mut();

        for index in permutation {
            branches_new.push(data.branches[*index]);
        }

        data.branches = branches_new;
    }

    /// Adds a new "entry" [ValueInput] to a [SwitchNode].
    ///
    /// Returns the index of the new value-input.
    ///
    /// The new [ValueInput] will be appended onto the end of the [SwitchNode]'s list of inputs. If
    /// a [SwitchNode] had `N` prior value-inputs (including the branch-selector input), then the
    /// index of the new value-input will be `N`. Each of the [SwitchNode]'s branch regions will
    /// have a new region argument appended to their list of region arguments. This argument will
    /// have a matching type and will initially have an empty user-set. The argument's index will
    /// be `N-1`.
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

    /// Removes the value-input at the given `input` index from a [SwitchNode].
    ///
    /// The value-input at index `0` (the branch-selector input) cannot be removed. Also removes
    /// the region argument associated with the value-input in each of the [SwitchNode]'s branch
    /// regions. For all of these region arguments, their user-set (see [ValueOutput::users]) must
    /// be empty.
    ///
    /// This also removes the [ValueInput] from the user-set of its [ValueInput::origin].
    ///
    /// Unless the `input` is the [SwitchNode]s last input, then this will affect the indices
    /// associated with any value-inputs/arguments at greater indices, which will all be reduced by
    /// `1`. This needs to be taken into account when holding onto indices to identify
    /// switch-values. Internal to the [Rvsdg], this method will automatically adjust connections to
    /// value-input/arguments at greater indices.
    ///
    /// # Panics
    ///
    /// Panics of the `input` index is `0`. Panics if for any of the [SwitchNode]'s branch region
    /// the argument associated with the value-input does not have an empty user-set.
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
                    "cannot remove input {input} for switch node `{switch_node:?}`, because \
                    argument {arg} in branch {branch_index} still has users ({:?})",
                    &arguments[arg].users
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

    /// Adds a new [ValueOutput] to a [SwitchNode].
    ///
    /// Returns the index for the new [ValueOutput].
    ///
    /// The [ValueOutput] will be of the specified `ty` and will initially have an empty user-set.
    /// The new [ValueOutput] will be appended onto the end of the [SwitchNode]'s list of
    /// value-outputs. If a [SwitchNode] had `N` prior value-outputs, then the index of the new
    /// value-output will be `N`. Each of the [SwitchNode]'s branch regions will
    /// have a new region result appended to their list of region results. This result will have a
    /// matching type and will initially have a "placeholder" value-origin (see
    /// [ValueOrigin::placeholder]); you must replace this placeholder origin with a valid origin
    /// with [Rvsdg::reconnect_region_result].
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

    /// Removes the value-output at the given `output` index from a [SwitchNode].
    ///
    /// The value-output cannot be removed if its user-set is not empty (see [ValueOutput::users]).
    /// This also removes the associated region result for each of the [SwitchNode]'s branch
    /// regions.
    ///
    /// Unless the `output` is the [SwitchNode]'s last output, then this will affect the indices
    /// associated with any value-outputs/branch-results at greater indices, which will all be
    /// reduced by `1`. This needs to be taken into account when holding onto indices to identify
    /// switch-values. Internal to the [Rvsdg], this method will automatically adjust connections to
    /// value-outputs/results at greater indices.
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

            if !origin.is_placeholder() {
                self.remove_user(branch, origin, ValueUser::Result(index as u32));
            }

            self.regions[branch].value_results.remove(index);
            self.correct_region_result_connections(branch, index, -1);
        }
    }

    /// Links a [SwitchNode] into its region's state chain, in between the given `state_origin` and
    /// its original user.
    ///
    /// The [SwitchNode] will be the `state_origin`'s new user, and the [SwitchNode] will be the
    /// new [StateOrigin] for the `state_origin`'s original user.
    ///
    /// # Panics
    ///
    /// Panics if the [SwitchNode] is already linked into the state chain.
    pub fn link_switch_state(&mut self, switch_node: Node, state_origin: StateOrigin) {
        if self.nodes[switch_node].state().is_some() {
            panic!("switch node is already linked into the state chain")
        }

        self.nodes[switch_node].expect_switch_mut().state = Some(State {
            origin: state_origin,
            user: StateUser::Result,
        });

        let region = self.nodes[switch_node].region();

        self.link_state(region, switch_node, state_origin);
    }

    /// Unlinks the [SwitchNode] from its region's state chain.
    ///
    /// Will leave the [SwitchNode]'s [StateOrigin] directly connected to the [SwitchNode]'s
    /// [StateUser].
    ///
    /// # Panics
    ///
    /// Panics if the [SwitchNode] is not linked into its region's state chain.
    pub fn unlink_switch_state(&mut self, switch_node: Node) {
        if self.nodes[switch_node].state().is_none() {
            panic!("switch node is not linked into the state chain")
        }

        self.unlink_state(switch_node);
    }

    /// Adds a [LoopNode] to the given `region`.
    ///
    /// May supply a set of `value_inputs` that define the loop node's loop-values the node at
    /// creation (see the [Loop Values](LoopNode#loop-values)). Loop-values may also be added later
    /// with [Rvsdg::add_loop_input], or removed with [Rvsdg::remove_loop_input].
    ///
    /// May optionally supply a `state_origin`: if supplied, then the node will be inserted into the
    /// state chain between the origin and the origin's prior user; if `None` the switch node will
    /// not be part of the state chain. The [LoopNode] may also be linked into the state chain
    /// later with [Rvsdg::link_loop_node], or unlinked with [Rvsdg::unlink_loop_node].
    ///
    /// Returns a [Node] handle for the newly created loop node.
    ///
    /// See the [LoopNode] documentation for more information.
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

    /// Adds a new loop-value to a [SwitchNode].
    ///
    /// Since all parts of a loop-value (value-input, value-output, region-argument, and
    /// region-result) are tightly coupled, we use a [ValueInput] to fully define all loop-value
    /// parts.
    ///
    /// Returns the index of the new value-input and the new value-output.
    ///
    /// The new [ValueInput] will be appended onto the end of the [SwitchNode]'s list of inputs. If
    /// a [SwitchNode] had `N` prior value-inputs (including the branch-selector input), then the
    /// index of the new value-input will be `N`. Likewise, the new [ValueOutput] will be appended
    /// onto the end of the [SwitchNode]'s list of outputs at index `N`; the [ValueOutput] will
    /// initially have an empty user-set (see [ValueOutput::users]). A new argument will be appended
    /// to the loop region's arguments with index `N`. A new result will be appended to the loop
    /// region's result with index `N+1` (the first loop region argument is the reentry-condition,
    /// so all loop-value result indices are offset by `1`). The new loop region result will use
    /// the new loop region argument as its [ValueOrigin]; the new loop region argument will have
    /// the new loop region result as its only [ValueUser]. The loop region result may be
    /// reconnected to another value-output later with [Rvsdg::reconnect_region_result].
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

    /// Removes the loop-value for the given `input` index from a [LoopNode].
    ///
    /// Because for [LoopNode]s, value-inputs and value-outputs are tightly coupled, this also
    /// removes the value-output at the same index, and its associated result in the loop region.
    /// This value-output's user-set (see [ValueOutput::users]) must be empty. Because all
    /// value-inputs for a [LoopNode] flow to the loop region's arguments, this also removes the
    /// loop region argument at the same index. This argument's user-set must also be empty.
    ///
    /// This also removes the value-input from the user-set of its [ValueInput::origin].
    ///
    /// Unless the `input` is the [LoopNode]s last input, this will affect the indices associated
    /// with any value-inputs/value-outputs/argument/results at greater indices, which will all be
    /// reduced by `1`. This needs to be taken into account when holding onto indices to identify
    /// loop-values. Internal to the [Rvsdg], this method will automatically adjust connections to
    /// value-inputs/value-outputs/arguments/results at greater indices.
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

    /// Links a [LoopNode] into its region's state chain, in between the given `state_origin` and
    /// its original user.
    ///
    /// The [LoopNode] will be the `state_origin`'s new user, and the [LoopNode] will be the new
    /// [StateOrigin] for the `state_origin`'s original user.
    ///
    /// # Panics
    ///
    /// Panics if the [LoopNode] is already linked into the state chain.
    pub fn link_loop_state(&mut self, loop_node: Node, state_origin: StateOrigin) {
        if self.nodes[loop_node].state().is_some() {
            panic!("loop node is already linked into the state chain")
        }

        self.nodes[loop_node].expect_loop_mut().state = Some(State {
            origin: state_origin,
            user: StateUser::Result,
        });

        let region = self.nodes[loop_node].region();

        self.link_state(region, loop_node, state_origin);
    }

    /// Unlinks the [LoopNode] from its region's state chain.
    ///
    /// Will leave the [LoopNode]'s [StateOrigin] directly connected to the [LoopNode]'s
    /// [StateUser].
    ///
    /// # Panics
    ///
    /// Panics if the [LoopNode] is not linked into its region's state chain.
    pub fn unlink_loop_state(&mut self, loop_node: Node) {
        if self.nodes[loop_node].state().is_none() {
            panic!("loop node is not linked into the state chain")
        }

        self.unlink_state(loop_node);
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

    pub fn add_intrinsic_op<T>(
        &mut self,
        region: Region,
        intrinsic: T,
        value_inputs: impl IntoIterator<Item = ValueInput>,
        state_origin: Option<StateOrigin>,
    ) -> Node
    where
        T: Intrinsic,
        SimpleNode: From<IntrinsicNode<T>>,
    {
        let value_inputs: SmallVec<[ValueInput; 2]> = value_inputs.into_iter().collect();

        for input in &value_inputs {
            self.validate_node_value_input(region, input);
        }

        let output_ty = intrinsic
            .process_args(self.ty(), value_inputs.iter().map(|i| i.ty))
            .unwrap();
        let value_output = output_ty.map(|ty| ValueOutput::new(ty));
        let state = state_origin.map(|origin| {
            State {
                origin,
                user: StateUser::Result, // Temp value
            }
        });

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                IntrinsicNode {
                    intrinsic,
                    value_inputs,
                    value_output,
                    state,
                }
                .into(),
            ),
            region: Some(region),
        });

        if let Some(state_origin) = state_origin {
            self.link_state(region, node, state_origin);
        }

        self.regions[region].nodes.insert(node);
        self.connect_node_value_inputs(node);

        node
    }

    pub fn add_op_alloca(&mut self, region: Region, ty: Type) -> Node {
        self.add_intrinsic_op(region, intrinsic::OpAlloca { ty }, [], None)
    }

    pub fn add_op_load(
        &mut self,
        region: Region,
        ptr_input: ValueInput,
        state_origin: StateOrigin,
    ) -> Node {
        self.add_intrinsic_op(region, intrinsic::OpLoad, [ptr_input], Some(state_origin))
    }

    pub fn add_op_store(
        &mut self,
        region: Region,
        ptr_input: ValueInput,
        value_input: ValueInput,
        state_origin: StateOrigin,
    ) -> Node {
        self.add_intrinsic_op(
            region,
            intrinsic::OpStore,
            [ptr_input, value_input],
            Some(state_origin),
        )
    }

    pub fn add_op_extract_field(
        &mut self,
        region: Region,
        value_input: ValueInput,
        field_index: u32,
    ) -> Node {
        self.add_intrinsic_op(
            region,
            intrinsic::OpExtractField { field_index },
            [value_input],
            None,
        )
    }

    pub fn add_op_extract_element(
        &mut self,
        region: Region,
        value_input: ValueInput,
        index_input: ValueInput,
    ) -> Node {
        self.add_intrinsic_op(
            region,
            intrinsic::OpExtractElement,
            [value_input, index_input],
            None,
        )
    }

    pub fn add_op_field_ptr(
        &mut self,
        region: Region,
        ptr_input: ValueInput,
        field_index: u32,
    ) -> Node {
        self.add_intrinsic_op(
            region,
            intrinsic::OpFieldPtr { field_index },
            [ptr_input],
            None,
        )
    }

    pub fn add_op_element_ptr(
        &mut self,
        region: Region,
        ptr_input: ValueInput,
        index_input: ValueInput,
    ) -> Node {
        self.add_intrinsic_op(
            region,
            intrinsic::OpElementPtr,
            [ptr_input, index_input],
            None,
        )
    }

    pub fn add_op_discriminant_ptr(&mut self, region: Region, ptr_input: ValueInput) -> Node {
        self.add_intrinsic_op(region, intrinsic::OpDiscriminantPtr, [ptr_input], None)
    }

    pub fn add_op_variant_ptr(
        &mut self,
        region: Region,
        ptr_input: ValueInput,
        variant_index: u32,
    ) -> Node {
        self.add_intrinsic_op(
            region,
            intrinsic::OpVariantPtr { variant_index },
            [ptr_input],
            None,
        )
    }

    pub fn add_op_get_discriminant(
        &mut self,
        region: Region,
        ptr_input: ValueInput,
        state_origin: StateOrigin,
    ) -> Node {
        self.add_intrinsic_op(
            region,
            intrinsic::OpGetDiscriminant,
            [ptr_input],
            Some(state_origin),
        )
    }

    pub fn add_op_set_discriminant(
        &mut self,
        region: Region,
        ptr_input: ValueInput,
        variant_index: u32,
        state_origin: StateOrigin,
    ) -> Node {
        self.add_intrinsic_op(
            region,
            intrinsic::OpSetDiscriminant { variant_index },
            [ptr_input],
            Some(state_origin),
        )
    }

    pub fn add_op_offset_slice(
        &mut self,
        region: Region,
        ptr_input: ValueInput,
        offset_input: ValueInput,
    ) -> Node {
        self.add_intrinsic_op(
            region,
            intrinsic::OpOffsetSlice,
            [ptr_input, offset_input],
            None,
        )
    }

    pub fn add_op_get_ptr_offset(&mut self, region: Region, ptr_input: ValueInput) -> Node {
        self.add_intrinsic_op(region, intrinsic::OpGetSliceOffset, [ptr_input], None)
    }

    pub fn add_op_unary(
        &mut self,
        region: Region,
        operator: UnaryOperator,
        input: ValueInput,
    ) -> Node {
        self.add_intrinsic_op(region, intrinsic::OpUnary { operator }, [input], None)
    }

    pub fn add_op_binary(
        &mut self,
        region: Region,
        operator: BinaryOperator,
        lhs_input: ValueInput,
        rhs_input: ValueInput,
    ) -> Node {
        self.add_intrinsic_op(
            region,
            intrinsic::OpBinary { operator },
            [lhs_input, rhs_input],
            None,
        )
    }

    pub fn add_op_vector(
        &mut self,
        region: Region,
        vector_ty: ty::Vector,
        inputs: impl IntoIterator<Item = ValueInput>,
    ) -> Node {
        self.add_intrinsic_op(region, intrinsic::OpVector { ty: vector_ty }, inputs, None)
    }

    pub fn add_op_matrix(
        &mut self,
        region: Region,
        matrix_ty: ty::Matrix,
        inputs: impl IntoIterator<Item = ValueInput>,
    ) -> Node {
        self.add_intrinsic_op(region, intrinsic::OpMatrix { ty: matrix_ty }, inputs, None)
    }

    pub fn add_op_case_to_branch_selector(
        &mut self,
        region: Region,
        input: ValueInput,
        cases: impl IntoIterator<Item = u32>,
    ) -> Node {
        self.add_intrinsic_op(
            region,
            intrinsic::OpCaseToBranchSelector {
                cases: cases.into_iter().collect(),
            },
            [input],
            None,
        )
    }

    pub fn permute_op_case_to_branch_selector_cases(&mut self, node: Node, permutation: &[usize]) {
        let mut new_cases = Vec::with_capacity(permutation.len());

        let data = self.nodes[node].expect_op_case_to_branch_selector_mut();
        let cases = data.cases();

        for index in permutation {
            new_cases.push(cases[*index]);
        }

        data.intrinsic.cases = new_cases;
    }

    pub fn add_op_bool_to_branch_selector(&mut self, region: Region, input: ValueInput) -> Node {
        self.add_intrinsic_op(region, intrinsic::OpBoolToBranchSelector, [input], None)
    }

    pub fn add_op_u32_to_branch_selector(
        &mut self,
        region: Region,
        branch_count: u32,
        input: ValueInput,
    ) -> Node {
        self.add_intrinsic_op(
            region,
            intrinsic::OpU32ToBranchSelector { branch_count },
            [input],
            None,
        )
    }

    pub fn add_op_branch_selector_to_case(
        &mut self,
        region: Region,
        input: ValueInput,
        cases: impl IntoIterator<Item = u32>,
    ) -> Node {
        self.add_intrinsic_op(
            region,
            intrinsic::OpBranchSelectorToCase {
                cases: cases.into_iter().collect(),
            },
            [input],
            None,
        )
    }

    pub fn add_op_convert_to_u32(&mut self, region: Region, input: ValueInput) -> Node {
        self.add_intrinsic_op(region, intrinsic::OpConvertToU32, [input], None)
    }

    pub fn add_op_convert_to_i32(&mut self, region: Region, input: ValueInput) -> Node {
        self.add_intrinsic_op(region, intrinsic::OpConvertToI32, [input], None)
    }

    pub fn add_op_convert_to_f32(&mut self, region: Region, input: ValueInput) -> Node {
        self.add_intrinsic_op(region, intrinsic::OpConvertToF32, [input], None)
    }

    pub fn add_op_convert_to_bool(&mut self, region: Region, input: ValueInput) -> Node {
        self.add_intrinsic_op(region, intrinsic::OpConvertToBool, [input], None)
    }

    pub fn add_op_array_length(&mut self, region: Region, input: ValueInput) -> Node {
        self.add_intrinsic_op(region, intrinsic::OpArrayLength, [input], None)
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
            let _sig_arg_ty = sig.args[i].ty;
            let _value_input_ty = value_inputs[i + 1].ty;
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

    pub fn add_value_proxy(&mut self, region: Region, input: ValueInput) -> Node {
        self.validate_node_value_input(region, &input);

        let node = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                ValueProxy {
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
    /// For details on the purpose of proxy nodes, see the documentation for the [ValueProxy]
    /// struct.
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
    ) -> Node {
        let proxy = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                ValueProxy {
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

    pub fn proxy_origin_users(&mut self, region: Region, ty: Type, origin: ValueOrigin) -> Node {
        let proxy = self.nodes.insert(NodeData {
            kind: NodeKind::Simple(
                ValueProxy {
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

        output.users = thin_set![ValueUser::Input {
            consumer: proxy,
            input: 0
        }];

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

    pub fn dissolve_reaggregation(&mut self, reaggregation_node: Node) {
        let region = self.nodes[reaggregation_node].region();
        let data = self.nodes[reaggregation_node].expect_reaggregation();
        let original_origin = data.original().origin;
        let user_count = data.output().users.len();

        for i in (0..user_count).rev() {
            let user = self.nodes[reaggregation_node]
                .expect_reaggregation()
                .output()
                .users[i];

            self.reconnect_value_user(region, user, original_origin);
        }

        self.remove_node(reaggregation_node);
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
                    if !self.ty.can_coerce(a.ty, value_input.ty) {
                        panic!(
                            "cannot connect a node input of type `{}` to a region argument of type \
                            `{}`",
                            value_input.ty.to_string(self.ty()),
                            a.ty.to_string(self.ty()),
                        );
                    }
                } else {
                    panic!(
                        "tried to connect to region argument `{}`, but region only has {} \
                        arguments",
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
                    if !self.ty.can_coerce(output.ty, value_input.ty) {
                        panic!(
                            "cannot connect a node input of type `{:?}` to an output of type `{:?}",
                            value_input.ty.to_string(self.ty()),
                            output.ty.to_string(self.ty())
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
    use crate::ty::{TY_DUMMY, TY_PTR_U32};
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
