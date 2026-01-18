//! Coalesces multi-operation vector and matrix stores into a single store.
//!
//! A typical initialization pattern for a vector or matrix coming from the CFG format will look
//! something like this (in pseudocode):
//!
//! ```pseudocode
//! let vector_ptr = alloca::<vec2_f32>();
//!
//! let x_ptr = &vector_ptr.0;
//! store(x_ptr, 1.0);
//!
//! let y_ptr = &vector_ptr.1;
//! store(y_ptr, 2.0);
//! ```
//!
//! The alloca pointer will be projected to its individual elements, and then multiple store
//! operations will initialize each element value individually.
//!
//! This pass will coalesce this into a single store operation like so:
//!
//! ```pseudocode
//! let vector_ptr = alloca::<vec2_f32>();
//! let initial_value = vec2_f32(1.0, 2.0);
//!
//! store(vector_ptr, initial_value);
//! ```
//!
//! This transform prepares us for a memory-to-value-flow promotion pass; once coalesced, the
//! promotion pass can promote any load operations to simply resolve to the last stored coalesced
//! value.
//!
//! For example:
//!
//! ```pseudocode
//! let vector_ptr = alloca::<vec2_f32>();
//! let initial_value = vec2_f32(1.0, 2.0);
//!
//! store(vector_ptr, initial_value);
//!
//! let vector = load(vector_ptr);
//! ```
//!
//! Becomes:
//!
//! ```pseudocode
//! let vector = vec2_f32(1.0, 2.0);
//! ```
//!
//! The example above uses a vector value, but this pass will also coalesce matrix values.
//!
//! This pass allows a sequence of element stores to be interrupted by load operations that can be
//! proven load from a different root identifier. It does not currently allow the sequence to be
//! interrupted by any other store operations. This is conservative: a more advanced version of this
//! pass might be able to prove that interrupting loads/stores operate on disjoint memory in more
//! cases. However, this covers the main case we're interested in: coalescing the initialization of
//! vector and matrix alloca values, so that they may become candidates for memory-to-value-flow
//! promotion.
//!
//! Note that we do this for vectors and matrices, but not for other "aggregate" types, as only
//! vectors and matrices tend to be loaded from their alloca pointers "whole" to be used in
//! intrinsic operations. For structs and arrays we follow the opposite approach and instead split
//! the aggregate alloca into multiple scalar allocas whenever possible, see the scalar-replacement
//! pass. Said differently, we don't quite consider vectors and matrices to be "aggregates", as our
//! backends effectively treat them like scalar values in some important ways. In that sense, this
//! pass can also be thought of as a "scalarization" pass for vector and matrix values.

use arrayvec::ArrayVec;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::rvsdg::NodeKind::Simple;
use crate::rvsdg::SimpleNode::OpElementPtr;
use crate::rvsdg::visit::region_nodes::RegionNodesVisitor;
use crate::rvsdg::visit::reverse_value_flow::ReverseValueFlowVisitor;
use crate::rvsdg::{
    Connectivity, Node, NodeKind, Region, Rvsdg, SimpleNode, StateUser, ValueInput, ValueOrigin,
    visit,
};
use crate::ty::TypeKind;
use crate::{Function, Module, StorageBinding, WorkgroupBinding, ty};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Vector,
    Matrix,
}

fn try_resolve_const_index(rvsdg: &Rvsdg, origin: ValueOrigin) -> Option<u32> {
    if let ValueOrigin::Output {
        producer,
        output: 0,
    } = origin
    {
        if let NodeKind::Simple(SimpleNode::ConstU32(n)) = rvsdg[producer].kind() {
            return Some(n.value());
        }
    }

    None
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AggregateKind {
    Vector(ty::Vector),
    Matrix(ty::Matrix),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RootIdentifier {
    Unknown,
    Alloca(Node),
    StorageBinding(StorageBinding),
    WorkgroupBinding(WorkgroupBinding),
    Immutable,
}

impl RootIdentifier {
    fn is_disjoint(&self, other: RootIdentifier) -> bool {
        use RootIdentifier::*;
        match (*self, other) {
            // If we couldn't resolve either root identifier, then we cannot prove they are
            // disjoint and conservatively return `false`.
            (Unknown, _) | (_, Unknown) => false,
            (Alloca(a), Alloca(b)) => a != b,
            (StorageBinding(a), StorageBinding(b)) => a != b,
            (WorkgroupBinding(a), WorkgroupBinding(b)) => a != b,
            // If the root identifiers are not in the same address-space, then they are definitely
            // disjoint.
            _ => true,
        }
    }
}

#[derive(Debug)]
struct PointerAnalyzer {
    cache: FxHashMap<(Region, ValueOrigin), RootIdentifier>,
    root_identifier: RootIdentifier,
}

impl PointerAnalyzer {
    fn new() -> Self {
        Self {
            cache: FxHashMap::default(),
            root_identifier: RootIdentifier::Unknown,
        }
    }

    fn analyze(&mut self, rvsdg: &Rvsdg, region: Region, origin: ValueOrigin) -> RootIdentifier {
        self.root_identifier = RootIdentifier::Unknown;

        self.visit_value_origin(rvsdg, region, origin);

        self.root_identifier
    }

    fn aggregate_info(
        &mut self,
        rvsdg: &Rvsdg,
        region: Region,
        input: ValueInput,
    ) -> Option<AggregateInfo> {
        if let TypeKind::Ptr(pointee_ty) = *rvsdg.ty().kind(input.ty) {
            match &*rvsdg.ty().kind(pointee_ty) {
                TypeKind::Vector(v) => Some(AggregateInfo {
                    root_identifier: self.analyze(rvsdg, region, input.origin),
                    kind: AggregateKind::Vector(*v),
                    pointer_input: input,
                }),
                TypeKind::Matrix(m) => Some(AggregateInfo {
                    root_identifier: self.analyze(rvsdg, region, input.origin),
                    kind: AggregateKind::Matrix(*m),
                    pointer_input: input,
                }),
                _ => None,
            }
        } else {
            None
        }
    }
}

impl ReverseValueFlowVisitor for PointerAnalyzer {
    fn should_visit(&mut self, _region: Region, _origin: ValueOrigin) -> bool {
        true
    }

    fn visit_value_origin(&mut self, rvsdg: &Rvsdg, region: Region, origin: ValueOrigin) {
        if let Some(root_identifier) = self.cache.get(&(region, origin)) {
            self.root_identifier = *root_identifier;
        } else {
            visit::reverse_value_flow::visit_value_origin(self, rvsdg, region, origin);

            self.cache.insert((region, origin), self.root_identifier);
        }
    }

    fn visit_value_output(&mut self, rvsdg: &Rvsdg, node: Node, output: u32) {
        use NodeKind::*;
        use SimpleNode::*;

        match rvsdg[node].kind() {
            Simple(OpLoad(_)) | Simple(OpCall(_)) => {
                // Can't trace a pointer through memory operations or function calls.
                self.root_identifier = RootIdentifier::Unknown;
            }
            Switch(_) | Loop(_) => {
                // We don't try to resolve a (potentially) variable pointer.
                self.root_identifier = RootIdentifier::Unknown;
            }
            Simple(OpAlloca(_)) => {
                self.root_identifier = RootIdentifier::Alloca(node);
            }
            Simple(OpFieldPtr(_))
            | Simple(OpElementPtr(_))
            | Simple(OpVariantPtr(_))
            | Simple(OpOffsetSlice(_))
            | Simple(ConstPtr(_)) => {
                // Only visit the pointer input.
                visit::reverse_value_flow::visit_value_input(self, rvsdg, node, 0);
            }
            _ => unreachable!("node kind cannot output a pointer"),
        }
    }

    fn visit_region_argument(&mut self, rvsdg: &Rvsdg, region: Region, argument: u32) {
        let owner = rvsdg[region].owner();

        if let NodeKind::Function(fn_node) = rvsdg[owner].kind() {
            if let Some(input) = fn_node.dependencies().get(argument as usize)
                && let ValueOrigin::Output {
                    producer,
                    output: 0,
                } = input.origin
            {
                match rvsdg[producer].kind() {
                    NodeKind::StorageBinding(n) => {
                        self.root_identifier = RootIdentifier::StorageBinding(n.binding())
                    }
                    NodeKind::WorkgroupBinding(n) => {
                        self.root_identifier = RootIdentifier::WorkgroupBinding(n.binding())
                    }
                    NodeKind::UniformBinding(_) | NodeKind::Constant(_) => {
                        self.root_identifier = RootIdentifier::Immutable
                    }
                    _ => unreachable!("node kind cannot be a pointer root identifier"),
                }
            } else {
                self.root_identifier = RootIdentifier::Unknown;
            }
        } else {
            visit::reverse_value_flow::visit_region_argument(self, rvsdg, region, argument);
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct AggregateInfo {
    root_identifier: RootIdentifier,
    kind: AggregateKind,
    pointer_input: ValueInput,
}

impl AggregateInfo {
    fn matches_mode(&self, mode: Mode) -> bool {
        match (self.kind, mode) {
            (AggregateKind::Vector(_), Mode::Vector) | (AggregateKind::Matrix(_), Mode::Matrix) => {
                true
            }
            _ => false,
        }
    }

    fn size(&self) -> usize {
        match self.kind {
            AggregateKind::Vector(v) => v.size.to_usize(),
            AggregateKind::Matrix(m) => m.columns.to_usize(),
        }
    }
}

fn remove_element_store(rvsdg: &mut Rvsdg, node: Node) {
    let store = rvsdg[node].expect_op_store();
    let ValueOrigin::Output {
        producer: element_ptr_node,
        output: 0,
    } = store.ptr_input().origin
    else {
        panic!("the store node's pointer should originate from an element-ptr node");
    };

    rvsdg.remove_node(node);

    if rvsdg[element_ptr_node]
        .expect_op_element_ptr()
        .value_output()
        .users
        .is_empty()
    {
        rvsdg.remove_node(element_ptr_node);
    }
}

#[derive(Debug)]
struct State {
    aggregate: Option<AggregateInfo>,
    slots: [Option<Node>; 4],
    stale_store_ops: Vec<Node>,
}

impl State {
    fn new() -> Self {
        Self {
            aggregate: None,
            slots: [None; 4],
            stale_store_ops: Vec::new(),
        }
    }

    fn push(&mut self, aggregate: AggregateInfo, element: usize, store_node: Node) -> bool {
        if Some(aggregate) != self.aggregate {
            self.slots = [None; 4];
            self.aggregate = Some(aggregate);
        }

        // A value was already stored for this element, which makes this previous store redundant.
        // Store it as a stale store operation so that we can remove it when coalescing.
        if let Some(store_node) = self.slots[element].take() {
            self.stale_store_ops.push(store_node);
        }

        self.slots[element] = Some(store_node);

        let size = self.aggregate.map(|a| a.size()).unwrap();

        for i in 0..size {
            if self.slots[i].is_none() {
                return false;
            }
        }

        true
    }

    fn reset(&mut self) {
        self.aggregate = None;
        self.slots = [None; 4];
        self.stale_store_ops.clear();
    }

    fn coalesce(&mut self, rvsdg: &mut Rvsdg) -> Node {
        let aggregate = self
            .aggregate
            .take()
            .expect("can only coalesce if aggregate is set");
        let size = aggregate.size();

        let mut inputs: ArrayVec<ValueInput, 4> = ArrayVec::new();

        let first_store_node = self.slots[0].unwrap();
        let region = rvsdg[first_store_node].region();
        let state_origin = rvsdg[first_store_node].state().unwrap().origin;

        for i in 0..size {
            let store_node = self.slots[i].take().unwrap();
            let store = rvsdg[store_node].expect_op_store();
            let value_input = store.value_input().clone();

            inputs.push(value_input);

            remove_element_store(rvsdg, store_node);
        }

        debug_assert_eq!(&self.slots, &[None; 4], "should have reset all slots");

        for stale_store_node in self.stale_store_ops.drain(..) {
            remove_element_store(rvsdg, stale_store_node);
        }

        let (ty, coalesced) = match aggregate.kind {
            AggregateKind::Vector(ty) => {
                let coalesced = rvsdg.add_op_vector(region, ty, inputs);
                let ty = rvsdg.ty().register(TypeKind::Vector(ty));

                (ty, coalesced)
            }
            AggregateKind::Matrix(ty) => {
                let coalesced = rvsdg.add_op_matrix(region, ty, inputs);
                let ty = rvsdg.ty().register(TypeKind::Matrix(ty));

                (ty, coalesced)
            }
        };

        rvsdg.add_op_store(
            region,
            aggregate.pointer_input,
            ValueInput::output(ty, coalesced, 0),
            state_origin,
        )
    }
}

struct Coalescer<'a> {
    pointer_analyzer: &'a mut PointerAnalyzer,
    mode: Mode,
    state: State,
    did_coalesce: bool,
}

impl<'a> Coalescer<'a> {
    fn new(pointer_analyzer: &'a mut PointerAnalyzer, mode: Mode) -> Self {
        Self {
            pointer_analyzer,
            mode,
            state: State::new(),
            did_coalesce: false,
        }
    }

    fn coalesce_in_region(&mut self, rvsdg: &mut Rvsdg, region: Region) -> bool {
        self.did_coalesce = false;

        self.visit_region(rvsdg, region);

        self.did_coalesce
    }

    fn visit_region(&mut self, rvsdg: &mut Rvsdg, region: Region) {
        let user = rvsdg[region].state_argument();

        self.visit_state_user(rvsdg, region, *user);
    }

    fn visit_node(&mut self, rvsdg: &mut Rvsdg, node: Node) {
        use NodeKind::*;
        use SimpleNode::*;

        let node = match rvsdg[node].kind() {
            Simple(OpStore(_)) => {
                let coalesce = self.record_op_store(rvsdg, node, self.mode);

                if coalesce {
                    self.did_coalesce = true;

                    self.state.coalesce(rvsdg)
                } else {
                    node
                }
            }
            Simple(OpLoad(op)) => {
                // Reset when a load happens that may overlap with the store sequence. Only do the
                // analysis when the state is currently tracking a store sequence.
                if let Some(aggregate) = &self.state.aggregate {
                    let region = rvsdg[node].region();
                    let root_identifier =
                        self.pointer_analyzer
                            .analyze(rvsdg, region, op.ptr_input().origin);

                    if !root_identifier.is_disjoint(aggregate.root_identifier) {
                        self.state.reset();
                    }
                }

                node
            }
            Simple(_) => {
                // Reset when any other stateful node interrupts the store sequence.
                self.state.reset();

                node
            }
            Switch(n) => {
                // Control-flow causes a reset.
                self.state.reset();

                let branch_count = n.branches().len();

                for i in 0..branch_count {
                    let branch = rvsdg[node].expect_switch().branches()[i];

                    self.visit_region(rvsdg, branch);
                }

                node
            }
            Loop(n) => {
                // Control-flow causes a reset.
                self.state.reset();

                let loop_region = n.loop_region();

                self.visit_region(rvsdg, loop_region);

                node
            }
            _ => unreachable!("node kind cannot be part of a state chain"),
        };

        if let Some(state) = rvsdg[node].state() {
            let region = rvsdg[node].region();

            self.visit_state_user(rvsdg, region, state.user);
        }
    }

    fn visit_state_user(&mut self, rvsdg: &mut Rvsdg, region: Region, user: StateUser) {
        if StateUser::Result == user {
            // Reset at region boundaries.
            self.state.reset();
        }

        match user {
            StateUser::Result => {
                let owner = rvsdg[region].owner();

                match rvsdg[owner].kind() {
                    NodeKind::Switch(_) | NodeKind::Loop(_) => {
                        let owner_region = rvsdg[owner].region();

                        if let Some(state) = rvsdg[owner].state() {
                            self.visit_state_user(rvsdg, owner_region, state.user);
                        }
                    }
                    NodeKind::Function(_) => {
                        // Do nothing, we're done...
                    }
                    _ => unreachable!("node kind cannot own a region"),
                }
            }
            StateUser::Node(node) => self.visit_node(rvsdg, node),
        }
    }

    fn record_op_store(&mut self, rvsdg: &Rvsdg, op_store_node: Node, mode: Mode) -> bool {
        let region = rvsdg[op_store_node].region();
        let op_store = rvsdg[op_store_node].expect_op_store();
        let ptr_origin = op_store.ptr_input().origin;

        if let ValueOrigin::Output {
            producer,
            output: 0,
        } = ptr_origin
            && let Simple(OpElementPtr(op)) = rvsdg[producer].kind()
            && let Some(aggregate) =
                self.pointer_analyzer
                    .aggregate_info(rvsdg, region, *op.ptr_input())
            && aggregate.matches_mode(mode)
            && let Some(element) = try_resolve_const_index(rvsdg, op.index_input().origin)
        {
            self.state.push(aggregate, element as usize, op_store_node)
        } else {
            self.state.reset();

            false
        }
    }
}

pub fn region_coalesce_store_ops(rvsdg: &mut Rvsdg, region: Region) -> bool {
    let mut did_coalesce = false;

    let mut pointer_analyzer = PointerAnalyzer::new();

    let mut vector_coalescer = Coalescer::new(&mut pointer_analyzer, Mode::Vector);

    did_coalesce |= vector_coalescer.coalesce_in_region(rvsdg, region);

    let mut matrix_coalescer = Coalescer::new(&mut pointer_analyzer, Mode::Matrix);

    did_coalesce |= matrix_coalescer.coalesce_in_region(rvsdg, region);

    did_coalesce
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::rvsdg::StateOrigin;
    use crate::ty::{TY_BOOL, TY_DUMMY, TY_F32, TY_PREDICATE, TY_U32, TY_VEC2_F32, TY_VEC3_F32};
    use crate::{FnArg, FnSig, Symbol};

    #[test]
    fn test_valid_sequence() {
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
                args: vec![],
                ret_ty: Some(TY_VEC2_F32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_VEC3_F32));
        let element_ptr_ty = module.ty.register(TypeKind::Ptr(TY_F32));

        let alloca_node = rvsdg.add_op_alloca(region, TY_VEC3_F32);

        let index_0_node = rvsdg.add_const_u32(region, 0);
        let ptr_0_node = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            ValueInput::output(TY_U32, index_0_node, 0),
        );
        let value_0_node = rvsdg.add_const_f32(region, 1.0);
        let store_0_node = rvsdg.add_op_store(
            region,
            ValueInput::output(element_ptr_ty, ptr_0_node, 0),
            ValueInput::output(TY_F32, value_0_node, 0),
            StateOrigin::Argument,
        );

        let index_1_node = rvsdg.add_const_u32(region, 1);
        let ptr_1_node = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            ValueInput::output(TY_U32, index_1_node, 0),
        );
        let value_1_node = rvsdg.add_const_f32(region, 2.0);
        let store_1_node = rvsdg.add_op_store(
            region,
            ValueInput::output(element_ptr_ty, ptr_1_node, 0),
            ValueInput::output(TY_F32, value_1_node, 0),
            StateOrigin::Node(store_0_node),
        );

        let index_2_node = rvsdg.add_const_u32(region, 2);
        let ptr_2_node = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            ValueInput::output(TY_U32, index_2_node, 0),
        );
        let value_2_node = rvsdg.add_const_f32(region, 3.0);
        let store_2_node = rvsdg.add_op_store(
            region,
            ValueInput::output(element_ptr_ty, ptr_2_node, 0),
            ValueInput::output(TY_F32, value_2_node, 0),
            StateOrigin::Node(store_1_node),
        );

        let load_node = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            StateOrigin::Node(store_2_node),
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_node,
                output: 0,
            },
        );

        region_coalesce_store_ops(&mut rvsdg, region);

        let StateUser::Node(coalesced_store_node) = *rvsdg[region].state_argument() else {
            panic!("the state argument should connect to a node");
        };

        let coalesced_store_data = rvsdg[coalesced_store_node].expect_op_store();

        assert_eq!(
            coalesced_store_data.ptr_input().origin,
            ValueOrigin::Output {
                producer: alloca_node,
                output: 0,
            },
            "the coalesced store node should store directly to the alloca"
        );

        let ValueOrigin::Output {
            producer: coalesced_value_node,
            output: 0,
        } = coalesced_store_data.value_input().origin
        else {
            panic!("the coalesced store node's value input should connect to a node")
        };

        let coalesced_value_data = rvsdg[coalesced_value_node].expect_op_vector();

        assert_eq!(
            coalesced_value_data.value_inputs(),
            &[
                ValueInput::output(TY_F32, value_0_node, 0),
                ValueInput::output(TY_F32, value_1_node, 0),
                ValueInput::output(TY_F32, value_2_node, 0),
            ],
            "the coalesced value node should aggregate all three values of the original stores"
        );

        assert_eq!(
            rvsdg[load_node].state().unwrap().origin,
            StateOrigin::Node(coalesced_store_node),
            "the load node should be in the state chain after the coalesced store node"
        );

        assert!(
            !rvsdg.is_live_node(store_0_node),
            "the original store op for the first element value should be dead"
        );
        assert!(
            !rvsdg.is_live_node(store_1_node),
            "the original store op for the second element value should be dead"
        );
        assert!(
            !rvsdg.is_live_node(store_2_node),
            "the original store op for the third element value should be dead"
        );
    }

    #[test]
    fn test_incomplete_sequence() {
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
                args: vec![],
                ret_ty: Some(TY_VEC2_F32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_VEC3_F32));
        let element_ptr_ty = module.ty.register(TypeKind::Ptr(TY_F32));

        let alloca_node = rvsdg.add_op_alloca(region, TY_VEC3_F32);

        let index_0_node = rvsdg.add_const_u32(region, 0);
        let ptr_0_node = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            ValueInput::output(TY_U32, index_0_node, 0),
        );
        let value_0_node = rvsdg.add_const_f32(region, 1.0);
        let store_0_node = rvsdg.add_op_store(
            region,
            ValueInput::output(element_ptr_ty, ptr_0_node, 0),
            ValueInput::output(TY_F32, value_0_node, 0),
            StateOrigin::Argument,
        );

        let index_2_node = rvsdg.add_const_u32(region, 2);
        let ptr_2_node = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            ValueInput::output(TY_U32, index_2_node, 0),
        );
        let value_2_node = rvsdg.add_const_f32(region, 3.0);
        let store_2_node = rvsdg.add_op_store(
            region,
            ValueInput::output(element_ptr_ty, ptr_2_node, 0),
            ValueInput::output(TY_F32, value_2_node, 0),
            StateOrigin::Node(store_0_node),
        );

        let load_node = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            StateOrigin::Node(store_2_node),
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_node,
                output: 0,
            },
        );

        region_coalesce_store_ops(&mut rvsdg, region);

        // The state chain should be unaltered.
        assert_eq!(
            rvsdg[region].state_argument(),
            &StateUser::Node(store_0_node),
            "the state user of the function region argument should be still be the store-0 node"
        );
        assert_eq!(
            rvsdg[store_0_node].state().unwrap().user,
            StateUser::Node(store_2_node),
            "the state user of the store-0 node should be still be the store-2 node"
        );
        assert_eq!(
            rvsdg[store_2_node].state().unwrap().user,
            StateUser::Node(load_node),
            "the state user of the store-2 node should be still be the load node"
        );
        assert_eq!(
            rvsdg[load_node].state().unwrap().user,
            StateUser::Result,
            "the state user of the load node should be still be the region's state result"
        );
    }

    #[test]
    fn test_sequence_interrupted_by_load_from_same_alloca() {
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
                args: vec![],
                ret_ty: Some(TY_VEC2_F32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_VEC2_F32));
        let element_ptr_ty = module.ty.register(TypeKind::Ptr(TY_F32));

        let alloca_node = rvsdg.add_op_alloca(region, TY_VEC2_F32);

        let index_0_node = rvsdg.add_const_u32(region, 0);
        let ptr_0_node = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            ValueInput::output(TY_U32, index_0_node, 0),
        );
        let value_0_node = rvsdg.add_const_f32(region, 1.0);
        let store_0_node = rvsdg.add_op_store(
            region,
            ValueInput::output(element_ptr_ty, ptr_0_node, 0),
            ValueInput::output(TY_F32, value_0_node, 0),
            StateOrigin::Argument,
        );

        let interrupting_load_node = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            StateOrigin::Node(store_0_node),
        );

        let index_1_node = rvsdg.add_const_u32(region, 1);
        let ptr_1_node = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            ValueInput::output(TY_U32, index_1_node, 0),
        );
        let value_1_node = rvsdg.add_const_f32(region, 3.0);
        let store_1_node = rvsdg.add_op_store(
            region,
            ValueInput::output(element_ptr_ty, ptr_1_node, 0),
            ValueInput::output(TY_F32, value_1_node, 0),
            StateOrigin::Node(interrupting_load_node),
        );

        let load_node = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            StateOrigin::Node(store_1_node),
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_node,
                output: 0,
            },
        );

        region_coalesce_store_ops(&mut rvsdg, region);

        // The state chain should be unaltered.
        assert_eq!(
            rvsdg[region].state_argument(),
            &StateUser::Node(store_0_node),
            "the state user of the function region argument should be still be the store-0 node"
        );
        assert_eq!(
            rvsdg[store_0_node].state().unwrap().user,
            StateUser::Node(interrupting_load_node),
            "the state user of the store-0 node should be still be the interrupting-load node"
        );
        assert_eq!(
            rvsdg[interrupting_load_node].state().unwrap().user,
            StateUser::Node(store_1_node),
            "the state user of the interrupting-load node should be still be the store-1 node"
        );
        assert_eq!(
            rvsdg[store_1_node].state().unwrap().user,
            StateUser::Node(load_node),
            "the state user of the store-1 node should be still be the load node"
        );
        assert_eq!(
            rvsdg[load_node].state().unwrap().user,
            StateUser::Result,
            "the state user of the load node should be still be the region's state result"
        );
    }

    #[test]
    fn test_sequence_interrupted_by_load_from_different_alloca() {
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
                args: vec![],
                ret_ty: Some(TY_VEC2_F32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_VEC2_F32));
        let element_ptr_ty = module.ty.register(TypeKind::Ptr(TY_F32));

        let alloca_node = rvsdg.add_op_alloca(region, TY_VEC2_F32);
        let other_alloca_node = rvsdg.add_op_alloca(region, TY_VEC2_F32);

        let index_0_node = rvsdg.add_const_u32(region, 0);
        let ptr_0_node = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            ValueInput::output(TY_U32, index_0_node, 0),
        );
        let value_0_node = rvsdg.add_const_f32(region, 1.0);
        let store_0_node = rvsdg.add_op_store(
            region,
            ValueInput::output(element_ptr_ty, ptr_0_node, 0),
            ValueInput::output(TY_F32, value_0_node, 0),
            StateOrigin::Argument,
        );

        let interrupting_load_node = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, other_alloca_node, 0),
            StateOrigin::Node(store_0_node),
        );

        let index_1_node = rvsdg.add_const_u32(region, 1);
        let ptr_1_node = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            ValueInput::output(TY_U32, index_1_node, 0),
        );
        let value_1_node = rvsdg.add_const_f32(region, 3.0);
        let store_1_node = rvsdg.add_op_store(
            region,
            ValueInput::output(element_ptr_ty, ptr_1_node, 0),
            ValueInput::output(TY_F32, value_1_node, 0),
            StateOrigin::Node(interrupting_load_node),
        );

        let load_node = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            StateOrigin::Node(store_1_node),
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_node,
                output: 0,
            },
        );

        region_coalesce_store_ops(&mut rvsdg, region);

        let StateUser::Node(coalesced_store_node) = *rvsdg[region].state_argument() else {
            panic!("the state argument should connect to a node");
        };

        let coalesced_store_data = rvsdg[coalesced_store_node].expect_op_store();

        assert_eq!(
            coalesced_store_data.ptr_input().origin,
            ValueOrigin::Output {
                producer: alloca_node,
                output: 0,
            },
            "the coalesced store node should store directly to the alloca"
        );

        let ValueOrigin::Output {
            producer: coalesced_value_node,
            output: 0,
        } = coalesced_store_data.value_input().origin
        else {
            panic!("the coalesced store node's value input should connect to a node")
        };

        let coalesced_value_data = rvsdg[coalesced_value_node].expect_op_vector();

        assert_eq!(
            coalesced_value_data.value_inputs(),
            &[
                ValueInput::output(TY_F32, value_0_node, 0),
                ValueInput::output(TY_F32, value_1_node, 0),
            ],
            "the coalesced value node should aggregate all three values of the original stores"
        );

        assert_eq!(
            rvsdg[interrupting_load_node].state().unwrap().origin,
            StateOrigin::Node(coalesced_store_node),
            "the interrupting-load node should be in the state chain after the coalesced-store node"
        );

        assert_eq!(
            rvsdg[load_node].state().unwrap().origin,
            StateOrigin::Node(interrupting_load_node),
            "the load node should be in the state chain after the interrupting-load node"
        );

        assert!(
            !rvsdg.is_live_node(store_0_node),
            "the original store op for the first element value should be dead"
        );
        assert!(
            !rvsdg.is_live_node(store_1_node),
            "the original store op for the second element value should be dead"
        );
    }

    #[test]
    fn test_sequence_interrupted_by_store() {
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
                args: vec![],
                ret_ty: Some(TY_VEC2_F32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_VEC2_F32));
        let element_ptr_ty = module.ty.register(TypeKind::Ptr(TY_F32));

        let alloca_0_node = rvsdg.add_op_alloca(region, TY_VEC2_F32);
        let alloca_1_node = rvsdg.add_op_alloca(region, TY_VEC2_F32);

        let index_0_node = rvsdg.add_const_u32(region, 0);
        let ptr_0_node = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(ptr_ty, alloca_0_node, 0),
            ValueInput::output(TY_U32, index_0_node, 0),
        );
        let value_0_node = rvsdg.add_const_f32(region, 1.0);
        let store_0_node = rvsdg.add_op_store(
            region,
            ValueInput::output(element_ptr_ty, ptr_0_node, 0),
            ValueInput::output(TY_F32, value_0_node, 0),
            StateOrigin::Argument,
        );

        let interrupting_index_node = rvsdg.add_const_u32(region, 0);
        let interrupting_ptr_node = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(ptr_ty, alloca_1_node, 0),
            ValueInput::output(TY_U32, interrupting_index_node, 0),
        );
        let interrupting_value_node = rvsdg.add_const_f32(region, 1.0);
        let interrupting_store_node = rvsdg.add_op_store(
            region,
            ValueInput::output(element_ptr_ty, interrupting_ptr_node, 0),
            ValueInput::output(TY_F32, interrupting_value_node, 0),
            StateOrigin::Node(store_0_node),
        );

        let index_1_node = rvsdg.add_const_u32(region, 1);
        let ptr_1_node = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(ptr_ty, alloca_0_node, 0),
            ValueInput::output(TY_U32, index_1_node, 0),
        );
        let value_1_node = rvsdg.add_const_f32(region, 3.0);
        let store_1_node = rvsdg.add_op_store(
            region,
            ValueInput::output(element_ptr_ty, ptr_1_node, 0),
            ValueInput::output(TY_F32, value_1_node, 0),
            StateOrigin::Node(interrupting_store_node),
        );

        let load_node = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, alloca_0_node, 0),
            StateOrigin::Node(store_1_node),
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_node,
                output: 0,
            },
        );

        region_coalesce_store_ops(&mut rvsdg, region);

        // The state chain should be unaltered.
        assert_eq!(
            rvsdg[region].state_argument(),
            &StateUser::Node(store_0_node),
            "the state user of the function region argument should be still be the store-0 node"
        );
        assert_eq!(
            rvsdg[store_0_node].state().unwrap().user,
            StateUser::Node(interrupting_store_node),
            "the state user of the store-0 node should be still be the interrupting-store node"
        );
        assert_eq!(
            rvsdg[interrupting_store_node].state().unwrap().user,
            StateUser::Node(store_1_node),
            "the state user of the interrupting-store node should be still be the store-1 node"
        );
        assert_eq!(
            rvsdg[store_1_node].state().unwrap().user,
            StateUser::Node(load_node),
            "the state user of the store-1 node should be still be the load node"
        );
        assert_eq!(
            rvsdg[load_node].state().unwrap().user,
            StateUser::Result,
            "the state user of the load node should be still be the region's state result"
        );
    }

    #[test]
    fn test_sequence_interrupted_by_switch() {
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
                    ty: TY_BOOL,
                    shader_io_binding: None,
                }],
                ret_ty: Some(TY_VEC2_F32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_VEC2_F32));
        let element_ptr_ty = module.ty.register(TypeKind::Ptr(TY_F32));

        let alloca_node = rvsdg.add_op_alloca(region, TY_VEC2_F32);

        let index_0_node = rvsdg.add_const_u32(region, 0);
        let ptr_0_node = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            ValueInput::output(TY_U32, index_0_node, 0),
        );
        let value_0_node = rvsdg.add_const_f32(region, 1.0);
        let store_0_node = rvsdg.add_op_store(
            region,
            ValueInput::output(element_ptr_ty, ptr_0_node, 0),
            ValueInput::output(TY_F32, value_0_node, 0),
            StateOrigin::Argument,
        );

        let predicate_node =
            rvsdg.add_op_bool_to_branch_selector(region, ValueInput::argument(TY_BOOL, 0));
        let switch_node = rvsdg.add_switch(
            region,
            vec![ValueInput::output(TY_PREDICATE, predicate_node, 0)],
            vec![],
            Some(StateOrigin::Node(store_0_node)),
        );

        rvsdg.add_switch_branch(switch_node);
        rvsdg.add_switch_branch(switch_node);

        let index_1_node = rvsdg.add_const_u32(region, 1);
        let ptr_1_node = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            ValueInput::output(TY_U32, index_1_node, 0),
        );
        let value_1_node = rvsdg.add_const_f32(region, 3.0);
        let store_1_node = rvsdg.add_op_store(
            region,
            ValueInput::output(element_ptr_ty, ptr_1_node, 0),
            ValueInput::output(TY_F32, value_1_node, 0),
            StateOrigin::Node(switch_node),
        );

        let load_node = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            StateOrigin::Node(store_1_node),
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_node,
                output: 0,
            },
        );

        region_coalesce_store_ops(&mut rvsdg, region);

        // The state chain should be unaltered.
        assert_eq!(
            rvsdg[region].state_argument(),
            &StateUser::Node(store_0_node),
            "the state user of the function region argument should be still be the store-0 node"
        );
        assert_eq!(
            rvsdg[store_0_node].state().unwrap().user,
            StateUser::Node(switch_node),
            "the state user of the store-0 node should be still be the interrupting-store node"
        );
        assert_eq!(
            rvsdg[switch_node].state().unwrap().user,
            StateUser::Node(store_1_node),
            "the state user of the interrupting-store node should be still be the store-1 node"
        );
        assert_eq!(
            rvsdg[store_1_node].state().unwrap().user,
            StateUser::Node(load_node),
            "the state user of the store-1 node should be still be the load node"
        );
        assert_eq!(
            rvsdg[load_node].state().unwrap().user,
            StateUser::Result,
            "the state user of the load node should be still be the region's state result"
        );
    }

    #[test]
    fn test_valid_sequence_after_interrupt() {
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
                args: vec![],
                ret_ty: Some(TY_VEC2_F32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_VEC2_F32));
        let element_ptr_ty = module.ty.register(TypeKind::Ptr(TY_F32));

        let alloca_node = rvsdg.add_op_alloca(region, TY_VEC2_F32);

        let index_0_before_node = rvsdg.add_const_u32(region, 0);
        let ptr_0_before_node = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            ValueInput::output(TY_U32, index_0_before_node, 0),
        );
        let value_0_before_node = rvsdg.add_const_f32(region, 1.0);
        let store_0_before_node = rvsdg.add_op_store(
            region,
            ValueInput::output(element_ptr_ty, ptr_0_before_node, 0),
            ValueInput::output(TY_F32, value_0_before_node, 0),
            StateOrigin::Argument,
        );

        let interrupting_load_node = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            StateOrigin::Node(store_0_before_node),
        );

        let index_0_after_node = rvsdg.add_const_u32(region, 0);
        let ptr_0_after_node = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            ValueInput::output(TY_U32, index_0_after_node, 0),
        );
        let value_0_after_node = rvsdg.add_const_f32(region, 1.0);
        let store_0_after_node = rvsdg.add_op_store(
            region,
            ValueInput::output(element_ptr_ty, ptr_0_after_node, 0),
            ValueInput::output(TY_F32, value_0_after_node, 0),
            StateOrigin::Node(interrupting_load_node),
        );

        let index_1_node = rvsdg.add_const_u32(region, 1);
        let ptr_1_node = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            ValueInput::output(TY_U32, index_1_node, 0),
        );
        let value_1_node = rvsdg.add_const_f32(region, 3.0);
        let store_1_node = rvsdg.add_op_store(
            region,
            ValueInput::output(element_ptr_ty, ptr_1_node, 0),
            ValueInput::output(TY_F32, value_1_node, 0),
            StateOrigin::Node(store_0_after_node),
        );

        let load_node = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            StateOrigin::Node(store_1_node),
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_node,
                output: 0,
            },
        );

        region_coalesce_store_ops(&mut rvsdg, region);

        assert_eq!(
            rvsdg[region].state_argument(),
            &StateUser::Node(store_0_before_node),
            "the region's function argument should still connect to the store op that occurs \
            before the interrupt"
        );
        assert_eq!(
            rvsdg[store_0_before_node].state().unwrap().user,
            StateUser::Node(interrupting_load_node),
            "the interrupting-load should still occur after the first store op"
        );

        let StateUser::Node(coalesced_store_node) =
            rvsdg[interrupting_load_node].state().unwrap().user
        else {
            panic!("the interrupting-load node should connect to a node");
        };

        let coalesced_store_data = rvsdg[coalesced_store_node].expect_op_store();

        assert_eq!(
            coalesced_store_data.ptr_input().origin,
            ValueOrigin::Output {
                producer: alloca_node,
                output: 0,
            },
            "the coalesced store node should store directly to the alloca"
        );

        let ValueOrigin::Output {
            producer: coalesced_value_node,
            output: 0,
        } = coalesced_store_data.value_input().origin
        else {
            panic!("the coalesced store node's value input should connect to a node")
        };

        let coalesced_value_data = rvsdg[coalesced_value_node].expect_op_vector();

        assert_eq!(
            coalesced_value_data.value_inputs(),
            &[
                ValueInput::output(TY_F32, value_0_after_node, 0),
                ValueInput::output(TY_F32, value_1_node, 0),
            ],
            "the coalesced value node should use the values stored after the interrupt"
        );

        assert_eq!(
            rvsdg[load_node].state().unwrap().origin,
            StateOrigin::Node(coalesced_store_node),
            "the load node should be in the state chain after the coalesced store node"
        );

        assert!(
            rvsdg.is_live_node(store_0_before_node),
            "the store op that occurs before the interrupt should still be alive"
        );
        assert!(
            !rvsdg.is_live_node(store_0_after_node),
            "the store op for the first element after the interrupt should be dead"
        );
        assert!(
            !rvsdg.is_live_node(store_1_node),
            "the store op for the second element after the interrupt should be dead"
        );
    }

    #[test]
    fn test_valid_sequence_inside_switch() {
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
                    ty: TY_BOOL,
                    shader_io_binding: None,
                }],
                ret_ty: Some(TY_VEC2_F32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_VEC2_F32));
        let element_ptr_ty = module.ty.register(TypeKind::Ptr(TY_F32));

        let alloca_node = rvsdg.add_op_alloca(region, TY_VEC2_F32);

        let index_0_before_node = rvsdg.add_const_u32(region, 0);
        let ptr_0_before_node = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            ValueInput::output(TY_U32, index_0_before_node, 0),
        );
        let value_0_before_node = rvsdg.add_const_f32(region, 1.0);
        let store_0_before_node = rvsdg.add_op_store(
            region,
            ValueInput::output(element_ptr_ty, ptr_0_before_node, 0),
            ValueInput::output(TY_F32, value_0_before_node, 0),
            StateOrigin::Argument,
        );

        let predicate_node =
            rvsdg.add_op_bool_to_branch_selector(region, ValueInput::argument(TY_BOOL, 0));
        let switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::output(TY_PREDICATE, predicate_node, 0),
                ValueInput::output(ptr_ty, alloca_node, 0),
            ],
            vec![],
            Some(StateOrigin::Node(store_0_before_node)),
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);
        rvsdg.add_switch_branch(switch_node);

        let index_0_node = rvsdg.add_const_u32(branch_0, 0);
        let ptr_0_node = rvsdg.add_op_element_ptr(
            branch_0,
            ValueInput::argument(ptr_ty, 0),
            ValueInput::output(TY_U32, index_0_node, 0),
        );
        let value_0_node = rvsdg.add_const_f32(branch_0, 1.0);
        let store_0_node = rvsdg.add_op_store(
            branch_0,
            ValueInput::output(element_ptr_ty, ptr_0_node, 0),
            ValueInput::output(TY_F32, value_0_node, 0),
            StateOrigin::Argument,
        );

        let index_1_node = rvsdg.add_const_u32(branch_0, 1);
        let ptr_1_node = rvsdg.add_op_element_ptr(
            branch_0,
            ValueInput::argument(ptr_ty, 0),
            ValueInput::output(TY_U32, index_1_node, 0),
        );
        let value_1_node = rvsdg.add_const_f32(branch_0, 3.0);
        let store_1_node = rvsdg.add_op_store(
            branch_0,
            ValueInput::output(element_ptr_ty, ptr_1_node, 0),
            ValueInput::output(TY_F32, value_1_node, 0),
            StateOrigin::Node(store_0_node),
        );

        let load_node = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, alloca_node, 0),
            StateOrigin::Node(switch_node),
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_node,
                output: 0,
            },
        );

        region_coalesce_store_ops(&mut rvsdg, region);

        assert_eq!(
            rvsdg[region].state_argument(),
            &StateUser::Node(store_0_before_node),
            "the region's function argument should still connect to the store op that occurs \
            before the interrupt"
        );
        assert_eq!(
            rvsdg[store_0_before_node].state().unwrap().user,
            StateUser::Node(switch_node),
            "the switch should still occur after the first store op"
        );

        let StateUser::Node(coalesced_store_node) = *rvsdg[branch_0].state_argument() else {
            panic!("the first branch's state argument should connect to a node");
        };

        let coalesced_store_data = rvsdg[coalesced_store_node].expect_op_store();

        assert_eq!(
            coalesced_store_data.ptr_input().origin,
            ValueOrigin::Argument(0),
            "the coalesced store node should store to first branch's first argument"
        );

        let ValueOrigin::Output {
            producer: coalesced_value_node,
            output: 0,
        } = coalesced_store_data.value_input().origin
        else {
            panic!("the coalesced store node's value input should connect to a node")
        };

        let coalesced_value_data = rvsdg[coalesced_value_node].expect_op_vector();

        assert_eq!(
            coalesced_value_data.value_inputs(),
            &[
                ValueInput::output(TY_F32, value_0_node, 0),
                ValueInput::output(TY_F32, value_1_node, 0),
            ],
            "the coalesced value node should use the values stored inside the switch"
        );

        assert_eq!(
            rvsdg[load_node].state().unwrap().origin,
            StateOrigin::Node(switch_node),
            "the load node should be in the state chain after the switch node"
        );

        assert!(
            rvsdg.is_live_node(store_0_before_node),
            "the store op that occurs before the switch still be alive"
        );
        assert!(
            !rvsdg.is_live_node(store_0_node),
            "the store op for the first element inside the switch should be dead"
        );
        assert!(
            !rvsdg.is_live_node(store_1_node),
            "the store op for the second element inside the switch should be dead"
        );
    }
}
