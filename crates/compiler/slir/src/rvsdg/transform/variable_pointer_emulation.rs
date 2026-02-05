use indexmap::IndexSet;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::rvsdg::{
    Connectivity, Node, NodeKind, Region, Rvsdg, SimpleNode, StateOrigin, ValueInput, ValueOrigin,
    ValueOutput,
};
use crate::ty::{TY_PREDICATE, TY_U32, Type, TypeKind};

/// A variable pointer emulation program description.
#[derive(Clone, Debug)]
struct PointerEmulationInfo {
    pointer_ty: Type,
    emulation_root: EmulationTreeNode,
}

/// A node in a pointer emulation program description.
#[derive(Clone, Debug)]
enum EmulationTreeNode {
    Branching(BranchingNode),
    Leaf(LeafNode),
}

impl EmulationTreeNode {
    fn assign_child_inputs(&mut self) {
        if let EmulationTreeNode::Branching(node) = self {
            node.assign_child_inputs();
        }
    }
}

impl From<BranchingNode> for EmulationTreeNode {
    fn from(branching: BranchingNode) -> Self {
        EmulationTreeNode::Branching(branching)
    }
}

impl From<LeafNode> for EmulationTreeNode {
    fn from(leaf: LeafNode) -> Self {
        EmulationTreeNode::Leaf(leaf)
    }
}

/// Represents a branching node in a variable pointer emulation program description.
///
/// This will translate to a [Switch] node when constructing the emulation program.
#[derive(Clone, Debug)]
struct BranchingNode {
    /// The [ValueOrigin] for the branch selector predicate, relative to the root emulation node.
    ///
    /// For the root emulation switch node, this directly identifies the branch selector input. For
    /// a switch node nested deeper in the pointer emulation program, this needs to be resolved
    /// against the parent [BranchingNode]'s [child_inputs].
    branch_selector: ValueOrigin,

    /// The branches of the node.
    ///
    /// Each branch will translate to its own [Region] in the [Switch] node when constructing the
    /// emulation program.
    branches: Vec<EmulationTreeNode>,

    /// The complete set of [ValueOrigin]s used by the child nodes (branches) of this branching
    /// node.
    ///
    /// These map to inputs/arguments for the [Switch] node when constructing the emulation program
    /// in index order.
    ///
    /// Child nodes look to map a [ValueOrigin] that is relative to the emulation root region, to
    /// the index of the argument at which their direct parent node makes the value available to
    /// its child nodes.
    child_inputs: IndexSet<ValueOrigin>,
}

impl BranchingNode {
    fn assign_child_inputs(&mut self) {
        ChildInputAssigner.visit_branching_node(self)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Access {
    Field(u32),
    StaticElement(u32),
    DynamicElement(ValueOrigin),
}

impl Access {
    pub fn from_element_origin(rvsdg: &Rvsdg, origin: ValueOrigin) -> Self {
        match origin {
            ValueOrigin::Argument(_) => Access::DynamicElement(origin),
            ValueOrigin::Output { producer, .. } => {
                if let NodeKind::Simple(SimpleNode::ConstU32(n)) = rvsdg[producer].kind() {
                    Access::StaticElement(n.value())
                } else {
                    Access::DynamicElement(origin)
                }
            }
        }
    }
}

/// Represents a leaf node in a variable pointer emulation program description.
///
/// A leaf node resolves to a pointer without any further branching.
#[derive(Clone, Debug)]
struct LeafNode {
    /// A pointer to the root identifier of the pointer that this leaf emulates.
    root_pointer: ValueOrigin,

    /// A complete access chain that will refine the [root_identifier] into the pointer that this
    /// leave emulates.
    access_chain: Vec<Access>,
}

impl EmulationTreeNode {
    fn visit_leaves_mut(&mut self, visitor: &mut impl FnMut(&mut LeafNode)) {
        match self {
            EmulationTreeNode::Branching(node) => {
                for branch in &mut node.branches {
                    branch.visit_leaves_mut(visitor);
                }
            }
            EmulationTreeNode::Leaf(node) => visitor(node),
        }
    }

    fn visit_value_origins_mut_internal(
        &mut self,
        visitor: &mut impl FnMut(&mut ValueOrigin),
        visit_non_ptr_origins: bool,
        visit_ptr_origins: bool,
    ) {
        match self {
            EmulationTreeNode::Branching(node) => {
                if visit_non_ptr_origins {
                    visitor(&mut node.branch_selector);
                }

                for branch in &mut node.branches {
                    branch.visit_value_origins_mut_internal(
                        visitor,
                        visit_non_ptr_origins,
                        visit_ptr_origins,
                    );
                }
            }
            EmulationTreeNode::Leaf(node) => {
                if visit_ptr_origins {
                    visitor(&mut node.root_pointer);
                }

                if visit_non_ptr_origins {
                    for access in &mut node.access_chain {
                        if let Access::DynamicElement(origin) = access {
                            visitor(origin);
                        }
                    }
                }
            }
        }
    }

    fn visit_value_origins_mut(&mut self, visitor: &mut impl FnMut(&mut ValueOrigin)) {
        self.visit_value_origins_mut_internal(visitor, true, true)
    }

    fn visit_non_ptr_origins_mut(&mut self, visitor: &mut impl FnMut(&mut ValueOrigin)) {
        self.visit_value_origins_mut_internal(visitor, true, false)
    }

    fn visit_ptr_origins_mut(&mut self, visitor: &mut impl FnMut(&mut ValueOrigin)) {
        self.visit_value_origins_mut_internal(visitor, false, true)
    }
}

/// Stores information about the extra values that have been added to a switch node to make inner
/// values available outside the node for pointer emulation.
///
/// See [SwitchEmulationRegistry] for details.
struct SwitchEmulationValues {
    start: u32,
    end: u32,
    branches_end: Vec<u32>,
}

/// Stores information about the extra values that have been added to switch nodes to make inner
/// values available outside the node for pointer emulation.
///
/// Not every branch of a switch node will want to make the same values available. However, we want
/// to maximally reuse output values because every extra output value will likely end up costing a
/// registry. So we track the start and end of output values that have been added so far for the
/// node as a whole, and we'll also track for each branch which results/outputs are actually being
/// used to pass out values. Note that this reuse of outputs works because all extra emulation
/// outputs will all be of type `u32`.
struct SwitchEmulationRegistry {
    switch_emulation_values: FxHashMap<Node, SwitchEmulationValues>,
}

impl SwitchEmulationRegistry {
    fn new() -> Self {
        SwitchEmulationRegistry {
            switch_emulation_values: Default::default(),
        }
    }

    /// Returns the index of the next available output for the given [branch].
    ///
    /// This may add a new output for the [switch_node] if no output is currently available. In that
    /// case, for each branch other than the requested [branch], the corresponding new region result
    /// will be connected to a placeholder `0u32` constant value.
    fn next_output(&mut self, rvsdg: &mut Rvsdg, switch_node: Node, branch: u32) -> u32 {
        let emulation_values = self
            .switch_emulation_values
            .entry(switch_node)
            .or_insert_with(|| {
                let branch_count = rvsdg[switch_node].expect_switch().branches().len();
                let start = rvsdg[switch_node].value_outputs().len() as u32;
                let end = start;
                let branches_end = vec![start; branch_count];

                SwitchEmulationValues {
                    start,
                    end,
                    branches_end,
                }
            });

        let branch_end = emulation_values.branches_end[branch as usize];

        if branch_end >= emulation_values.end {
            rvsdg.add_switch_output(switch_node, TY_U32);

            // Connect a `0` as a fallback value for all branches except for the current branch
            // (since for the current branch we should expect that the reason an output is
            // requested is to connect something to it). Other branches that request an output
            // later and may get an output that already has such a placeholder `0` attached to it
            // and may reconnect the output to some other input. This should leave the placeholder
            // `0` node unused and ready to be eliminated by a dead-connectible-elimination pass.
            let branch_count = rvsdg[switch_node].expect_switch().branches().len();

            for i in 0..branch_count {
                let current_branch = rvsdg[switch_node].expect_switch().branches()[i];

                if i != branch as usize {
                    let placeholder = rvsdg.add_const_u32(current_branch, 0);

                    rvsdg.reconnect_region_result(
                        current_branch,
                        emulation_values.end,
                        ValueOrigin::Output {
                            producer: placeholder,
                            output: 0,
                        },
                    );
                }
            }

            emulation_values.end += 1;
        }

        emulation_values.branches_end[branch as usize] += 1;

        branch_end
    }

    fn clear(&mut self) {
        self.switch_emulation_values.clear();
    }
}

/// Propagates input requirements for pointer emulation from the bottom up, and for each branching
/// node it visits, assigns the combined set of inputs to which its children require access.
///
/// These input sets help construct the sets of switch node inputs when constructing a node graph
/// to emulate a load or store operation on a variable pointer that requires emulation.
struct ChildInputAssigner;

impl ChildInputAssigner {
    fn visit_branching_node(&self, node: &mut BranchingNode) {
        let mut child_inputs = IndexSet::new();

        for branch in &mut node.branches {
            match branch {
                EmulationTreeNode::Branching(node) => {
                    self.visit_branching_node(node);

                    child_inputs.insert(node.branch_selector);
                    child_inputs.extend(node.child_inputs.iter());
                }
                EmulationTreeNode::Leaf(node) => self.visit_leaf_node(node, &mut child_inputs),
            }
        }

        node.child_inputs = child_inputs;
    }

    fn visit_leaf_node(&self, node: &LeafNode, parent_value_inputs: &mut IndexSet<ValueOrigin>) {
        parent_value_inputs.insert(node.root_pointer);

        for access in &node.access_chain {
            if let &Access::DynamicElement(index_origin) = access {
                parent_value_inputs.insert(index_origin);
            }
        }
    }
}

pub struct EmulationContext {
    pointer_emulation_info: FxHashMap<(Region, ValueOrigin), PointerEmulationInfo>,
    switch_emulation_registry: SwitchEmulationRegistry,
}

impl EmulationContext {
    pub fn new() -> Self {
        EmulationContext {
            pointer_emulation_info: FxHashMap::default(),
            switch_emulation_registry: SwitchEmulationRegistry::new(),
        }
    }

    pub fn emulate_op_load(&mut self, rvsdg: &mut Rvsdg, op_load: Node) {
        let region = rvsdg[op_load].region();
        let data = rvsdg[op_load].expect_op_load();
        let pointer_ty = data.ptr_input().ty;
        let output_ty = data.value_output().ty;
        let state_origin = data.state().unwrap().origin;
        let ptr_origin = data.ptr_input().origin;
        let info = self.resolve_pointer_emulation_info(rvsdg, region, ptr_origin);

        let gen_op_load = |rvsdg: &mut Rvsdg,
                           region: Region,
                           ptr_input: ValueInput,
                           _additional_inputs: &[ValueInput],
                           state_origin: StateOrigin| {
            rvsdg.add_op_load(region, ptr_input, state_origin)
        };

        let mut emulator = Emulator {
            rvsdg,
            outer_region: region,
            state_origin,
            pointer_ty,
            output_ty: Some(output_ty),
            op_gen: gen_op_load,
            additional_values: &[],
        };

        let emulated = emulator.emulate(&info.emulation_root);

        // Now that we've constructed a node to emulate the original OpLoad, reconnect all users of
        // its value output to the emulated node's output and remove the original OpLoad.

        let user_count = rvsdg[op_load].expect_op_load().value_output().users.len();

        for i in (0..user_count).rev() {
            let user = rvsdg[op_load].expect_op_load().value_output().users[i];

            rvsdg.reconnect_value_user(
                region,
                user,
                ValueOrigin::Output {
                    producer: emulated,
                    output: 0,
                },
            );
        }

        rvsdg.remove_node(op_load);
    }

    pub fn emulate_op_store(&mut self, rvsdg: &mut Rvsdg, op_store: Node) {
        let outer_region = rvsdg[op_store].region();
        let data = rvsdg[op_store].expect_op_store();
        let pointer_ty = data.ptr_input().ty;
        let state_origin = data.state().unwrap().origin;
        let ptr_origin = data.ptr_input().origin;
        let value_input = *data.value_input();
        let info = self.resolve_pointer_emulation_info(rvsdg, outer_region, ptr_origin);

        let gen_op_store = |rvsdg: &mut Rvsdg,
                            region: Region,
                            ptr_input: ValueInput,
                            additional_inputs: &[ValueInput],
                            state_origin: StateOrigin| {
            rvsdg.add_op_store(region, ptr_input, additional_inputs[0], state_origin)
        };

        let mut emulator = Emulator {
            rvsdg,
            outer_region,
            state_origin,
            pointer_ty,
            output_ty: None,
            op_gen: gen_op_store,
            additional_values: &[value_input],
        };

        emulator.emulate(&info.emulation_root);

        rvsdg.remove_node(op_store);
    }

    pub fn clear(&mut self) {
        self.pointer_emulation_info.clear();
        self.switch_emulation_registry.clear();
    }

    fn resolve_pointer_emulation_info(
        &mut self,
        rvsdg: &mut Rvsdg,
        region: Region,
        pointer_origin: ValueOrigin,
    ) -> &PointerEmulationInfo {
        if self
            .pointer_emulation_info
            .contains_key(&(region, pointer_origin))
        {
            self.pointer_emulation_info
                .get(&(region, pointer_origin))
                .unwrap()
        } else {
            self.create_pointer_emulation_info(rvsdg, region, pointer_origin)
        }
    }

    fn create_pointer_emulation_info(
        &mut self,
        rvsdg: &mut Rvsdg,
        region: Region,
        pointer_origin: ValueOrigin,
    ) -> &PointerEmulationInfo {
        use NodeKind::*;
        use SimpleNode::*;

        let info = match pointer_origin {
            ValueOrigin::Argument(arg) => self.create_argument_info(rvsdg, region, arg),
            ValueOrigin::Output { producer, output } => match rvsdg[producer].kind() {
                Switch(_) => self.create_switch_output_info(rvsdg, producer, output),
                Loop(_) => self.create_loop_output_info(rvsdg, producer, output),
                Simple(OpFieldPtr(_)) => self.create_field_ptr_info(rvsdg, producer),
                Simple(OpElementPtr(_)) => self.create_element_ptr_info(rvsdg, producer),
                Simple(OpAlloca(_)) => self.create_alloca_info(rvsdg, producer),
                Simple(ConstPtr(_)) => self.create_const_ptr_info(rvsdg, producer),
                Simple(ConstFallback(_)) => self.create_fallback_info(rvsdg, producer),
                Simple(OpOffsetSlice(_)) => self.create_add_offset_slice_info(rvsdg, producer),
                Simple(OpLoad(_)) => panic!(
                    "cannot emulate a pointer for which the access chain information cannot be \
                    tracked through value-flow"
                ),
                _ => unreachable!("node kind cannot output a value of a pointer type"),
            },
        };

        self.pointer_emulation_info
            .insert((region, pointer_origin), info);
        self.pointer_emulation_info
            .get(&(region, pointer_origin))
            .unwrap()
    }

    fn create_argument_info(
        &mut self,
        rvsdg: &mut Rvsdg,
        region: Region,
        argument: u32,
    ) -> PointerEmulationInfo {
        let owner = rvsdg[region].owner();

        match rvsdg[owner].kind() {
            NodeKind::Switch(_) => self.create_switch_argument_info(rvsdg, owner, argument),
            NodeKind::Loop(_) => self.create_loop_argument_info(rvsdg, owner, argument),
            NodeKind::Function(_) => {
                let pointer_ty = rvsdg[region].value_arguments()[argument as usize].ty;

                PointerEmulationInfo {
                    pointer_ty,
                    emulation_root: LeafNode {
                        root_pointer: ValueOrigin::Argument(argument),
                        access_chain: vec![],
                    }
                    .into(),
                }
            }
            _ => unreachable!("node kind cannot own a region"),
        }
    }

    fn create_switch_argument_info(
        &mut self,
        rvsdg: &mut Rvsdg,
        switch_node: Node,
        argument: u32,
    ) -> PointerEmulationInfo {
        fn resolve_inner_origin(
            rvsdg: &mut Rvsdg,
            switch_node: Node,
            outer_origin: ValueOrigin,
        ) -> ValueOrigin {
            for (i, input) in rvsdg[switch_node].value_inputs()[1..].iter().enumerate() {
                if input.origin == outer_origin {
                    return ValueOrigin::Argument(i as u32);
                }
            }

            let region = rvsdg[switch_node].region();
            let input_count = rvsdg[switch_node].value_inputs().len() as u32;
            let argument_count = input_count - 1;
            let ty = rvsdg.value_origin_ty(region, outer_origin);

            rvsdg.add_switch_input(
                switch_node,
                ValueInput {
                    ty,
                    origin: outer_origin,
                },
            );

            ValueOrigin::Argument(argument_count)
        }

        let input = argument + 1;
        let outer_region = rvsdg[switch_node].region();
        let switch_data = rvsdg[switch_node].expect_switch();
        let origin = switch_data.value_inputs()[input as usize].origin;
        let mut info = self
            .resolve_pointer_emulation_info(rvsdg, outer_region, origin)
            .clone();

        info.emulation_root
            .visit_value_origins_mut(&mut |outer_origin| {
                *outer_origin = resolve_inner_origin(rvsdg, switch_node, *outer_origin);
            });
        info.emulation_root.assign_child_inputs();

        info
    }

    fn create_loop_argument_info(
        &mut self,
        rvsdg: &mut Rvsdg,
        loop_node: Node,
        argument: u32,
    ) -> PointerEmulationInfo {
        fn resolve_inner_origin(
            rvsdg: &mut Rvsdg,
            loop_node: Node,
            outer_origin: ValueOrigin,
        ) -> ValueOrigin {
            for (i, input) in rvsdg[loop_node].value_inputs().iter().enumerate() {
                if input.origin == outer_origin {
                    return ValueOrigin::Argument(i as u32);
                }
            }

            let region = rvsdg[loop_node].region();
            let loop_region = rvsdg[loop_node].expect_loop().loop_region();

            // The index for the new input/argument that we'll add is the count before adding.
            let argument = rvsdg[loop_node].value_inputs().len() as u32;
            // The corresponding result index is 1 greater than the argument index.
            let result = argument + 1;
            let ty = rvsdg.value_origin_ty(region, outer_origin);

            rvsdg.add_loop_input(
                loop_node,
                ValueInput {
                    ty,
                    origin: outer_origin,
                },
            );

            // Connect the new result to the new argument to ensure the value is available on all
            // loop iterations, not just the first iteration
            rvsdg.reconnect_region_result(loop_region, result, ValueOrigin::Argument(argument));

            ValueOrigin::Argument(argument)
        }

        let input = argument;
        let outer_region = rvsdg[loop_node].region();
        let loop_data = rvsdg[loop_node].expect_switch();
        let origin = loop_data.value_inputs()[input as usize].origin;
        let mut info = self
            .resolve_pointer_emulation_info(rvsdg, outer_region, origin)
            .clone();

        info.emulation_root
            .visit_value_origins_mut(&mut |outer_origin| {
                *outer_origin = resolve_inner_origin(rvsdg, loop_node, *outer_origin);
            });
        info.emulation_root.assign_child_inputs();

        info
    }

    fn create_switch_output_info(
        &mut self,
        rvsdg: &mut Rvsdg,
        switch_node: Node,
        output: u32,
    ) -> PointerEmulationInfo {
        // Helper function to find or create a non-pointer value origin outside the switch node that
        // represents the same value as the given `inner_origin` inside the switch node
        fn resolve_non_ptr_outer_origin(
            rvsdg: &mut Rvsdg,
            switch_emulation_registry: &mut SwitchEmulationRegistry,
            switch_node: Node,
            branch_index: u32,
            branch: Region,
            inner_origin: ValueOrigin,
        ) -> ValueOrigin {
            // First, check if the inner value origin connects directly to a branch argument and, if
            // so, use the origin of the corresponding input as the outer origin.
            if let ValueOrigin::Argument(argument) = inner_origin {
                // The argument's corresponding input is at an index 1 greater than the argument's
                // index, as the first input to a switch node is the branch selector predicate.
                let input = argument + 1;

                return rvsdg[switch_node].value_inputs()[input as usize].origin;
            }

            // Next, check if the inner value origin is already connected to a region result and, if
            // so, use the corresponding output as the outer origin
            for (i, input) in rvsdg[branch].value_results().iter().enumerate() {
                if input.origin == inner_origin {
                    return ValueOrigin::Output {
                        producer: switch_node,
                        output: i as u32,
                    };
                }
            }

            // The value origin is not yet available outside the switch. To make the value available
            // outside the switch node, we ask the registry to either provide a currently unused
            // emulation result for the current branch or to create a new switch output and provide
            // its corresponding result.
            let output = switch_emulation_registry.next_output(rvsdg, switch_node, branch_index);

            rvsdg.reconnect_region_result(branch, output, inner_origin);

            ValueOrigin::Output {
                producer: switch_node,
                output,
            }
        }

        // Helper function to find or create a pointer value origin outside the switch node that
        // represents the same value as the given `inner_origin` inside the switch node
        fn resolve_ptr_outer_origin(
            rvsdg: &mut Rvsdg,
            switch_node: Node,
            inner_origin: ValueOrigin,
        ) -> ValueOrigin {
            // Originating pointer values must already be available outside the switch node, as
            // an originating pointer value may not outlive its region. (We don't currently enforce
            // this in SLIR, but all pointer values derive from Rust references, and the Rust
            // borrow checker will enforce this for us.) Therefore, the `inner_origin` must be an
            // argument origin, and we'll resolve the outer origin as the origin of the argument's
            // corresponding input.

            let ValueOrigin::Argument(argument) = inner_origin else {
                panic!("originating pointer value must originate outside the switch region")
            };

            // The argument's corresponding input is at an index 1 greater than the argument's
            // index, as the first input to a switch node is the branch selector predicate.
            let input = argument + 1;

            rvsdg[switch_node].value_inputs()[input as usize].origin
        }

        let switch_data = rvsdg[switch_node].expect_switch();
        let pointer_ty = switch_data.value_outputs()[output as usize].ty;
        let branch_count = switch_data.branches().len();
        let branch_selector = switch_data.predicate().origin;
        let mut branches = Vec::with_capacity(branch_count);

        for i in 0..branch_count {
            let branch = rvsdg[switch_node].expect_switch().branches()[i];
            let origin = rvsdg[branch].value_results()[output as usize].origin;
            let mut info = self
                .resolve_pointer_emulation_info(rvsdg, branch, origin)
                .clone();

            info.emulation_root
                .visit_non_ptr_origins_mut(&mut |inner_origin| {
                    *inner_origin = resolve_non_ptr_outer_origin(
                        rvsdg,
                        &mut self.switch_emulation_registry,
                        switch_node,
                        i as u32,
                        branch,
                        *inner_origin,
                    );
                });

            info.emulation_root
                .visit_ptr_origins_mut(&mut |inner_origin| {
                    *inner_origin = resolve_ptr_outer_origin(rvsdg, switch_node, *inner_origin);
                });

            branches.push(info.emulation_root)
        }

        let mut branching_node = BranchingNode {
            branch_selector,
            branches,
            child_inputs: Default::default(),
        };

        branching_node.assign_child_inputs();

        PointerEmulationInfo {
            pointer_ty,
            emulation_root: branching_node.into(),
        }
    }

    fn create_loop_output_info(
        &mut self,
        rvsdg: &mut Rvsdg,
        loop_node: Node,
        output: u32,
    ) -> PointerEmulationInfo {
        // Helper function to find or create a non-pointer value origin outside the loop node that
        // represents the same value as the given `inner_origin` inside the loop node
        fn resolve_non_ptr_outer_origin(
            rvsdg: &mut Rvsdg,
            loop_node: Node,
            loop_region: Region,
            inner_origin: ValueOrigin,
        ) -> ValueOrigin {
            // First check if the inner value origin is already connected to a region result and, if
            // available, use the corresponding output as the outer origin
            for (i, input) in rvsdg[loop_region].value_results()[1..].iter().enumerate() {
                if input.origin == inner_origin {
                    return ValueOrigin::Output {
                        producer: loop_node,
                        output: i as u32,
                    };
                }
            }

            // The value origin is not available outside the loop, add a new loop value to make it
            // available. We'll have to provide an input value (that will be unused). All
            // non-pointer values in pointer emulation are `u32` values, so we'll add a constant
            // `u32` node with value `0`.

            let outer_region = rvsdg[loop_node].region();
            let input_value = rvsdg.add_const_u32(outer_region, 0);
            let output = rvsdg[loop_node].value_inputs().len() as u32;
            let result = output + 1;

            rvsdg.add_loop_input(loop_node, ValueInput::output(TY_U32, input_value, 0));
            rvsdg.reconnect_region_result(loop_region, result, inner_origin);

            ValueOrigin::Output {
                producer: loop_node,
                output,
            }
        }

        // Helper function to find or create a pointer value origin outside the loop node that
        // represents the same value as the given `inner_origin` inside the loop node
        fn resolve_ptr_outer_origin(
            rvsdg: &mut Rvsdg,
            loop_node: Node,
            inner_origin: ValueOrigin,
        ) -> ValueOrigin {
            // Originating pointer values must already be available outside the loop region, as
            // an originating pointer value may not outlive its region. (We don't currently enforce
            // this in SLIR, but all pointer values derive from Rust references, and the Rust
            // borrow checker will enforce this for us.) Therefore, the `inner_origin` must be an
            // argument origin, and we'll resolve the outer origin as the origin of the argument's
            // corresponding input.

            let ValueOrigin::Argument(argument) = inner_origin else {
                panic!("originating pointer value must originate outside the loop region")
            };

            rvsdg[loop_node].value_inputs()[argument as usize].origin
        }

        let loop_region = rvsdg[loop_node].expect_loop().loop_region();
        let result = output + 1;
        let origin = rvsdg[loop_region].value_results()[result as usize].origin;
        let mut info = self
            .resolve_pointer_emulation_info(rvsdg, loop_region, origin)
            .clone();

        info.emulation_root
            .visit_non_ptr_origins_mut(&mut |inner_origin| {
                *inner_origin =
                    resolve_non_ptr_outer_origin(rvsdg, loop_node, loop_region, *inner_origin);
            });

        info.emulation_root
            .visit_ptr_origins_mut(&mut |inner_origin| {
                *inner_origin = resolve_ptr_outer_origin(rvsdg, loop_node, *inner_origin);
            });
        info.emulation_root.assign_child_inputs();

        info
    }

    fn create_field_ptr_info(
        &mut self,
        rvsdg: &mut Rvsdg,
        op_field_ptr: Node,
    ) -> PointerEmulationInfo {
        let region = rvsdg[op_field_ptr].region();
        let field_ptr = rvsdg[op_field_ptr].expect_op_field_ptr();
        let pointer_ty = field_ptr.value_output().ty;
        let ptr_origin = field_ptr.ptr_input().origin;
        let access = Access::Field(field_ptr.field_index());

        let mut info = self
            .resolve_pointer_emulation_info(rvsdg, region, ptr_origin)
            .clone();

        info.emulation_root.visit_leaves_mut(&mut |leaf| {
            leaf.access_chain.push(access);
        });
        info.pointer_ty = pointer_ty;

        info
    }

    fn create_element_ptr_info(
        &mut self,
        rvsdg: &mut Rvsdg,
        op_element_ptr: Node,
    ) -> PointerEmulationInfo {
        let region = rvsdg[op_element_ptr].region();
        let element_ptr = rvsdg[op_element_ptr].expect_op_element_ptr();
        let pointer_ty = element_ptr.value_output().ty;
        let ptr_origin = element_ptr.ptr_input().origin;
        let index_origin = element_ptr.index_input().origin;
        let access = Access::from_element_origin(rvsdg, index_origin);

        let mut info = self
            .resolve_pointer_emulation_info(rvsdg, region, ptr_origin)
            .clone();

        info.emulation_root.visit_leaves_mut(&mut |leaf| {
            leaf.access_chain.push(access);
        });
        info.pointer_ty = pointer_ty;

        info
    }

    fn create_alloca_info(&mut self, rvsdg: &Rvsdg, op_alloca: Node) -> PointerEmulationInfo {
        let emulation_root = LeafNode {
            root_pointer: ValueOrigin::Output {
                producer: op_alloca,
                output: 0,
            },
            access_chain: vec![],
        };

        PointerEmulationInfo {
            pointer_ty: rvsdg[op_alloca].expect_op_alloca().value_output().ty,
            emulation_root: emulation_root.into(),
        }
    }

    fn create_const_ptr_info(&mut self, rvsdg: &Rvsdg, const_ptr: Node) -> PointerEmulationInfo {
        let emulation_root = LeafNode {
            root_pointer: ValueOrigin::Output {
                producer: const_ptr,
                output: 0,
            },
            access_chain: vec![],
        };

        PointerEmulationInfo {
            pointer_ty: rvsdg[const_ptr].value_outputs()[0].ty,
            emulation_root: emulation_root.into(),
        }
    }

    fn create_fallback_info(
        &mut self,
        rvsdg: &Rvsdg,
        const_fallback: Node,
    ) -> PointerEmulationInfo {
        let emulation_root = LeafNode {
            root_pointer: ValueOrigin::Output {
                producer: const_fallback,
                output: 0,
            },
            access_chain: vec![],
        };

        PointerEmulationInfo {
            pointer_ty: rvsdg[const_fallback].expect_const_fallback().ty(),
            emulation_root: emulation_root.into(),
        }
    }

    fn create_add_offset_slice_info(
        &mut self,
        rvsdg: &mut Rvsdg,
        op_offset_slice: Node,
    ) -> PointerEmulationInfo {
        let region = rvsdg[op_offset_slice].region();
        let offset_slice = rvsdg[op_offset_slice].expect_op_offset_slice();
        let ptr_origin = offset_slice.ptr_input().origin;

        self.resolve_pointer_emulation_info(rvsdg, region, ptr_origin)
            .clone()
    }
}

struct Emulator<'a, F> {
    rvsdg: &'a mut Rvsdg,

    /// The region that hosts the "load" or "store" node that we're emulating.
    outer_region: Region,

    /// The state origin of the "load" or "store" node that we're emulating.
    state_origin: StateOrigin,

    /// The type of the pointer input of the "load" or "store" node that we're emulating.
    pointer_ty: Type,

    /// If we're emulating a "load" node, the output type of the load operation.
    output_ty: Option<Type>,

    /// A function that will generate a new "load" or "store" node operating on an emulated pointer.
    ///
    /// The first argument passed to this function is a `mut` reference to the [Rvsdg].
    ///
    /// The second argument is the region into which the generated node must be added.
    ///
    /// The third argument is a [ValueInput] that is to be used as the "pointer" input for the
    /// operation.
    ///
    /// The fourth argument is a slice of additional input values that are to be used as the
    /// remaining value inputs for the generated node. For example, if we're emulating an [OpStore],
    /// this slice will have a length of one and represents the value to be stored.
    ///
    /// The fifth argument is the [StateOrigin] that the generated node must use.
    op_gen: F,

    /// A slice containing the original additional value inputs for the "load" or "store" node we're
    /// emulating.
    ///
    /// For example, if we're emulating an [OpLoad], then this slice should be empty. If we're
    /// emulating an [OpStore], then this slice should have a length of one and represents the value
    /// to be stored.
    additional_values: &'a [ValueInput],
}

impl<'a, F> Emulator<'a, F>
where
    F: Fn(&mut Rvsdg, Region, ValueInput, &[ValueInput], StateOrigin) -> Node,
{
    fn emulate(&mut self, root_node: &EmulationTreeNode) -> Node {
        match root_node {
            EmulationTreeNode::Branching(node) => {
                self.visit_branching_node(self.outer_region, node, None)
            }
            EmulationTreeNode::Leaf(node) => self.visit_leaf_node(self.outer_region, node, None),
        }
    }

    fn visit_branching_node(
        &mut self,
        region: Region,
        branching_node: &BranchingNode,
        input_mapping: Option<&IndexSet<ValueOrigin>>,
    ) -> Node {
        let resolve_input = |origin, ty| {
            if let Some(input_mapping) = input_mapping {
                let argument = input_mapping
                    .get_index_of(&origin)
                    .expect("input mapping was not correctly constructed");

                ValueInput::argument(ty, argument as u32)
            } else {
                ValueInput { ty, origin }
            }
        };

        let mut value_inputs = Vec::with_capacity(branching_node.child_inputs.len() + 1);

        // Connect the first input to the branch selector predicate.
        value_inputs.push(resolve_input(branching_node.branch_selector, TY_PREDICATE));

        // Connect inputs for the emulation values required by the branching node's child nodes.
        for child_input in branching_node.child_inputs.iter().copied() {
            let ty = self.rvsdg.value_origin_ty(self.outer_region, child_input);

            value_inputs.push(resolve_input(child_input, ty));
        }

        // Connect inputs for any non-pointer additional values passed in for the operation (e.g.
        // the value to be stored by an OpStore).
        if let Some(input_mapping) = input_mapping {
            // We're inside a branching node, the additional inputs will be available as
            // arguments after the emulation arguments represented in the input_mapping.
            let base_index = input_mapping.len();

            for (i, additional_value) in self.additional_values.iter().enumerate() {
                let argument_index = base_index + i;

                value_inputs.push(ValueInput::argument(
                    additional_value.ty,
                    argument_index as u32,
                ));
            }
        } else {
            // We're processing the root node, the additional inputs should be available
            // directly
            value_inputs.extend(self.additional_values.iter().copied());
        }

        let value_outputs = if let Some(output_ty) = self.output_ty {
            vec![ValueOutput::new(output_ty)]
        } else {
            vec![]
        };

        let state_origin = if input_mapping.is_some() {
            // We're inside a switch node for a branching node, directly connect to the state
            // argument.
            StateOrigin::Argument
        } else {
            // This was the root of the emulation tree, so we're inside the outer region. Connect
            // to the provided state origin for the original load/store operation.
            self.state_origin
        };

        let switch_node =
            self.rvsdg
                .add_switch(region, value_inputs, value_outputs, Some(state_origin));

        for child in &branching_node.branches {
            let branch = self.rvsdg.add_switch_branch(switch_node);

            let node = match child {
                EmulationTreeNode::Branching(node) => {
                    self.visit_branching_node(branch, node, Some(&branching_node.child_inputs))
                }
                EmulationTreeNode::Leaf(node) => {
                    self.visit_leaf_node(branch, node, Some(&branching_node.child_inputs))
                }
            };

            if self.output_ty.is_some() {
                self.rvsdg.reconnect_region_result(
                    branch,
                    0,
                    ValueOrigin::Output {
                        producer: node,
                        output: 0,
                    },
                )
            }
        }

        switch_node
    }

    fn visit_leaf_node(
        &mut self,
        region: Region,
        leaf_node: &LeafNode,
        input_mapping: Option<&IndexSet<ValueOrigin>>,
    ) -> Node {
        let resolve_input = |origin, ty| {
            if let Some(input_mapping) = input_mapping {
                let argument = input_mapping
                    .get_index_of(&origin)
                    .expect("input mapping was not correctly constructed");

                ValueInput::argument(ty, argument as u32)
            } else {
                ValueInput { ty, origin }
            }
        };

        let ptr_ty = self
            .rvsdg
            .value_origin_ty(self.outer_region, leaf_node.root_pointer);
        let ptr_input = resolve_input(leaf_node.root_pointer, ptr_ty);

        let ptr_input = if leaf_node.access_chain.is_empty() {
            ptr_input
        } else {
            let mut ptr_input = ptr_input;
            let mut last = None;

            for access in &leaf_node.access_chain {
                let (node, ty) = match access {
                    Access::Field(field_index) => {
                        let node = self.rvsdg.add_op_field_ptr(region, ptr_input, *field_index);
                        let ty = self.rvsdg[node].expect_op_field_ptr().value_output().ty;

                        (node, ty)
                    }
                    Access::StaticElement(element_index) => {
                        let index_node = self.rvsdg.add_const_u32(region, *element_index);
                        let node = self.rvsdg.add_op_element_ptr(
                            region,
                            ptr_input,
                            ValueInput::output(TY_U32, index_node, 0),
                        );
                        let ty = self.rvsdg[node].expect_op_element_ptr().value_output().ty;

                        (node, ty)
                    }
                    Access::DynamicElement(origin) => {
                        let input = resolve_input(*origin, TY_U32);
                        let node = self.rvsdg.add_op_element_ptr(region, ptr_input, input);
                        let ty = self.rvsdg[node].expect_op_element_ptr().value_output().ty;

                        (node, ty)
                    }
                };

                ptr_input = ValueInput::output(ty, node, 0);
                last = Some(node);
            }

            let last = last.expect("verified that leaf node access chain is non-empty");

            ValueInput::output(self.pointer_ty, last, 0)
        };

        if let Some(input_mapping) = input_mapping {
            // We're inside a switch node that represents a parent branching node. The additional
            // values will have been passed in as arguments after all the arguments used for pointer
            // emulation itself.

            let base_index = input_mapping.len();
            let additional_values = self
                .additional_values
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let argument_index = base_index + i;

                    ValueInput::argument(v.ty, argument_index as u32)
                })
                .collect::<SmallVec<[ValueInput; 2]>>();

            (self.op_gen)(
                self.rvsdg,
                region,
                ptr_input,
                &additional_values,
                StateOrigin::Argument,
            )
        } else {
            // There is no parent branching node, so we are inside the outer region and the
            // additional values can be connected to directly.
            (self.op_gen)(
                self.rvsdg,
                region,
                ptr_input,
                self.additional_values,
                self.state_origin,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::rvsdg::ValueUser;
    use crate::ty::TY_DUMMY;
    use crate::{BinaryOperator, FnArg, FnSig, Function, Module, Symbol, thin_set};

    #[test]
    fn test_emulate_single_switch_output_load() {
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
                    ty: TY_PREDICATE,
                    shader_io_binding: None,
                }],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ptr_0 = rvsdg.add_op_alloca(region, TY_U32);
        let ptr_1 = rvsdg.add_op_alloca(region, TY_U32);

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(ptr_ty, ptr_0, 0),
                ValueInput::output(ptr_ty, ptr_1, 0),
            ],
            vec![ValueOutput::new(ptr_ty)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);

        rvsdg.reconnect_region_result(branch_0, 0, ValueOrigin::Argument(0));

        let branch_1 = rvsdg.add_switch_branch(switch_node);

        rvsdg.reconnect_region_result(branch_1, 0, ValueOrigin::Argument(1));

        let load_op = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, switch_node, 0),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_op,
                output: 0,
            },
        );

        let mut emulation_context = EmulationContext::new();

        emulation_context.emulate_op_load(&mut rvsdg, load_op);

        let ValueOrigin::Output {
            producer: emulation_switch_node,
            output: 0,
        } = rvsdg[region].value_results()[0].origin
        else {
            panic!("the function result should connect to the first output of a node")
        };

        let emulation_switch_data = rvsdg[emulation_switch_node].expect_switch();

        assert_eq!(
            emulation_switch_data.value_inputs(),
            &[
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(ptr_ty, ptr_0, 0),
                ValueInput::output(ptr_ty, ptr_1, 0),
            ]
        );
        assert_eq!(
            emulation_switch_data.value_outputs(),
            &[ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Result(0)]
            }]
        );
        assert_eq!(emulation_switch_data.branches().len(), 2);

        let emulation_branch_0 = emulation_switch_data.branches()[0];

        let ValueOrigin::Output {
            producer: emulation_load_0_node,
            output: 0,
        } = rvsdg[emulation_branch_0].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_load_0_data = rvsdg[emulation_load_0_node].expect_op_load();

        assert_eq!(
            emulation_load_0_data.value_inputs(),
            &[ValueInput::argument(ptr_ty, 0)]
        );

        let emulation_branch_1 = emulation_switch_data.branches()[1];

        let ValueOrigin::Output {
            producer: emulation_load_1_node,
            output: 0,
        } = rvsdg[emulation_branch_1].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_load_1_data = rvsdg[emulation_load_1_node].expect_op_load();

        assert_eq!(
            emulation_load_1_data.value_inputs(),
            &[ValueInput::argument(ptr_ty, 1)]
        );

        assert_eq!(
            &rvsdg[ptr_0].expect_op_alloca().value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: switch_node,
                    input: 1,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 1,
                }
            ],
            "the emulation node should have been added as a user to the first alloca node"
        );

        assert_eq!(
            &rvsdg[ptr_1].expect_op_alloca().value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: switch_node,
                    input: 2,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 2,
                }
            ],
            "the emulation node should have been added as a user to the second alloca node"
        );

        assert!(
            rvsdg[switch_node].value_outputs()[0].users.is_empty(),
            "original switch node should no longer have any users"
        );
    }

    #[test]
    fn test_emulate_single_switch_output_load_inside_switch() {
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
                    ty: TY_PREDICATE,
                    shader_io_binding: None,
                }],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ptr_0 = rvsdg.add_op_alloca(region, TY_U32);
        let ptr_1 = rvsdg.add_op_alloca(region, TY_U32);

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let switch_0_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(ptr_ty, ptr_0, 0),
                ValueInput::output(ptr_ty, ptr_1, 0),
            ],
            vec![ValueOutput::new(ptr_ty)],
            None,
        );

        let switch_0_branch_0 = rvsdg.add_switch_branch(switch_0_node);

        rvsdg.reconnect_region_result(switch_0_branch_0, 0, ValueOrigin::Argument(0));

        let switch_0_branch_1 = rvsdg.add_switch_branch(switch_0_node);

        rvsdg.reconnect_region_result(switch_0_branch_1, 0, ValueOrigin::Argument(1));

        let switch_1_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(ptr_ty, switch_0_node, 0),
            ],
            vec![ValueOutput::new(TY_U32)],
            Some(StateOrigin::Argument),
        );

        let switch_1_branch_0 = rvsdg.add_switch_branch(switch_1_node);

        let load_op = rvsdg.add_op_load(
            switch_1_branch_0,
            ValueInput::argument(ptr_ty, 0),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            switch_1_branch_0,
            0,
            ValueOrigin::Output {
                producer: load_op,
                output: 0,
            },
        );

        let switch_1_branch_1 = rvsdg.add_switch_branch(switch_1_node);

        let fallback = rvsdg.add_const_u32(switch_1_branch_1, 0);

        rvsdg.reconnect_region_result(
            switch_1_branch_1,
            0,
            ValueOrigin::Output {
                producer: fallback,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: switch_1_node,
                output: 0,
            },
        );

        let mut emulation_context = EmulationContext::new();

        emulation_context.emulate_op_load(&mut rvsdg, load_op);

        let ValueOrigin::Output {
            producer: emulation_switch_node,
            output: 0,
        } = rvsdg[switch_1_branch_0].value_results()[0].origin
        else {
            panic!("the second switch's result should connect to the first output of a node")
        };

        let emulation_switch_data = rvsdg[emulation_switch_node].expect_switch();

        assert_eq!(
            emulation_switch_data.value_inputs(),
            &[
                ValueInput::argument(TY_PREDICATE, 1),
                ValueInput::argument(ptr_ty, 2),
                ValueInput::argument(ptr_ty, 3),
            ]
        );
        assert_eq!(
            emulation_switch_data.value_outputs(),
            &[ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Result(0)]
            }]
        );

        assert_eq!(
            rvsdg[switch_1_node].value_inputs(),
            &[
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(ptr_ty, switch_0_node, 0),
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(ptr_ty, ptr_0, 0),
                ValueInput::output(ptr_ty, ptr_1, 0),
            ],
            "the predicate and both alloca nodes have been added as inputs to the second switch node"
        );

        assert!(
            rvsdg[switch_1_branch_0].value_arguments()[0]
                .users
                .is_empty(),
            "the second switch's variable pointer argument in the first branch should no longer \
            have any users"
        );
        assert_eq!(
            &rvsdg[switch_1_branch_0].value_arguments()[1].users,
            &thin_set![ValueUser::Input {
                consumer: emulation_switch_node,
                input: 0
            }],
            "the newly added predicate argument in the second switch's first branch should be used \
            by the emulation node"
        );
        assert_eq!(
            &rvsdg[switch_1_branch_0].value_arguments()[2].users,
            &thin_set![ValueUser::Input {
                consumer: emulation_switch_node,
                input: 1
            }],
            "the newly added ptr_0 argument in the second switch's first branch should be used \
            by the emulation node"
        );
        assert_eq!(
            &rvsdg[switch_1_branch_0].value_arguments()[3].users,
            &thin_set![ValueUser::Input {
                consumer: emulation_switch_node,
                input: 2
            }],
            "the newly added ptr_1 argument in the second switch's first branch should be used \
            by the emulation node"
        );

        assert!(
            rvsdg[switch_1_branch_1].value_arguments()[0]
                .users
                .is_empty(),
            "the second switch's variable pointer argument in the second branch should not have \
            any users"
        );
        assert!(
            rvsdg[switch_1_branch_1].value_arguments()[1]
                .users
                .is_empty(),
            "the newly added predicate argument in the second switch's second branch should not \
            have any users"
        );
        assert!(
            rvsdg[switch_1_branch_1].value_arguments()[2]
                .users
                .is_empty(),
            "the newly added ptr_0 argument in the second switch's second branch should not \
            have any users"
        );
        assert!(
            rvsdg[switch_1_branch_1].value_arguments()[3]
                .users
                .is_empty(),
            "the newly added ptr_1 argument in the second switch's second branch should not \
            have any users"
        );

        assert_eq!(
            &rvsdg[region].value_arguments()[0].users,
            &thin_set![
                ValueUser::Input {
                    consumer: switch_0_node,
                    input: 0,
                },
                ValueUser::Input {
                    consumer: switch_1_node,
                    input: 0,
                },
                ValueUser::Input {
                    consumer: switch_1_node,
                    input: 2,
                }
            ],
            "the second switch node should have been added as a user to the predicate function \
            argument"
        );

        assert_eq!(
            &rvsdg[ptr_0].expect_op_alloca().value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: switch_0_node,
                    input: 1,
                },
                ValueUser::Input {
                    consumer: switch_1_node,
                    input: 3,
                }
            ],
            "the second switch node should have been added as a user to the first alloca node"
        );

        assert_eq!(
            &rvsdg[ptr_1].expect_op_alloca().value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: switch_0_node,
                    input: 2,
                },
                ValueUser::Input {
                    consumer: switch_1_node,
                    input: 4,
                }
            ],
            "the second switch node should have been added as a user to the second alloca node"
        );
    }

    #[test]
    fn test_emulate_single_static_access_single_switch_output_load() {
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
                    ty: TY_PREDICATE,
                    shader_io_binding: None,
                }],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let array_ty = module.ty.register(TypeKind::Array {
            element_ty: TY_U32,
            count: 2,
            stride: 4,
        });
        let array_ptr_ty = module.ty.register(TypeKind::Ptr(array_ty));

        let array_alloca_node = rvsdg.add_op_alloca(region, array_ty);
        let index_node = rvsdg.add_const_u32(region, 1);

        let ptr_0_node = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(array_ptr_ty, array_alloca_node, 0),
            ValueInput::output(TY_U32, index_node, 0),
        );
        let ptr_1_node = rvsdg.add_op_alloca(region, TY_U32);

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(ptr_ty, ptr_0_node, 0),
                ValueInput::output(ptr_ty, ptr_1_node, 0),
            ],
            vec![ValueOutput::new(ptr_ty)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);

        rvsdg.reconnect_region_result(branch_0, 0, ValueOrigin::Argument(0));

        let branch_1 = rvsdg.add_switch_branch(switch_node);

        rvsdg.reconnect_region_result(branch_1, 0, ValueOrigin::Argument(1));

        let load_op = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, switch_node, 0),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_op,
                output: 0,
            },
        );

        let mut emulation_context = EmulationContext::new();

        emulation_context.emulate_op_load(&mut rvsdg, load_op);

        let ValueOrigin::Output {
            producer: emulation_switch_node,
            output: 0,
        } = rvsdg[region].value_results()[0].origin
        else {
            panic!("the function result should connect to the first output of a node")
        };

        let emulation_switch_data = rvsdg[emulation_switch_node].expect_switch();

        assert_eq!(
            emulation_switch_data.value_inputs(),
            &[
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(array_ptr_ty, array_alloca_node, 0),
                ValueInput::output(ptr_ty, ptr_1_node, 0),
            ]
        );
        assert_eq!(
            emulation_switch_data.value_outputs(),
            &[ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Result(0)]
            }]
        );
        assert_eq!(emulation_switch_data.branches().len(), 2);

        let emulation_branch_0 = emulation_switch_data.branches()[0];

        let ValueOrigin::Output {
            producer: emulation_0_load_node,
            output: 0,
        } = rvsdg[emulation_branch_0].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_0_load_data = rvsdg[emulation_0_load_node].expect_op_load();

        let ValueOrigin::Output {
            producer: emulation_0_ep_node,
            output: 0,
        } = emulation_0_load_data.ptr_input().origin
        else {
            panic!("the load node in branch `0` should connect to the first output of a node")
        };

        let emulation_ep_data = rvsdg[emulation_0_ep_node].expect_op_element_ptr();

        assert_eq!(emulation_ep_data.value_inputs().len(), 2);
        assert_eq!(
            emulation_ep_data.ptr_input(),
            &ValueInput::argument(array_ptr_ty, 0)
        );

        let ValueOrigin::Output {
            producer: emulation_0_index_node,
            output: 0,
        } = emulation_ep_data.index_input().origin
        else {
            panic!("the second element-ptr node input should connect to the first output of a node")
        };

        let emulation_0_index_data = rvsdg[emulation_0_index_node].expect_const_u32();

        assert_eq!(emulation_0_index_data.value(), 1);

        let emulation_branch_1 = emulation_switch_data.branches()[1];

        let ValueOrigin::Output {
            producer: emulation_1_load_node,
            output: 0,
        } = rvsdg[emulation_branch_1].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_1_load_data = rvsdg[emulation_1_load_node].expect_op_load();

        assert_eq!(
            emulation_1_load_data.value_inputs(),
            &[ValueInput::argument(ptr_ty, 1)]
        );

        assert_eq!(
            &rvsdg[array_alloca_node]
                .expect_op_alloca()
                .value_output()
                .users,
            &thin_set![
                ValueUser::Input {
                    consumer: ptr_0_node,
                    input: 0,
                },
                ValueUser::Input {
                    consumer: switch_node,
                    input: 3,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 1,
                }
            ],
            "the switch node and the emulation node should have been added as a user to the first \
            alloca node"
        );

        assert!(
            rvsdg[branch_0].value_arguments()[2].users.is_empty(),
            "the array pointer should have been made available inside the first branch of the \
            switch node, but should not be used"
        );
        assert!(
            rvsdg[branch_1].value_arguments()[2].users.is_empty(),
            "the array pointer should have been made available inside the second branch of the \
            switch node, but should not be used"
        );

        assert_eq!(
            &rvsdg[ptr_1_node].expect_op_alloca().value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: switch_node,
                    input: 2,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 2,
                }
            ],
            "the emulation node should have been added as a user to the second alloca node"
        );

        assert!(
            rvsdg[switch_node].value_outputs()[0].users.is_empty(),
            "original switch node should no longer have any users"
        );
    }

    #[test]
    fn test_emulate_single_dynamic_access_single_switch_output_load() {
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

        let array_ty = module.ty.register(TypeKind::Array {
            element_ty: TY_U32,
            count: 2,
            stride: 4,
        });
        let array_ptr_ty = module.ty.register(TypeKind::Ptr(array_ty));

        let array_alloca_node = rvsdg.add_op_alloca(region, array_ty);

        let ptr_0_node = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(array_ptr_ty, array_alloca_node, 0),
            ValueInput::argument(TY_U32, 1),
        );
        let ptr_1_node = rvsdg.add_op_alloca(region, TY_U32);

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(ptr_ty, ptr_0_node, 0),
                ValueInput::output(ptr_ty, ptr_1_node, 0),
            ],
            vec![ValueOutput::new(ptr_ty)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);

        rvsdg.reconnect_region_result(branch_0, 0, ValueOrigin::Argument(0));

        let branch_1 = rvsdg.add_switch_branch(switch_node);

        rvsdg.reconnect_region_result(branch_1, 0, ValueOrigin::Argument(1));

        let load_op = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, switch_node, 0),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_op,
                output: 0,
            },
        );

        let mut emulation_context = EmulationContext::new();

        emulation_context.emulate_op_load(&mut rvsdg, load_op);

        let ValueOrigin::Output {
            producer: emulation_switch_node,
            output: 0,
        } = rvsdg[region].value_results()[0].origin
        else {
            panic!("the function result should connect to the first output of a node")
        };

        let emulation_switch_data = rvsdg[emulation_switch_node].expect_switch();

        assert_eq!(
            emulation_switch_data.value_inputs(),
            &[
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(array_ptr_ty, array_alloca_node, 0),
                ValueInput::argument(TY_U32, 1),
                ValueInput::output(ptr_ty, ptr_1_node, 0),
            ]
        );
        assert_eq!(
            emulation_switch_data.value_outputs(),
            &[ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Result(0)]
            }]
        );
        assert_eq!(emulation_switch_data.branches().len(), 2);

        let emulation_branch_0 = emulation_switch_data.branches()[0];

        let ValueOrigin::Output {
            producer: emulation_0_load_node,
            output: 0,
        } = rvsdg[emulation_branch_0].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_0_load_data = rvsdg[emulation_0_load_node].expect_op_load();

        let ValueOrigin::Output {
            producer: emulation_0_ep_node,
            output: 0,
        } = emulation_0_load_data.ptr_input().origin
        else {
            panic!("the load node in branch `0` should connect to the first output of a node")
        };

        let emulation_ep_data = rvsdg[emulation_0_ep_node].expect_op_element_ptr();

        assert_eq!(
            emulation_ep_data.value_inputs(),
            &[
                ValueInput::argument(array_ptr_ty, 0),
                ValueInput::argument(TY_U32, 1),
            ]
        );

        let emulation_branch_1 = emulation_switch_data.branches()[1];

        let ValueOrigin::Output {
            producer: emulation_1_load_node,
            output: 0,
        } = rvsdg[emulation_branch_1].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_1_load_data = rvsdg[emulation_1_load_node].expect_op_load();

        assert_eq!(
            emulation_1_load_data.value_inputs(),
            &[ValueInput::argument(ptr_ty, 2)]
        );

        assert_eq!(
            &rvsdg[array_alloca_node]
                .expect_op_alloca()
                .value_output()
                .users,
            &thin_set![
                ValueUser::Input {
                    consumer: ptr_0_node,
                    input: 0,
                },
                ValueUser::Input {
                    consumer: switch_node,
                    input: 3,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 1,
                }
            ],
            "the switch node and the emulation node should have been added as a user to the first \
            alloca node"
        );

        assert!(
            rvsdg[branch_0].value_arguments()[2].users.is_empty(),
            "the array pointer should have been made available inside the first branch of the \
            switch node, but should not be used"
        );
        assert!(
            rvsdg[branch_0].value_arguments()[3].users.is_empty(),
            "the dynamic index should have been made available inside the first branch of the \
            switch node, but should not be used"
        );
        assert!(
            rvsdg[branch_1].value_arguments()[2].users.is_empty(),
            "the array pointer should have been made available inside the second branch of the \
            switch node, but should not be used"
        );
        assert!(
            rvsdg[branch_1].value_arguments()[3].users.is_empty(),
            "the dynamic index should have been made available inside the second branch of the \
            switch node, but should not be used"
        );

        assert_eq!(
            &rvsdg[ptr_1_node].expect_op_alloca().value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: switch_node,
                    input: 2,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 3,
                }
            ],
            "the emulation node should have been added as a user to the second alloca node"
        );

        assert_eq!(
            &rvsdg[region].value_arguments()[1].users,
            &thin_set![
                ValueUser::Input {
                    consumer: ptr_0_node,
                    input: 1,
                },
                ValueUser::Input {
                    consumer: switch_node,
                    input: 4,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 2,
                }
            ],
            "the switch node and the emulation node should have been added as a user to the \
            dynamic index argument"
        );

        assert!(
            rvsdg[switch_node].value_outputs()[0].users.is_empty(),
            "original switch node should no longer have any users"
        );
    }

    #[test]
    fn test_emulate_double_static_access_single_switch_output_load() {
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
                    ty: TY_PREDICATE,
                    shader_io_binding: None,
                }],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let array_ty = module.ty.register(TypeKind::Array {
            element_ty: TY_U32,
            count: 2,
            stride: 4,
        });
        let array_of_array_ty = module.ty.register(TypeKind::Array {
            element_ty: array_ty,
            count: 2,
            stride: 8,
        });
        let array_ptr_ty = module.ty.register(TypeKind::Ptr(array_ty));
        let array_of_array_ptr_ty = module.ty.register(TypeKind::Ptr(array_of_array_ty));
        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let array_alloca_node = rvsdg.add_op_alloca(region, array_of_array_ty);
        let index_node = rvsdg.add_const_u32(region, 1);

        let ptr_0_node = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(array_of_array_ptr_ty, array_alloca_node, 0),
            ValueInput::output(TY_U32, index_node, 0),
        );
        let ptr_1_node = rvsdg.add_op_alloca(region, TY_U32);

        let switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(array_ptr_ty, ptr_0_node, 0),
                ValueInput::output(ptr_ty, ptr_1_node, 0),
            ],
            vec![ValueOutput::new(ptr_ty)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);

        let branch_0_index_node = rvsdg.add_const_u32(branch_0, 0);
        let branch_0_ptr_node = rvsdg.add_op_element_ptr(
            branch_0,
            ValueInput::argument(array_ptr_ty, 0),
            ValueInput::output(TY_U32, branch_0_index_node, 0),
        );

        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: branch_0_ptr_node,
                output: 0,
            },
        );

        let branch_1 = rvsdg.add_switch_branch(switch_node);

        rvsdg.reconnect_region_result(branch_1, 0, ValueOrigin::Argument(1));

        let load_op = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, switch_node, 0),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_op,
                output: 0,
            },
        );

        let mut emulation_context = EmulationContext::new();

        emulation_context.emulate_op_load(&mut rvsdg, load_op);

        let ValueOrigin::Output {
            producer: emulation_switch_node,
            output: 0,
        } = rvsdg[region].value_results()[0].origin
        else {
            panic!("the function result should connect to the first output of a node")
        };

        let emulation_switch_data = rvsdg[emulation_switch_node].expect_switch();

        assert_eq!(
            emulation_switch_data.value_inputs(),
            &[
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(array_of_array_ptr_ty, array_alloca_node, 0),
                ValueInput::output(ptr_ty, ptr_1_node, 0),
            ]
        );
        assert_eq!(
            emulation_switch_data.value_outputs(),
            &[ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Result(0)]
            }]
        );
        assert_eq!(emulation_switch_data.branches().len(), 2);

        let emulation_branch_0 = emulation_switch_data.branches()[0];

        let ValueOrigin::Output {
            producer: emulation_0_load_node,
            output: 0,
        } = rvsdg[emulation_branch_0].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_0_load_data = rvsdg[emulation_0_load_node].expect_op_load();

        let ValueOrigin::Output {
            producer: emulation_0_ep1_node,
            output: 0,
        } = emulation_0_load_data.ptr_input().origin
        else {
            panic!("the load node in branch `0` should connect to the first output of a node")
        };

        let emulation_ep1_data = rvsdg[emulation_0_ep1_node].expect_op_element_ptr();

        let ValueOrigin::Output {
            producer: emulation_0_index_1_node,
            output: 0,
        } = emulation_ep1_data.index_input().origin
        else {
            panic!("the third element-ptr node input should connect to the first output of a node")
        };

        let emulation_0_index_1_data = rvsdg[emulation_0_index_1_node].expect_const_u32();

        assert_eq!(emulation_0_index_1_data.value(), 0);

        let ValueOrigin::Output {
            producer: emulation_0_ep0_node,
            output: 0,
        } = emulation_ep1_data.ptr_input().origin
        else {
            panic!(
                "the second element-pointer node input should connect to the first output of a \
                node"
            )
        };

        let emulation_0_ep0_data = rvsdg[emulation_0_ep0_node].expect_op_element_ptr();

        assert_eq!(
            emulation_0_ep0_data.ptr_input(),
            &ValueInput::argument(array_of_array_ptr_ty, 0)
        );

        let ValueOrigin::Output {
            producer: emulation_0_index_0_node,
            output: 0,
        } = emulation_0_ep0_data.index_input().origin
        else {
            panic!("the second element-ptr node input should connect to the first output of a node")
        };

        let emulation_0_index_0_data = rvsdg[emulation_0_index_0_node].expect_const_u32();

        assert_eq!(emulation_0_index_0_data.value(), 1);

        let emulation_branch_1 = emulation_switch_data.branches()[1];

        let ValueOrigin::Output {
            producer: emulation_1_load_node,
            output: 0,
        } = rvsdg[emulation_branch_1].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_1_load_data = rvsdg[emulation_1_load_node].expect_op_load();

        assert_eq!(
            emulation_1_load_data.value_inputs(),
            &[ValueInput::argument(ptr_ty, 1)]
        );

        assert_eq!(
            &rvsdg[array_alloca_node]
                .expect_op_alloca()
                .value_output()
                .users,
            &thin_set![
                ValueUser::Input {
                    consumer: ptr_0_node,
                    input: 0,
                },
                ValueUser::Input {
                    consumer: switch_node,
                    input: 3,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 1,
                }
            ],
            "the switch node and the emulation node should have been added as a user to the first \
            alloca node"
        );

        assert!(
            rvsdg[branch_0].value_arguments()[2].users.is_empty(),
            "the array pointer should have been made available inside the first branch of the \
            switch node, but should not be used"
        );
        assert!(
            rvsdg[branch_1].value_arguments()[2].users.is_empty(),
            "the array pointer should have been made available inside the second branch of the \
            switch node, but should not be used"
        );

        assert_eq!(
            &rvsdg[ptr_1_node].expect_op_alloca().value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: switch_node,
                    input: 2,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 2,
                }
            ],
            "the emulation node should have been added as a user to the second alloca node"
        );

        assert!(
            rvsdg[switch_node].value_outputs()[0].users.is_empty(),
            "original switch node should no longer have any users"
        );
    }

    #[test]
    fn test_emulate_double_switch_output_load() {
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
                        ty: TY_PREDICATE,
                        shader_io_binding: None,
                    },
                ],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let ptr_0_node = rvsdg.add_op_alloca(region, TY_U32);
        let ptr_1_node = rvsdg.add_op_alloca(region, TY_U32);
        let ptr_2_node = rvsdg.add_op_alloca(region, TY_U32);

        let first_switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 1),
                ValueInput::output(ptr_ty, ptr_0_node, 0),
                ValueInput::output(ptr_ty, ptr_1_node, 0),
            ],
            vec![ValueOutput::new(ptr_ty)],
            None,
        );

        let first_switch_branch_0 = rvsdg.add_switch_branch(first_switch_node);

        rvsdg.reconnect_region_result(first_switch_branch_0, 0, ValueOrigin::Argument(0));

        let first_switch_branch_1 = rvsdg.add_switch_branch(first_switch_node);

        rvsdg.reconnect_region_result(first_switch_branch_1, 0, ValueOrigin::Argument(1));

        let second_switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(ptr_ty, first_switch_node, 0),
                ValueInput::output(ptr_ty, ptr_2_node, 0),
            ],
            vec![ValueOutput::new(ptr_ty)],
            None,
        );

        let second_switch_branch_0 = rvsdg.add_switch_branch(second_switch_node);

        rvsdg.reconnect_region_result(second_switch_branch_0, 0, ValueOrigin::Argument(0));

        let second_switch_branch_1 = rvsdg.add_switch_branch(second_switch_node);

        rvsdg.reconnect_region_result(second_switch_branch_1, 0, ValueOrigin::Argument(1));

        let load_op = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, second_switch_node, 0),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_op,
                output: 0,
            },
        );

        let mut emulation_context = EmulationContext::new();

        emulation_context.emulate_op_load(&mut rvsdg, load_op);

        let ValueOrigin::Output {
            producer: emulation_switch_node,
            output: 0,
        } = rvsdg[region].value_results()[0].origin
        else {
            panic!("the function result should connect to the first output of a node")
        };

        let emulation_switch_data = rvsdg[emulation_switch_node].expect_switch();

        assert_eq!(
            emulation_switch_data.value_inputs(),
            &[
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::argument(TY_PREDICATE, 1),
                ValueInput::output(ptr_ty, ptr_0_node, 0),
                ValueInput::output(ptr_ty, ptr_1_node, 0),
                ValueInput::output(ptr_ty, ptr_2_node, 0),
            ]
        );
        assert_eq!(
            emulation_switch_data.value_outputs(),
            &[ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Result(0)]
            }]
        );
        assert_eq!(emulation_switch_data.branches().len(), 2);

        let emulation_branch_0 = emulation_switch_data.branches()[0];

        let ValueOrigin::Output {
            producer: emulation_0_switch_node,
            output: 0,
        } = rvsdg[emulation_branch_0].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_0_switch_data = rvsdg[emulation_0_switch_node].expect_switch();

        assert_eq!(
            emulation_0_switch_data.value_inputs(),
            &[
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::argument(ptr_ty, 1),
                ValueInput::argument(ptr_ty, 2),
            ]
        );
        assert_eq!(
            emulation_0_switch_data.value_outputs(),
            &[ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Result(0)]
            }]
        );
        assert_eq!(emulation_0_switch_data.branches().len(), 2);

        let emulation_branch_0_0 = emulation_0_switch_data.branches()[0];

        let ValueOrigin::Output {
            producer: emulation_0_0_load_node,
            output: 0,
        } = rvsdg[emulation_branch_0_0].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_0_0_load_data = rvsdg[emulation_0_0_load_node].expect_op_load();

        assert_eq!(
            emulation_0_0_load_data.value_inputs(),
            &[ValueInput::argument(ptr_ty, 0)]
        );

        let emulation_branch_0_1 = emulation_0_switch_data.branches()[1];

        let ValueOrigin::Output {
            producer: emulation_0_1_load_node,
            output: 0,
        } = rvsdg[emulation_branch_0_1].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_0_1_load_data = rvsdg[emulation_0_1_load_node].expect_op_load();

        assert_eq!(
            emulation_0_1_load_data.value_inputs(),
            &[ValueInput::argument(ptr_ty, 1)]
        );

        let emulation_branch_1 = emulation_switch_data.branches()[1];

        let ValueOrigin::Output {
            producer: emulation_1_load_node,
            output: 0,
        } = rvsdg[emulation_branch_1].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_1_load_data = rvsdg[emulation_1_load_node].expect_op_load();

        assert_eq!(
            emulation_1_load_data.value_inputs(),
            &[ValueInput::argument(ptr_ty, 3)]
        );

        assert_eq!(
            &rvsdg[ptr_0_node].expect_op_alloca().value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: first_switch_node,
                    input: 1,
                },
                ValueUser::Input {
                    consumer: second_switch_node,
                    input: 4,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 2,
                }
            ],
            "the second switch node and the emulation should have been added as a user to the \
            first alloca node"
        );

        assert_eq!(
            &rvsdg[ptr_1_node].expect_op_alloca().value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: first_switch_node,
                    input: 2,
                },
                ValueUser::Input {
                    consumer: second_switch_node,
                    input: 5,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 3,
                }
            ],
            "the second switch node and the emulation should have been added as a user to the \
            second alloca node"
        );

        assert_eq!(
            &rvsdg[ptr_2_node].expect_op_alloca().value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: second_switch_node,
                    input: 2,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 4,
                }
            ],
            "the emulation node should have been added as a user to the third alloca node"
        );

        assert!(
            rvsdg[second_switch_node].value_outputs()[0]
                .users
                .is_empty(),
            "the second switch node should no longer have any users"
        );
    }

    #[test]
    fn test_emulate_nested_switch_output_load() {
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
                        ty: TY_PREDICATE,
                        shader_io_binding: None,
                    },
                ],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let ptr_0_node = rvsdg.add_op_alloca(region, TY_U32);
        let ptr_1_node = rvsdg.add_op_alloca(region, TY_U32);
        let ptr_2_node = rvsdg.add_op_alloca(region, TY_U32);

        let outer_switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::argument(TY_PREDICATE, 1),
                ValueInput::output(ptr_ty, ptr_0_node, 0),
                ValueInput::output(ptr_ty, ptr_1_node, 0),
                ValueInput::output(ptr_ty, ptr_2_node, 0),
            ],
            vec![ValueOutput::new(ptr_ty)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(outer_switch_node);

        let inner_switch_node = rvsdg.add_switch(
            branch_0,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::argument(ptr_ty, 1),
                ValueInput::argument(ptr_ty, 2),
            ],
            vec![ValueOutput::new(ptr_ty)],
            None,
        );

        let branch_0_branch_0 = rvsdg.add_switch_branch(inner_switch_node);

        rvsdg.reconnect_region_result(branch_0_branch_0, 0, ValueOrigin::Argument(0));

        let branch_0_branch_1 = rvsdg.add_switch_branch(inner_switch_node);

        rvsdg.reconnect_region_result(branch_0_branch_1, 0, ValueOrigin::Argument(1));

        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: inner_switch_node,
                output: 0,
            },
        );

        let branch_1 = rvsdg.add_switch_branch(outer_switch_node);

        rvsdg.reconnect_region_result(branch_1, 0, ValueOrigin::Argument(3));

        let load_op = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, outer_switch_node, 0),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_op,
                output: 0,
            },
        );

        let mut emulation_context = EmulationContext::new();

        emulation_context.emulate_op_load(&mut rvsdg, load_op);

        let ValueOrigin::Output {
            producer: emulation_switch_node,
            output: 0,
        } = rvsdg[region].value_results()[0].origin
        else {
            panic!("the function result should connect to the first output of a node")
        };

        let emulation_switch_data = rvsdg[emulation_switch_node].expect_switch();

        assert_eq!(
            emulation_switch_data.value_inputs(),
            &[
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::argument(TY_PREDICATE, 1),
                ValueInput::output(ptr_ty, ptr_0_node, 0),
                ValueInput::output(ptr_ty, ptr_1_node, 0),
                ValueInput::output(ptr_ty, ptr_2_node, 0),
            ]
        );
        assert_eq!(
            emulation_switch_data.value_outputs(),
            &[ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Result(0)]
            }]
        );
        assert_eq!(emulation_switch_data.branches().len(), 2);

        let emulation_branch_0 = emulation_switch_data.branches()[0];

        let ValueOrigin::Output {
            producer: emulation_0_switch_node,
            output: 0,
        } = rvsdg[emulation_branch_0].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_0_switch_data = rvsdg[emulation_0_switch_node].expect_switch();

        assert_eq!(
            emulation_0_switch_data.value_inputs(),
            &[
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::argument(ptr_ty, 1),
                ValueInput::argument(ptr_ty, 2),
            ]
        );
        assert_eq!(
            emulation_0_switch_data.value_outputs(),
            &[ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Result(0)]
            }]
        );
        assert_eq!(emulation_0_switch_data.branches().len(), 2);

        let emulation_branch_0_0 = emulation_0_switch_data.branches()[0];

        let ValueOrigin::Output {
            producer: emulation_0_0_load_node,
            output: 0,
        } = rvsdg[emulation_branch_0_0].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_0_0_load_data = rvsdg[emulation_0_0_load_node].expect_op_load();

        assert_eq!(
            emulation_0_0_load_data.value_inputs(),
            &[ValueInput::argument(ptr_ty, 0)]
        );

        let emulation_branch_0_1 = emulation_0_switch_data.branches()[1];

        let ValueOrigin::Output {
            producer: emulation_0_1_load_node,
            output: 0,
        } = rvsdg[emulation_branch_0_1].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_0_1_load_data = rvsdg[emulation_0_1_load_node].expect_op_load();

        assert_eq!(
            emulation_0_1_load_data.value_inputs(),
            &[ValueInput::argument(ptr_ty, 1)]
        );

        let emulation_branch_1 = emulation_switch_data.branches()[1];

        let ValueOrigin::Output {
            producer: emulation_1_load_node,
            output: 0,
        } = rvsdg[emulation_branch_1].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_1_load_data = rvsdg[emulation_1_load_node].expect_op_load();

        assert_eq!(
            emulation_1_load_data.value_inputs(),
            &[ValueInput::argument(ptr_ty, 3)]
        );

        assert_eq!(
            &rvsdg[ptr_0_node].expect_op_alloca().value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: outer_switch_node,
                    input: 2,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 2,
                }
            ],
            "the emulation node should have been added as a user to the first alloca node"
        );

        assert_eq!(
            &rvsdg[ptr_1_node].expect_op_alloca().value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: outer_switch_node,
                    input: 3,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 3,
                }
            ],
            "the emulation node should have been added as a user to the second alloca node"
        );

        assert_eq!(
            &rvsdg[ptr_2_node].expect_op_alloca().value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: outer_switch_node,
                    input: 4,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 4,
                }
            ],
            "the emulation node should have been added as a user to the third alloca node"
        );

        assert!(
            rvsdg[outer_switch_node].value_outputs()[0].users.is_empty(),
            "original switch node should no longer have any users"
        );
    }

    #[test]
    fn test_emulate_single_switch_output_store() {
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
                ret_ty: None,
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ptr_0 = rvsdg.add_op_alloca(region, TY_U32);
        let ptr_1 = rvsdg.add_op_alloca(region, TY_U32);

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(ptr_ty, ptr_0, 0),
                ValueInput::output(ptr_ty, ptr_1, 0),
            ],
            vec![ValueOutput::new(ptr_ty)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);

        rvsdg.reconnect_region_result(branch_0, 0, ValueOrigin::Argument(0));

        let branch_1 = rvsdg.add_switch_branch(switch_node);

        rvsdg.reconnect_region_result(branch_1, 0, ValueOrigin::Argument(1));

        let store_op = rvsdg.add_op_store(
            region,
            ValueInput::output(ptr_ty, switch_node, 0),
            ValueInput::argument(TY_U32, 1),
            StateOrigin::Argument,
        );

        let mut emulation_context = EmulationContext::new();

        emulation_context.emulate_op_store(&mut rvsdg, store_op);

        assert_eq!(rvsdg[region].value_arguments()[0].users.len(), 2);
        assert_eq!(
            rvsdg[region].value_arguments()[0].users[0],
            ValueUser::Input {
                consumer: switch_node,
                input: 0,
            }
        );

        let ValueUser::Input {
            consumer: emulation_switch_node,
            input: 0,
        } = rvsdg[region].value_arguments()[0].users[1]
        else {
            panic!(
                "the second user of the first argument should connect to the first input of a node"
            )
        };

        let emulation_switch_data = rvsdg[emulation_switch_node].expect_switch();

        assert_eq!(
            emulation_switch_data.value_inputs(),
            &[
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(ptr_ty, ptr_0, 0),
                ValueInput::output(ptr_ty, ptr_1, 0),
                ValueInput::argument(TY_U32, 1),
            ]
        );
        assert!(emulation_switch_data.value_outputs().is_empty());
        assert_eq!(emulation_switch_data.branches().len(), 2);

        let emulation_branch_0 = emulation_switch_data.branches()[0];

        assert_eq!(
            rvsdg[emulation_branch_0].value_arguments()[0].users.len(),
            1
        );
        assert_eq!(
            rvsdg[emulation_branch_0].value_arguments()[1].users.len(),
            0
        );
        assert_eq!(
            rvsdg[emulation_branch_0].value_arguments()[2].users.len(),
            1
        );

        let ValueUser::Input {
            consumer: emulation_0_store_node,
            input: 0,
        } = rvsdg[emulation_branch_0].value_arguments()[0].users[0]
        else {
            panic!(
                "the first user of the first argument should connect to the first input of a node"
            )
        };

        let emulation_0_store_data = rvsdg[emulation_0_store_node].expect_op_store();

        assert_eq!(
            emulation_0_store_data.value_input(),
            &ValueInput::argument(TY_U32, 2)
        );
        assert_eq!(
            rvsdg[emulation_branch_0].value_arguments()[2].users[0],
            ValueUser::Input {
                consumer: emulation_0_store_node,
                input: 1,
            }
        );

        let emulation_branch_1 = emulation_switch_data.branches()[1];

        assert_eq!(
            rvsdg[emulation_branch_1].value_arguments()[0].users.len(),
            0
        );
        assert_eq!(
            rvsdg[emulation_branch_1].value_arguments()[1].users.len(),
            1
        );
        assert_eq!(
            rvsdg[emulation_branch_1].value_arguments()[2].users.len(),
            1
        );

        let ValueUser::Input {
            consumer: emulation_1_store_node,
            input: 0,
        } = rvsdg[emulation_branch_1].value_arguments()[1].users[0]
        else {
            panic!(
                "the first user of the first argument should connect to the first input of a node"
            )
        };

        let emulation_1_store_data = rvsdg[emulation_1_store_node].expect_op_store();

        assert_eq!(
            emulation_1_store_data.value_input(),
            &ValueInput::argument(TY_U32, 2)
        );
        assert_eq!(
            rvsdg[emulation_branch_1].value_arguments()[2].users[0],
            ValueUser::Input {
                consumer: emulation_1_store_node,
                input: 1,
            }
        );

        assert_eq!(
            &rvsdg[region].value_arguments()[0].users,
            &thin_set![
                ValueUser::Input {
                    consumer: switch_node,
                    input: 0,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 0,
                }
            ],
            "the emulation node should have been added as a user to the first function argument"
        );

        assert_eq!(
            &rvsdg[region].value_arguments()[1].users,
            &thin_set![ValueUser::Input {
                consumer: emulation_switch_node,
                input: 3,
            }],
            "the store node should have been replaced with the emulation node as a user of the \
            second function argument"
        );

        assert_eq!(
            &rvsdg[ptr_0].expect_op_alloca().value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: switch_node,
                    input: 1,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 1,
                }
            ],
            "the emulation node should have been added as a user to the first alloca node"
        );

        assert_eq!(
            &rvsdg[ptr_1].expect_op_alloca().value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: switch_node,
                    input: 2,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 2,
                }
            ],
            "the emulation node should have been added as a user to the second alloca node"
        );

        assert!(
            rvsdg[switch_node].value_outputs()[0].users.is_empty(),
            "original switch node should no longer have any users"
        );
    }

    #[test]
    fn test_emulate_load_switch_output_reuse() {
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

        let array_ty = module.ty.register(TypeKind::Array {
            element_ty: TY_U32,
            count: 2,
            stride: 4,
        });
        let array_of_array_ty = module.ty.register(TypeKind::Array {
            element_ty: array_ty,
            count: 2,
            stride: 8,
        });
        let array_ptr_ty = module.ty.register(TypeKind::Ptr(array_ty));
        let array_of_array_ptr_ty = module.ty.register(TypeKind::Ptr(array_of_array_ty));
        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let array_alloca_node = rvsdg.add_op_alloca(region, array_of_array_ty);

        let switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(array_of_array_ptr_ty, array_alloca_node, 0),
                ValueInput::argument(TY_U32, 1),
                ValueInput::argument(TY_U32, 2),
            ],
            vec![ValueOutput::new(ptr_ty)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);

        let branch_0_const_node = rvsdg.add_const_u32(branch_0, 1);

        // Both index values are dynamic and originate inside the switch node for the first branch
        let branch_0_index_0 = rvsdg.add_op_binary(
            branch_0,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 1),
            ValueInput::output(TY_U32, branch_0_const_node, 0),
        );
        let branch_0_index_1 = rvsdg.add_op_binary(
            branch_0,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 2),
            ValueInput::output(TY_U32, branch_0_const_node, 0),
        );

        let branch_0_ptr_0_node = rvsdg.add_op_element_ptr(
            branch_0,
            ValueInput::argument(array_of_array_ptr_ty, 0),
            ValueInput::output(TY_U32, branch_0_index_0, 0),
        );
        let branch_0_ptr_1_node = rvsdg.add_op_element_ptr(
            branch_0,
            ValueInput::output(array_ptr_ty, branch_0_ptr_0_node, 0),
            ValueInput::output(TY_U32, branch_0_index_1, 0),
        );

        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: branch_0_ptr_1_node,
                output: 0,
            },
        );

        let branch_1 = rvsdg.add_switch_branch(switch_node);

        let branch_1_const_node = rvsdg.add_const_u32(branch_1, 1);

        // For the second branch, the first index is a dynamic value that originates inside the
        // switch node, the second index is a static value
        let branch_1_index_0 = rvsdg.add_op_binary(
            branch_1,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 1),
            ValueInput::output(TY_U32, branch_1_const_node, 0),
        );
        let branch_1_index_1 = rvsdg.add_const_u32(branch_1, 0);

        let branch_1_ptr_0_node = rvsdg.add_op_element_ptr(
            branch_1,
            ValueInput::argument(array_of_array_ptr_ty, 0),
            ValueInput::output(TY_U32, branch_1_index_0, 0),
        );
        let branch_1_ptr_1_node = rvsdg.add_op_element_ptr(
            branch_1,
            ValueInput::output(array_ptr_ty, branch_1_ptr_0_node, 0),
            ValueInput::output(TY_U32, branch_1_index_1, 0),
        );

        rvsdg.reconnect_region_result(
            branch_1,
            0,
            ValueOrigin::Output {
                producer: branch_1_ptr_1_node,
                output: 0,
            },
        );

        let load_op = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, switch_node, 0),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_op,
                output: 0,
            },
        );

        let mut emulation_context = EmulationContext::new();

        emulation_context.emulate_op_load(&mut rvsdg, load_op);

        let ValueOrigin::Output {
            producer: emulation_switch_node,
            output: 0,
        } = rvsdg[region].value_results()[0].origin
        else {
            panic!("the function result should connect to the first output of a node")
        };

        let emulation_switch_data = rvsdg[emulation_switch_node].expect_switch();

        assert_eq!(
            emulation_switch_data.value_inputs(),
            &[
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(array_of_array_ptr_ty, array_alloca_node, 0),
                ValueInput::output(TY_U32, switch_node, 1),
                ValueInput::output(TY_U32, switch_node, 2),
            ]
        );
        assert_eq!(
            emulation_switch_data.value_outputs(),
            &[ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Result(0)]
            }]
        );
        assert_eq!(emulation_switch_data.branches().len(), 2);

        let emulation_branch_0 = emulation_switch_data.branches()[0];

        let ValueOrigin::Output {
            producer: emulation_0_load_node,
            output: 0,
        } = rvsdg[emulation_branch_0].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_0_load_data = rvsdg[emulation_0_load_node].expect_op_load();

        let ValueOrigin::Output {
            producer: emulation_0_ep_1_node,
            output: 0,
        } = emulation_0_load_data.ptr_input().origin
        else {
            panic!("the load node in branch `0` should connect to the first output of a node")
        };

        let emulation_0_ep1_data = rvsdg[emulation_0_ep_1_node].expect_op_element_ptr();

        assert_eq!(
            emulation_0_ep1_data.index_input(),
            &ValueInput::argument(TY_U32, 2)
        );

        let ValueOrigin::Output {
            producer: emulation_0_ep_0_node,
            output: 0,
        } = emulation_0_ep1_data.ptr_input().origin
        else {
            panic!(
                "the second element-ptr node in branch `0` should connect to the first output of a \
                node"
            )
        };

        let emulation_0_ep0_data = rvsdg[emulation_0_ep_0_node].expect_op_element_ptr();

        assert_eq!(
            emulation_0_ep0_data.ptr_input(),
            &ValueInput::argument(array_of_array_ptr_ty, 0)
        );
        assert_eq!(
            emulation_0_ep0_data.index_input(),
            &ValueInput::argument(TY_U32, 1)
        );

        let emulation_branch_1 = emulation_switch_data.branches()[1];

        let ValueOrigin::Output {
            producer: emulation_1_load_node,
            output: 0,
        } = rvsdg[emulation_branch_1].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_1_load_data = rvsdg[emulation_1_load_node].expect_op_load();

        let ValueOrigin::Output {
            producer: emulation_1_ep1_node,
            output: 0,
        } = emulation_1_load_data.ptr_input().origin
        else {
            panic!("the load node in branch `0` should connect to the first output of a node")
        };

        let emulation_1_ep1_data = rvsdg[emulation_1_ep1_node].expect_op_element_ptr();

        let ValueOrigin::Output {
            producer: emulation_1_index_node,
            output: 0,
        } = emulation_1_ep1_data.index_input().origin
        else {
            panic!(
                "the index-input of the second element-ptr node in branch `1` should connect to \
            the first output of a node"
            )
        };

        let emulation_1_index_data = rvsdg[emulation_1_index_node].expect_const_u32();

        assert_eq!(emulation_1_index_data.value(), 0);

        let ValueOrigin::Output {
            producer: emulation_1_ep0_data,
            output: 0,
        } = emulation_1_ep1_data.ptr_input().origin
        else {
            panic!(
                "the ptr-input of the second element-ptr node in branch `1` should connect to \
            the first output of a node"
            )
        };

        let emulation_1_ep0_data = rvsdg[emulation_1_ep0_data].expect_op_element_ptr();

        assert_eq!(
            emulation_1_ep0_data.ptr_input(),
            &ValueInput::argument(array_of_array_ptr_ty, 0)
        );
        assert_eq!(
            emulation_1_ep0_data.index_input(),
            &ValueInput::argument(TY_U32, 1)
        );

        assert_eq!(
            &rvsdg[array_alloca_node]
                .expect_op_alloca()
                .value_output()
                .users,
            &thin_set![
                ValueUser::Input {
                    consumer: switch_node,
                    input: 1,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 1,
                }
            ],
            "the emulation node should have been added as a user to the alloca node"
        );

        assert!(
            rvsdg[switch_node].value_outputs()[0].users.is_empty(),
            "original switch node pointer output should no longer have any users"
        );

        assert_eq!(
            rvsdg[switch_node].value_outputs().len(),
            3,
            "two additional outputs should have been added to the original switch node"
        );
        assert_eq!(
            &rvsdg[switch_node].value_outputs()[1].users,
            &thin_set![ValueUser::Input {
                consumer: emulation_switch_node,
                input: 2
            }],
            "the first new switch output should connect to the third input of the emulation node"
        );
        assert_eq!(
            &rvsdg[switch_node].value_outputs()[2].users,
            &thin_set![ValueUser::Input {
                consumer: emulation_switch_node,
                input: 3
            }],
            "the second new switch output should connect to the fourth input of the emulation node"
        );
    }
}
