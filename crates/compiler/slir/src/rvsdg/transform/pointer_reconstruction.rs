use indexmap::IndexSet;
use rustc_hash::FxHashMap;

use crate::rvsdg::{
    Connectivity, Node, NodeKind, Region, Rvsdg, SimpleNode, ValueInput, ValueOrigin,
};
use crate::ty::{TY_PREDICATE, Type};

/// A pointer reconstruction program description.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PointerReconstructionInfo {
    pub pointer_ty: Type,
    pub reconstruction_root: PointerReconstructionNode,
}

/// A node in a pointer reconstruction program description.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum PointerReconstructionNode {
    Branching(BranchingNode),
    Leaf(LeafNode),
}

impl PointerReconstructionNode {
    pub fn propagate_sub_tree_inputs<F>(&mut self, mut record_leaf_inputs: F)
    where
        F: FnMut(&LeafNode, &mut IndexSet<ValueOrigin>),
    {
        if let PointerReconstructionNode::Branching(node) = self {
            node.propagate_sub_tree_inputs(&mut record_leaf_inputs);
        }
    }
}

impl From<BranchingNode> for PointerReconstructionNode {
    fn from(branching: BranchingNode) -> Self {
        PointerReconstructionNode::Branching(branching)
    }
}

impl From<LeafNode> for PointerReconstructionNode {
    fn from(leaf: LeafNode) -> Self {
        PointerReconstructionNode::Leaf(leaf)
    }
}

/// Represents a branching node in a variable pointer reconstruction program description.
///
/// This will translate to a [Switch] node when constructing the reconstruction program.
#[derive(Clone, Debug)]
pub struct BranchingNode {
    /// The [ValueOrigin] for the branch selector predicate, relative to the root reconstruction
    /// node.
    ///
    /// For the root reconstruction switch node, this directly identifies the branch selector input.
    /// For a switch node nested deeper in the pointer reconstruction program, this needs to be
    /// resolved against the parent [BranchingNode]'s [child_inputs].
    pub branch_selector: ValueOrigin,

    /// The branches of the node.
    ///
    /// Each branch will translate to its own [Region] in the [Switch] node when constructing the
    /// reconstruction program.
    pub branches: Vec<PointerReconstructionNode>,

    /// The complete set of [ValueOrigin]s used by the nodes of the sub-tree represented by this
    /// branching node.
    ///
    /// Note that all nodes present their input values as relative to the root of the entire
    /// pointer reconstruction tree. When building a reconstruction program, branching nodes will
    /// map to [Switch] nodes. For the eventual leaf reconstructions to have access to their
    /// required input values, these values must be routed into these [Switch] nodes as branch
    /// region arguments. Therefore, this [IndexSet] acts as a mapping from which a child node can
    /// look up the argument index that represents the value it needs.
    pub sub_tree_inputs: IndexSet<ValueOrigin>,
}

impl BranchingNode {
    pub fn propagate_sub_tree_inputs<F>(&mut self, mut record_leaf_inputs: F)
    where
        F: FnMut(&LeafNode, &mut IndexSet<ValueOrigin>),
    {
        self.propagate_sub_tree_inputs_internal(&mut record_leaf_inputs);
    }

    fn propagate_sub_tree_inputs_internal<F>(&mut self, record_leaf_inputs: &mut F)
    where
        F: FnMut(&LeafNode, &mut IndexSet<ValueOrigin>),
    {
        self.sub_tree_inputs.clear();

        for branch in &mut self.branches {
            match branch {
                PointerReconstructionNode::Branching(node) => {
                    node.propagate_sub_tree_inputs_internal(record_leaf_inputs);

                    self.sub_tree_inputs.insert(node.branch_selector);
                    self.sub_tree_inputs.extend(node.sub_tree_inputs.iter());
                }
                PointerReconstructionNode::Leaf(node) => {
                    record_leaf_inputs(node, &mut self.sub_tree_inputs);
                }
            }
        }
    }
}

impl PartialEq for BranchingNode {
    fn eq(&self, other: &Self) -> bool {
        self.branch_selector == other.branch_selector && self.branches == other.branches
    }
}

impl Eq for BranchingNode {}

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

/// Represents a leaf node in a variable pointer reconstruction program description.
///
/// A leaf node resolves to a pointer without any further branching.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LeafNode {
    /// A pointer to the root identifier of the pointer that this leaf reconstructs.
    pub root_pointer: ValueInput,

    /// A complete access chain that will refine the [root_identifier] into the pointer that this
    /// leave reconstructs.
    pub access_chain: Vec<Access>,
}

impl PointerReconstructionNode {
    pub fn visit_leaves<F>(&self, mut visitor: F)
    where
        F: FnMut(&LeafNode),
    {
        self.visit_leaves_internal(&mut visitor);
    }

    fn visit_leaves_internal<F>(&self, visitor: &mut F)
    where
        F: FnMut(&LeafNode),
    {
        match self {
            PointerReconstructionNode::Branching(node) => {
                for branch in &node.branches {
                    branch.visit_leaves_internal(visitor);
                }
            }
            PointerReconstructionNode::Leaf(node) => visitor(node),
        }
    }

    fn visit_leaves_mut(&mut self, visitor: &mut impl FnMut(&mut LeafNode)) {
        match self {
            PointerReconstructionNode::Branching(node) => {
                for branch in &mut node.branches {
                    branch.visit_leaves_mut(visitor);
                }
            }
            PointerReconstructionNode::Leaf(node) => visitor(node),
        }
    }

    fn visit_value_origins_mut_internal(
        &mut self,
        visitor: &mut impl FnMut(&mut ValueOrigin),
        visit_non_ptr_origins: bool,
        visit_ptr_origins: bool,
    ) {
        match self {
            PointerReconstructionNode::Branching(node) => {
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
            PointerReconstructionNode::Leaf(node) => {
                if visit_ptr_origins {
                    visitor(&mut node.root_pointer.origin);
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

#[derive(Clone, Copy, PartialEq, Eq, thiserror::Error, Debug)]
pub enum PointerReconstructionError {
    #[error("the `{0:?}` load operation needs to be promoted to value-flow")]
    NeedsPromotion(Node),
    #[error("loop-value `{loop_value}` for loop node `{loop_node:?}` needs to be normalized")]
    NeedsLoopPointerNormalization { loop_node: Node, loop_value: u32 },
}

pub type PointerReconstructionResult =
    Result<PointerReconstructionInfo, PointerReconstructionError>;

pub struct PointerReconstructionContext {
    pointer_reconstruction_info: FxHashMap<(Region, ValueOrigin), PointerReconstructionResult>,
    bounding_region: Option<Region>,
}

impl PointerReconstructionContext {
    pub fn new() -> Self {
        PointerReconstructionContext {
            pointer_reconstruction_info: FxHashMap::default(),
            bounding_region: None,
        }
    }

    /// Creates a new [PointerReconstructionContext] with an explicit bounding region.
    ///
    /// Arguments of the bounding region are always considered "root" pointers, and the analysis
    /// won't trace into any regions that are "outer" to the bounding region.
    ///
    /// Note that a function body region is always considered a bounding region, even when a
    /// [PointerReconstructionContext] is constructed via [PointerReconstructionContext::new].
    pub fn with_bounding_region(bounding_region: Region) -> Self {
        PointerReconstructionContext {
            pointer_reconstruction_info: FxHashMap::default(),
            bounding_region: Some(bounding_region),
        }
    }

    pub fn clear(&mut self) {
        self.pointer_reconstruction_info.clear();
    }

    pub fn resolve_reconstruction_info(
        &mut self,
        rvsdg: &mut Rvsdg,
        region: Region,
        pointer_origin: ValueOrigin,
    ) -> Result<&PointerReconstructionInfo, PointerReconstructionError> {
        if !self
            .pointer_reconstruction_info
            .contains_key(&(region, pointer_origin))
        {
            let info = self.create_reconstruction_info(rvsdg, region, pointer_origin);

            self.pointer_reconstruction_info
                .insert((region, pointer_origin), info);
        }

        self.pointer_reconstruction_info
            .get(&(region, pointer_origin))
            .unwrap()
            .as_ref()
            .map_err(|err| *err)
    }

    fn create_reconstruction_info(
        &mut self,
        rvsdg: &mut Rvsdg,
        region: Region,
        pointer_origin: ValueOrigin,
    ) -> PointerReconstructionResult {
        use NodeKind::*;
        use SimpleNode::*;

        match pointer_origin {
            ValueOrigin::Argument(arg) => self.create_argument_info(rvsdg, region, arg),
            ValueOrigin::Output { producer, output } => match rvsdg[producer].kind() {
                Switch(_) => self.create_switch_output_info(rvsdg, producer, output),
                Loop(_) => self.create_loop_output_info(rvsdg, producer, output),
                Simple(OpFieldPtr(_)) => self.create_field_ptr_info(rvsdg, producer),
                Simple(OpElementPtr(_)) => self.create_element_ptr_info(rvsdg, producer),
                Simple(OpAlloca(_)) => self.create_alloca_info(rvsdg, producer),
                Simple(ConstFallback(_)) => self.create_fallback_info(rvsdg, producer),
                Simple(OpOffsetSlice(_)) => self.create_offset_slice_info(rvsdg, producer),
                Simple(OpLoad(_)) => Err(PointerReconstructionError::NeedsPromotion(producer)),
                _ => unreachable!("node kind cannot output a value of a pointer type"),
            },
        }
    }

    fn create_argument_info(
        &mut self,
        rvsdg: &mut Rvsdg,
        region: Region,
        argument: u32,
    ) -> PointerReconstructionResult {
        let owner = rvsdg[region].owner();
        let kind = rvsdg[owner].kind();

        if Some(region) == self.bounding_region || matches!(kind, NodeKind::Function(_)) {
            // If we hit the bounding region, we consider the argument to be a root pointer. A
            // function body region is always considered a bounding region, even if no bounding
            // regio was explicitly specified (see `with_bounding_region`).

            let pointer_ty = rvsdg[region].value_arguments()[argument as usize].ty;

            return Ok(PointerReconstructionInfo {
                pointer_ty,
                reconstruction_root: LeafNode {
                    root_pointer: ValueInput::argument(pointer_ty, argument),
                    access_chain: vec![],
                }
                .into(),
            });
        }

        match kind {
            NodeKind::Switch(_) => self.create_switch_argument_info(rvsdg, owner, argument),
            NodeKind::Loop(_) => self.create_loop_argument_info(rvsdg, owner, argument),
            _ => unreachable!("node kind cannot own a region"),
        }
    }

    fn create_switch_argument_info(
        &mut self,
        rvsdg: &mut Rvsdg,
        switch_node: Node,
        argument: u32,
    ) -> PointerReconstructionResult {
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
            .resolve_reconstruction_info(rvsdg, outer_region, origin)?
            .clone();

        info.reconstruction_root
            .visit_value_origins_mut(&mut |outer_origin| {
                *outer_origin = resolve_inner_origin(rvsdg, switch_node, *outer_origin);
            });

        Ok(info)
    }

    fn create_loop_argument_info(
        &mut self,
        rvsdg: &mut Rvsdg,
        loop_node: Node,
        argument: u32,
    ) -> PointerReconstructionResult {
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

        let outer_region = rvsdg[loop_node].region();
        let loop_data = rvsdg[loop_node].expect_loop();
        let loop_region = loop_data.loop_region();

        let input = argument;
        let result = argument + 1;
        let result_origin = rvsdg[loop_region].value_results()[result as usize].origin;

        if result_origin != ValueOrigin::Argument(argument) {
            return Err(PointerReconstructionError::NeedsLoopPointerNormalization {
                loop_node,
                loop_value: argument,
            });
        }

        let input_origin = loop_data.value_inputs()[input as usize].origin;
        let mut info = self
            .resolve_reconstruction_info(rvsdg, outer_region, input_origin)?
            .clone();

        info.reconstruction_root
            .visit_value_origins_mut(&mut |outer_origin| {
                *outer_origin = resolve_inner_origin(rvsdg, loop_node, *outer_origin);
            });

        Ok(info)
    }

    fn create_switch_output_info(
        &mut self,
        rvsdg: &mut Rvsdg,
        switch_node: Node,
        output: u32,
    ) -> PointerReconstructionResult {
        // Helper function to find or create a non-pointer value origin outside the switch node that
        // represents the same value as the given `inner_origin` inside the switch node
        fn resolve_non_ptr_outer_origin(
            rvsdg: &mut Rvsdg,
            switch_node: Node,
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

            let inner_origin_ty = rvsdg.value_origin_ty(branch, inner_origin);

            // Next, try to find a result of the correct type that is currently connected to a
            // ConstFallback node. A fallback node indicates that any value of the correct type is
            // valid, we'll claim it and replace it with our value. We don't do this for "predicate"
            // type values; since we intend to normalize these later, they need more than just type
            // compatibility, and we conservatively assume such outputs can never be reused.
            if inner_origin_ty != TY_PREDICATE {
                let result_count = rvsdg[branch].value_results().len();

                for result in 0..result_count {
                    let result_input = rvsdg[branch].value_results()[result];

                    if result_input.ty == rvsdg.value_origin_ty(branch, inner_origin)
                        && let ValueOrigin::Output {
                            producer,
                            output: 0,
                        } = result_input.origin
                        && rvsdg[producer].is_const_fallback()
                    {
                        rvsdg.reconnect_region_result(branch, result as u32, inner_origin);

                        return ValueOrigin::Output {
                            producer: switch_node,
                            output: result as u32,
                        };
                    }
                }
            }

            // If we reach this point, then apparently we don't already have an appropriate output
            // available, so we'll create a new one. We'll connect the corresponding result for our
            // current branch to our inner_origin. For all other branches we'll connect a fallback
            // node to the result; if we later need reconstruction values in those branches, we
            // might be able to reuse such a result.
            let output = rvsdg.add_switch_output(switch_node, inner_origin_ty);
            let branch_count = rvsdg[switch_node].expect_switch().branches().len();
            for i in 0..branch_count {
                let b = rvsdg[switch_node].expect_switch().branches()[i];

                let origin = if b == branch {
                    inner_origin
                } else {
                    let fallback_node = rvsdg.add_const_fallback(b, inner_origin_ty);

                    ValueOrigin::Output {
                        producer: fallback_node,
                        output: 0,
                    }
                };

                rvsdg.reconnect_region_result(b, output, origin);
            }

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
        let branch_selector = switch_data.branch_selector().origin;
        let mut branches = Vec::with_capacity(branch_count);

        for i in 0..branch_count {
            let branch = rvsdg[switch_node].expect_switch().branches()[i];
            let origin = rvsdg[branch].value_results()[output as usize].origin;
            let mut info = self
                .resolve_reconstruction_info(rvsdg, branch, origin)?
                .clone();

            info.reconstruction_root
                .visit_non_ptr_origins_mut(&mut |inner_origin| {
                    *inner_origin =
                        resolve_non_ptr_outer_origin(rvsdg, switch_node, branch, *inner_origin);
                });

            info.reconstruction_root
                .visit_ptr_origins_mut(&mut |inner_origin| {
                    *inner_origin = resolve_ptr_outer_origin(rvsdg, switch_node, *inner_origin);
                });

            branches.push(info.reconstruction_root)
        }

        // Check if all branches produce identical pointer reconstructions. If all branches do
        // produce identical pointer reconstructions, we don't need to create a new branching node
        // to reconstruct the switch node's output.
        if branches.iter().all(|b| b == &branches[0]) {
            return Ok(PointerReconstructionInfo {
                pointer_ty,
                reconstruction_root: branches.pop().unwrap(),
            });
        }

        let branching_node = BranchingNode {
            branch_selector,
            branches,
            sub_tree_inputs: Default::default(),
        };

        Ok(PointerReconstructionInfo {
            pointer_ty,
            reconstruction_root: branching_node.into(),
        })
    }

    fn create_loop_output_info(
        &mut self,
        rvsdg: &mut Rvsdg,
        loop_node: Node,
        output: u32,
    ) -> PointerReconstructionResult {
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
            // available. We'll have to provide an input value (that will be unused), so we'll
            // create a new "fallback" value.

            let outer_region = rvsdg[loop_node].region();
            let ty = rvsdg.value_origin_ty(outer_region, inner_origin);
            let input_value = rvsdg.add_const_fallback(outer_region, ty);
            let output = rvsdg[loop_node].value_inputs().len() as u32;
            let result = output + 1;

            rvsdg.add_loop_input(loop_node, ValueInput::output(ty, input_value, 0));
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
        let argument = output;
        let result = output + 1;
        let result_origin = rvsdg[loop_region].value_results()[result as usize].origin;

        if result_origin != ValueOrigin::Argument(argument) {
            return Err(PointerReconstructionError::NeedsLoopPointerNormalization {
                loop_node,
                loop_value: argument,
            });
        }

        let mut info = self
            .resolve_reconstruction_info(rvsdg, loop_region, result_origin)?
            .clone();

        info.reconstruction_root
            .visit_non_ptr_origins_mut(&mut |inner_origin| {
                *inner_origin =
                    resolve_non_ptr_outer_origin(rvsdg, loop_node, loop_region, *inner_origin);
            });

        info.reconstruction_root
            .visit_ptr_origins_mut(&mut |inner_origin| {
                *inner_origin = resolve_ptr_outer_origin(rvsdg, loop_node, *inner_origin);
            });

        Ok(info)
    }

    fn create_field_ptr_info(
        &mut self,
        rvsdg: &mut Rvsdg,
        op_field_ptr: Node,
    ) -> PointerReconstructionResult {
        let region = rvsdg[op_field_ptr].region();
        let field_ptr = rvsdg[op_field_ptr].expect_op_field_ptr();
        let pointer_ty = field_ptr.value_output().ty;
        let ptr_origin = field_ptr.ptr_input().origin;
        let access = Access::Field(field_ptr.field_index());

        let mut info = self
            .resolve_reconstruction_info(rvsdg, region, ptr_origin)?
            .clone();

        info.reconstruction_root.visit_leaves_mut(&mut |leaf| {
            leaf.access_chain.push(access);
        });
        info.pointer_ty = pointer_ty;

        Ok(info)
    }

    fn create_element_ptr_info(
        &mut self,
        rvsdg: &mut Rvsdg,
        op_element_ptr: Node,
    ) -> PointerReconstructionResult {
        let region = rvsdg[op_element_ptr].region();
        let element_ptr = rvsdg[op_element_ptr].expect_op_element_ptr();
        let pointer_ty = element_ptr.value_output().ty;
        let ptr_origin = element_ptr.ptr_input().origin;
        let index_origin = element_ptr.index_input().origin;
        let access = Access::from_element_origin(rvsdg, index_origin);

        let mut info = self
            .resolve_reconstruction_info(rvsdg, region, ptr_origin)?
            .clone();

        info.reconstruction_root.visit_leaves_mut(&mut |leaf| {
            leaf.access_chain.push(access);
        });
        info.pointer_ty = pointer_ty;

        Ok(info)
    }

    fn create_alloca_info(
        &mut self,
        rvsdg: &Rvsdg,
        op_alloca: Node,
    ) -> PointerReconstructionResult {
        let pointer_ty = rvsdg[op_alloca].expect_op_alloca().value_output().ty;

        let reconstruction_root = LeafNode {
            root_pointer: ValueInput::output(pointer_ty, op_alloca, 0),
            access_chain: vec![],
        };

        Ok(PointerReconstructionInfo {
            pointer_ty,
            reconstruction_root: reconstruction_root.into(),
        })
    }

    fn create_fallback_info(
        &mut self,
        rvsdg: &Rvsdg,
        const_fallback: Node,
    ) -> PointerReconstructionResult {
        let pointer_ty = rvsdg[const_fallback].expect_const_fallback().ty();

        let reconstruction_root = LeafNode {
            root_pointer: ValueInput::output(pointer_ty, const_fallback, 0),
            access_chain: vec![],
        };

        Ok(PointerReconstructionInfo {
            pointer_ty,
            reconstruction_root: reconstruction_root.into(),
        })
    }

    fn create_offset_slice_info(
        &mut self,
        rvsdg: &mut Rvsdg,
        op_offset_slice: Node,
    ) -> PointerReconstructionResult {
        // We run the offset-slice-elaboration pass before we run any transforms that involve
        // pointer reconstruction. After offset-slice-elaboration, the refinement of the pointer
        // itself is no longer done by the OpOffsetSlice node anymore; instead, it happens at the
        // next OpElementPtr node, which now already incorporates the additional offset in its
        // `index_input`. We therefore don't need to record the OpOffsetSlice into the
        // PointerReconstructionInfo here, we simply pass forward the PointerReconstructionInfo for
        // the OpOffsetSlice's `ptr_input`.

        let region = rvsdg[op_offset_slice].region();
        let offset_slice = rvsdg[op_offset_slice].expect_op_offset_slice();
        let ptr_origin = offset_slice.ptr_input().origin;

        self.resolve_reconstruction_info(rvsdg, region, ptr_origin)
            .cloned()
    }
}
