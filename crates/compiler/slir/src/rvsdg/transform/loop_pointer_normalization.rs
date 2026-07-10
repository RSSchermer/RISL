//! Rewrites pointer-type loop-values that are not loop-invariant into loop-invariant loop-values.
//!
//! This is a preparatory transformation for [variable_pointer_emulation]. Emulating operations
//! on variable pointers that originate from loop-values comes with additional complexities. The
//! core problem is that the current pointer-value can depend on the previous iteration's value,
//! creating a recursive dependency which the simple [PointerReconstructionInfo] descriptions
//! cannot directly represent. Rather than increasing the complexity of the pointer-emulation
//! algorithm, we instead opt to first rewrite variable pointer-type loop-values into loop-invariant
//! loop-values first.
//!
//! There are two key observations/constraints that make this transform possible:
//!
//! 1. A variable pointer-type loop-value cannot be iteratively "self-refining". "Refining" here
//!    refers to the [OpElementPtr] and [OpFieldPtr] operations, which we'll collectively refer to
//!    as "pointer-refining" operations ([OpVariantPtr] is also a pointer-refining operation, but
//!    all [OpVariantPtr] nodes will have been eliminated by an earlier transform and are therefore
//!    not relevant here). Note that this is not a limitation imposed by this transform; it
//!    naturally follows from the type constraints of the SLIR IR. A loop-value's argument and
//!    result both represent the same value and thus must have the same type. A pointer-refining
//!    operation could only produce a value of the same type if the base type is self-recursive: a
//!    struct `A` contains a field of type `A`, or an array type where the element type is the array
//!    type itself (recursively). Such self-recursive types are already disallowed in SLIR (they
//!    cannot be represented in memory unless they are zero-sized, and we don't allow zero-sized
//!    types).
//! 2. A pointer may not outlive the scope of the value it points to. This means that all
//!    pointer-type values that flow to loop-region results must have originated from loop-region
//!    arguments. Note that such pointer values may be refined inside the loop-region if this does
//!    not violate the above constraint.
//!
//! From these constraints it follows that there is a fixed set of "shapes" that can represent a
//! pointer-type loop-value for all iterations of a loop; for any particular iteration, one of these
//! shapes can represent the value of the pointer-type loop-value. A "shape" here is a
//! loop-invariant pointer-type loop-value with zero or more pointer-refining operations. Note that
//! a shape may be parameterized by dynamic element index values. Since there is a fixed number of
//! shapes, we can enumerate them all and assign each shape a unique index. We can turn this index
//! into a loop-value and use it to select the right shape for the current iteration using a switch
//! node with a single output the type of which matches the loop-value's type; we'll refer to such
//! switch nodes as "shape selectors". To normalize a variable pointer-type loop-value we'll insert
//! such a shape selector in two places:
//!
//! - Inside the loop-region, to represent the pointer value for the current iteration. All users of
//!   the original variable pointer-type loop-region argument can be reconnected to this shape
//!   selector's output value.
//! - In the loop node's outer region ("after" the loop node), to represent the loop-value's output
//!   for the final iteration. All users of the original variable pointer-type loop-output can be
//!   reconnected to this shape selector's output value.
//!
//! One of the shapes is always the variable pointer-type loop-value's input value. To collect the
//! other possible shapes, we do a reverse-value-flow analysis starting from the variable
//! pointer-type loop-value's result. Reverse-value-flow always starts as a single path, but switch
//! nodes can cause the path to split into many paths (I'm ignoring loop nodes for now, I'll
//! explain why that's justified later). We only trace the pointer value; we don't trace the
//! reverse-value-flow of index values used for refining the pointer value. We stop tracing the
//! pointer value when we reach a loop argument (and per obserservation 2 above, all paths must
//! reach a loop-region argument).
//!
//! If the argument represents a loop-invariant pointer value, then we add a new shape to a set of
//! possible shapes. If the argument represents a dependency on another variable pointer-type
//! loop-value, then we first (recursively) perform the same reverse value-flow analysis on that
//! loop-value's result. How we treat the variable pointer dependency depends on whether we find a
//! dependency cycle:
//!
//! - If there is no dependency cycle, we first generate the shape selectors for the dependency. We
//!   then treat the dependency similarly to how we treat a loop-invariant dependency, except in
//!   that we use the output of the shape selectors as the base pointer for the shape, rather than
//!   the original pointer-type loop-value.
//! - If there is a dependency cycle, then we combine all shapes for all loop-values that are part
//!   of the cycle into a single set, such that we assign each shape a consistent index across all
//!   loop-values that are part of the cycle. Note we still create separate shape selectors and
//!   separate shape index loop-values (and access paramater loop-values) for each loop-value in the
//!   cycle. We'll refer to all loop-values that are part of the same dependency cycle as a
//!   "dependency group".
//!
//! The shape selectors we generate are not parameterized only by the shape index; if a shape is
//! refined with dynamic index values, then these index values also need to be passed into the
//! switch node so that we can instantiate such shapes. Therefore, if during reverse value-flow
//! tracing we encounter a refining operation that uses a dynamic index, we first make sure the
//! value is available in the loop-region. If the access happens inside a switch node, we make sure
//! the index value is passed out as a switch output value, recursivily for nested switch nodes,
//! until we reach the loop-region. When the value is available in the loop-region, we need to
//! ensure that the value will be available in the next iteration. Therefore, if the value is not
//! already used by a loop-region result, we add a new loop-value and connect its result to the
//! index value. The shape selectors then use the corresponding loop-region arguments/outputs as
//! input values.
//!
//! Now that we have created the shape selectors to represent the variable pointer-type
//! loop-value(s), we have to ensure that at the end of each iteration, each shape index result and
//! its associated access parameter results are assigned the appropriate values. For this we'll
//! add another structure to the loop-region that we'll refer to as the "shape encoder". We use
//! [PointerReconstructionInfo] to guide the construction of the shape encoder. This behaves
//! similarly to the way [variable_pointer_emulation] uses the [PointerReconstructionInfo] to create
//! emulation programs for load and store operations. However, instead of the leaves emulating a
//! load of store operation, we'll instead output the appropriate shape index and any dynamic access
//! parameters used by the shape. How we decide the shape index depends on whether the shape's
//! originating loop-argument is part of the same dependency group as the current loop-value:
//!
//! - If the originating loop-argument is not part of the same dependency group, then we set a
//!   constant shape index that depends on the position of the shape in the current dependency
//!   group's shape set. If the shape uses any dynamic index values, then we pass these out as the
//!   access parameters, in the order they are used to refine the shape. Note that multiple leaves
//!   may to the same shape.
//! - If the originating loop-argument is part of the same dependency group, then we set the shape
//!   index to the shape index loop-value that we've created earlier for the depended-upon
//!   pointer-type loop-value. We likewise set all access parameter loop-values to the corresponding
//!   access parameter loop-values for the depended-upon pointer-type loop-value. This works because
//!   of obeservation 1 (above): if a pointer-type loop-result originates from a circular
//!   dependency, then there cannot be any pointer refinements along the control-flow path to the
//!   originating loop-argument, as the circular dependency means that the value would eventually
//!   flow back onto itself, and any pointer refinements would result in a type violation; the next
//!   iteration's value for the current loop-value will necessarily be equal to the depended-upon
//!   loop-value's value in the current iteration. Since we make sure that all loop-values in the
//!   same dependency group share the exact same shape set with the exact same shape indices, we can
//!   simply pass the shape index value and the access parameter values forward.
//!
//! So far we've only addressed the case of control-flow flowing through switch nodes and have
//! ignored control-flow through loop nodes. We can handle control-flow through loop nodes simply
//! by first applying this transform to the relevant loop-values of the inner loop node first.
//! After applying the transform, the relevant loop-values will be loop-invariant, meaning we do
//! not need to analyze the control-flow inside the loop node; the value output by the loop node
//! will be identical to the value input into the loop node.
//!
//! [variable_pointer_emulation]: crate::rvsdg::transform::variable_pointer_emulation
use std::collections::VecDeque;
use std::mem;
use std::ops::Range;

use indexmap::{IndexMap, IndexSet};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::rvsdg::transform::pointer_reconstruction::{
    Access, BranchingNode, LeafNode, PointerReconstructionContext, PointerReconstructionError,
    PointerReconstructionNode,
};
use crate::rvsdg::visit::region_nodes::RegionNodesVisitor;
use crate::rvsdg::{
    Connectivity, Node, NodeKind, Region, Rvsdg, ValueInput, ValueOrigin, ValueOutput, visit,
};
use crate::ty::{Int, TY_PREDICATE, TY_U32, Type};

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct Job {
    loop_node: Node,
    loop_value: u32,
}

struct JobCollector<'a> {
    jobs: &'a mut VecDeque<Job>,
}

impl RegionNodesVisitor for JobCollector<'_> {
    fn visit_node(&mut self, rvsdg: &Rvsdg, node: Node) {
        if let NodeKind::Loop(data) = rvsdg[node].kind() {
            for (i, input) in data.value_inputs().iter().enumerate() {
                if rvsdg.ty().kind(input.ty).is_ptr() {
                    self.jobs.push_back(Job {
                        loop_node: node,
                        loop_value: i as u32,
                    });
                }
            }
        }

        visit::region_nodes::visit_node(self, rvsdg, node);
    }
}

pub struct LoopPointersNormalizer {
    queue_current: VecDeque<Job>,
    queue_next: VecDeque<Job>,
}

impl LoopPointersNormalizer {
    pub fn new() -> Self {
        Self {
            queue_current: VecDeque::new(),
            queue_next: VecDeque::new(),
        }
    }

    pub fn collect_jobs(&mut self, rvsdg: &Rvsdg, region: Region) {
        self.queue_current.clear();
        self.queue_next.clear();

        let mut collector = JobCollector {
            jobs: &mut self.queue_current,
        };

        collector.visit_region(rvsdg, region);
    }

    pub fn process_jobs(&mut self, rvsdg: &mut Rvsdg) -> bool {
        let mut made_change = false;

        // Queues are ordered from the "outside-in". We want to process loop nodes from the
        // "inside-out", so pwe process jobs in reverse order.
        while let Some(job) = self.queue_current.pop_back() {
            let mut normalizer = VariableLoopPointerNormalizer::new(rvsdg, job.loop_node);

            if let Ok(did_normalize) = normalizer.normalize_loop_value(rvsdg, job.loop_value) {
                made_change |= did_normalize;
            } else {
                self.queue_next.push_front(job);
            }
        }

        mem::swap(&mut self.queue_current, &mut self.queue_next);

        made_change
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct PointerShape {
    loop_value: u32,
    root_ptr_ty: Type,
    access_chain: Vec<Access>,
}

impl PointerShape {
    fn from_leaf_node(leaf_node: &LeafNode) -> Self {
        let ValueOrigin::Argument(loop_argument) = leaf_node.root_pointer.origin else {
            panic!(
                "expected a pointer-type loop-value to originate from a loop-region argument, \
                    but got: {:?}",
                leaf_node.root_pointer
            );
        };

        PointerShape {
            loop_value: loop_argument,
            root_ptr_ty: leaf_node.root_pointer.ty,
            access_chain: leaf_node.access_chain.clone(),
        }
    }

    fn param_count(&self) -> usize {
        self.access_chain
            .iter()
            .filter(|access| matches!(access, Access::DynamicElement(..)))
            .count()
    }
}

struct LoopValueInfo {
    ty: Type,
    reconstruction_root: Option<PointerReconstructionNode>,
    shapes: IndexSet<PointerShape>,
    dependencies: FxHashSet<u32>,
    dep_group: Option<usize>,
    initial_loop_value: u32,
    shape_index_loop_value: u32,
    access_param_loop_values: Range<u32>,

    /// Holds keys for the shape-selector switch nodes once they have been constructed.
    shape_selectors: Option<(Node, Node)>,
}

struct DepGraphVertex {
    loop_value: u32,
    visit_index: Option<usize>,
    low_link: usize,
    on_stack: bool,
}

impl DepGraphVertex {
    fn update_low_link(&mut self, linked: usize) {
        self.low_link = usize::min(self.low_link, linked);
    }
}

struct DepGroup {
    loop_values: FxHashSet<u32>,
    shapes: IndexSet<PointerShape>,
    param_count: usize,
}

fn is_loop_value_invariant(rvsdg: &Rvsdg, loop_region: Region, loop_value: u32) -> bool {
    let result_index = loop_value + 1;
    let result = rvsdg[loop_region].value_results()[result_index as usize];

    result.origin == ValueOrigin::Argument(loop_value)
}

pub struct VariableLoopPointerNormalizer {
    loop_node: Node,
    reconstruction_context: PointerReconstructionContext,
    loop_value_info: FxHashMap<u32, LoopValueInfo>,
    dep_groups: Vec<DepGroup>,
}

impl VariableLoopPointerNormalizer {
    pub fn new(rvsdg: &Rvsdg, loop_node: Node) -> Self {
        let loop_region = rvsdg[loop_node].expect_loop().loop_region();

        Self {
            loop_node,
            reconstruction_context: PointerReconstructionContext::with_bounding_region(loop_region),
            loop_value_info: FxHashMap::default(),
            dep_groups: Vec::new(),
        }
    }

    /// Normalizes a pointer-type loop-value and all of its dependencies if applicable.
    ///
    /// Does nothing if the loop-value was already loop-invariant.
    ///
    /// Return `true` if the pointer-type loop-value was normalized, or `false` if it was already
    /// loop-invariant.
    pub fn normalize_loop_value(
        &mut self,
        rvsdg: &mut Rvsdg,
        loop_value: u32,
    ) -> Result<bool, PointerReconstructionError> {
        self.reconstruction_context.clear();
        self.loop_value_info.clear();
        self.dep_groups.clear();

        let outer_region = rvsdg[self.loop_node].region();
        let loop_region = rvsdg[self.loop_node].expect_loop().loop_region();

        self.init_loop_value_info(rvsdg, loop_value)?;

        if self.loop_value_info.is_empty() {
            return Ok(false);
        }

        self.init_dep_groups();

        // Each loop-value in the analysis set should now be assigned to exactly one dependency
        // group. We'll need to know which dep-group each loop-value belongs to later, but rather
        // than search through the dep-groups every time, we'll iterate over all of the dep-groups
        // once and store the associated dep-group indices into each result-analysis entry.
        for (i, dep_group) in self.dep_groups.iter().enumerate() {
            for loop_value in &dep_group.loop_values {
                self.loop_value_info
                    .get_mut(loop_value)
                    .expect("should have been initialized above")
                    .dep_group = Some(i);
            }
        }

        // Create the new loop-values we use for cross-iteration pointer encoding. For each
        // variable pointer-type loop-value we'll create one new loop-value to store the shape
        // index. Additionally, if the shape-set for the loop-value contains shapes that use
        // runtime-dynamic index values for pointer refinement, then we'll add one additional
        // loop-value for each dynamic index value. Note that if multiple shapes use runtime-dynamic
        // index values, we'll only need as many new loop-values as there are dynamic index values
        // used in the shape that uses the most dynamic index values; there can be only one shape
        // that is active for a given iteration, so the shape index value already disambiguates how
        // these dynamic index values are used.
        for (loop_value, info) in &mut self.loop_value_info {
            let dep_group_index = info.dep_group.expect("should have been initialized above");
            let dep_group = &self.dep_groups[dep_group_index];
            let initial_shape_index = dep_group
                .shapes
                .get_index_of(&PointerShape {
                    loop_value: *loop_value,
                    root_ptr_ty: info.ty,
                    access_chain: vec![],
                })
                .expect(
                    "the shape-set should always contain a shape that represents a loop-value's \
                initial input",
                );

            let initial_shape_index = rvsdg.add_const_u32(outer_region, initial_shape_index as u32);
            let shape_index = rvsdg.add_loop_input(
                self.loop_node,
                ValueInput::output(TY_U32, initial_shape_index, 0),
            );

            for _ in 0..dep_group.param_count {
                let initial_param = rvsdg.add_const_u32(outer_region, 0);

                rvsdg.add_loop_input(self.loop_node, ValueInput::output(TY_U32, initial_param, 0));
            }

            info.shape_index_loop_value = shape_index;

            let access_params_start = shape_index + 1;
            let access_params_end = access_params_start + dep_group.param_count as u32;

            info.access_param_loop_values = access_params_start..access_params_end;
        }

        // Create a sequence of shape-selector construction that ensures that shape selectors that
        // depend on shape selectors for other loop-values are constructed in order.
        let mut shape_selector_construction_sequence = IndexSet::new();
        for loop_value in self.loop_value_info.keys() {
            self.sequence_shape_selector_construction(
                *loop_value,
                &mut shape_selector_construction_sequence,
            );
        }

        for loop_value in shape_selector_construction_sequence {
            self.construct_shape_selectors(rvsdg, loop_value);
        }

        // Construct the shape encoders
        let jobs = self.loop_value_info.keys().copied().collect::<Vec<_>>();
        for loop_value in jobs {
            self.construct_shape_encoder(rvsdg, loop_value);
        }

        // Finally, move over all users of the loop-argument and loop-output to the new shape
        // selectors and make the loop-values loop-invariant by reconnecting the result directly
        // to the argument.
        for (loop_value, info) in &self.loop_value_info {
            let (inner_selector, outer_selector) = info
                .shape_selectors
                .expect("should have been initialized during shape selector construction");

            rvsdg.reconnect_value_users(
                loop_region,
                ValueOrigin::Argument(*loop_value),
                ValueOrigin::Output {
                    producer: inner_selector,
                    output: 0,
                },
            );

            rvsdg.reconnect_value_users(
                outer_region,
                ValueOrigin::Output {
                    producer: self.loop_node,
                    output: *loop_value,
                },
                ValueOrigin::Output {
                    producer: outer_selector,
                    output: 0,
                },
            );

            rvsdg.reconnect_region_result(
                loop_region,
                loop_value + 1,
                ValueOrigin::Argument(*loop_value),
            );
        }

        Ok(true)
    }

    fn init_loop_value_info(
        &mut self,
        rvsdg: &mut Rvsdg,
        loop_value: u32,
    ) -> Result<(), PointerReconstructionError> {
        if self.loop_value_info.contains_key(&loop_value) {
            // We've already initialized the loop-value info for this loop-value.
            return Ok(());
        }

        let result_index = loop_value + 1;
        let loop_region = rvsdg[self.loop_node].expect_loop().loop_region();
        let result = rvsdg[loop_region].value_results()[result_index as usize];
        let loop_value_ty = result.ty;

        // Sanity check that the loop-value is indeed of pointer type.
        assert!(rvsdg.ty().kind(loop_value_ty).is_ptr());

        let is_invariant = result.origin == ValueOrigin::Argument(loop_value);

        if is_invariant {
            // If the loop-value is already loop-invariant, we don't need to do anything.
            return Ok(());
        }

        let reconstruction_info = self.reconstruction_context.resolve_reconstruction_info(
            rvsdg,
            loop_region,
            result.origin,
        )?;

        let mut shapes = IndexSet::new();

        reconstruction_info
            .reconstruction_root
            .visit_leaves(|leaf_node| {
                shapes.insert(PointerShape::from_leaf_node(leaf_node));
            });

        // Collect a set of all unique arguments used by our shape-set as our dependencies.
        let dependencies = FxHashSet::from_iter(shapes.iter().map(|shape| shape.loop_value));

        // The shape set should also include the current loop-value's argument itself, as this sets
        // the initial value for the first iteration. It may already be in the shape set if the
        // loop-value is self-dependent, but in case it's not, we add it here.
        shapes.insert(PointerShape {
            loop_value,
            root_ptr_ty: loop_value_ty,
            access_chain: vec![],
        });

        // Add a new loop-value to represent the initial value.
        let initial_input = rvsdg[self.loop_node].value_inputs()[loop_value as usize];
        let initial_value = rvsdg.add_loop_input(self.loop_node, initial_input);
        rvsdg.reconnect_region_result(
            loop_region,
            initial_value + 1,
            ValueOrigin::Argument(initial_value),
        );

        // Before we initialize the loop-value info for our dependencies, add an entry for the
        // current loop-value in the loop-value info map, so that if we encounter a dependency
        // cycle, we will not recurse infinitely.
        self.loop_value_info.insert(
            loop_value,
            LoopValueInfo {
                ty: loop_value_ty,
                reconstruction_root: Some(reconstruction_info.reconstruction_root.clone()),
                shapes,
                dependencies: FxHashSet::default(),
                dep_group: None,
                initial_loop_value: initial_value,
                shape_index_loop_value: 0,
                access_param_loop_values: Default::default(),
                shape_selectors: None,
            },
        );

        for dependency in &dependencies {
            self.init_loop_value_info(rvsdg, *dependency)?;
        }

        // Invariant loop-values do not need shape-selectors, and should therefore not be part of
        // the dependency graph that partitions variable loop-values into shape-selector groups.
        let dependencies = dependencies
            .into_iter()
            .filter(|dependency| !is_loop_value_invariant(rvsdg, loop_region, *dependency))
            .collect();

        self.loop_value_info
            .get_mut(&loop_value)
            .expect("should have been inserted above")
            .dependencies = dependencies;

        Ok(())
    }

    /// Performs a strongly-connected-components analysis on the loop-value dependency graph to
    /// partition loop-values into groups that reflect dependency cycles.
    fn init_dep_groups(&mut self) {
        let mut vertices = Vec::new();
        let mut loop_value_vertex_map = FxHashMap::default();

        for (i, loop_value) in self.loop_value_info.keys().enumerate() {
            vertices.push(DepGraphVertex {
                loop_value: *loop_value,
                visit_index: None,
                low_link: 0,
                on_stack: false,
            });
            loop_value_vertex_map.insert(*loop_value, i);
        }

        let mut dep_groups = Vec::new();

        let mut visit_index = 0;
        let mut stack = Vec::new();

        for index in 0..vertices.len() {
            if vertices[index].visit_index.is_none() {
                search_vertex(
                    index,
                    &self.loop_value_info,
                    &mut vertices,
                    &loop_value_vertex_map,
                    &mut visit_index,
                    &mut stack,
                    &mut dep_groups,
                );
            }
        }

        self.dep_groups = dep_groups;
    }

    fn sequence_shape_selector_construction(&self, loop_value: u32, sequence: &mut IndexSet<u32>) {
        if sequence.contains(&loop_value) {
            return;
        }

        let dep_group_index = self.loop_value_info[&loop_value]
            .dep_group
            .expect("should have been initialized before calling this function");
        let dep_group = &self.dep_groups[dep_group_index];

        // Ensure that any dependencies that require shape-selectors and are not part of the same
        // circular dependency group are sequenced before the current loop-value. This is because
        // for a dependency that is not part of the same dependency group, we want to use its shape
        // selector's output as an input to the current loop-value's shape selector.
        for shape in &dep_group.shapes {
            let requires_shape_selector = self.loop_value_info.contains_key(&shape.loop_value);
            let same_dep_group = dep_group.loop_values.contains(&shape.loop_value);

            if shape.loop_value != loop_value && requires_shape_selector && !same_dep_group {
                self.sequence_shape_selector_construction(shape.loop_value, sequence);
            }
        }

        sequence.insert(loop_value);
    }

    fn construct_shape_selectors(&mut self, rvsdg: &mut Rvsdg, loop_value: u32) {
        let dep_group_index = self.loop_value_info[&loop_value]
            .dep_group
            .expect("should have been initialized before calling this function");
        let dep_group = &self.dep_groups[dep_group_index];

        // Construct the shape-selector inside the loop-region to represent the variable pointer
        // value in the current iteration.
        let loop_region = rvsdg[self.loop_node].expect_loop().loop_region();
        let inner_selector =
            self.construct_shape_selector(rvsdg, loop_value, loop_region, |loop_value| {
                if let Some(info) = self.loop_value_info.get(&loop_value) {
                    if !dep_group.loop_values.contains(&loop_value) {
                        let (inner_selector, _) = info.shape_selectors.expect(
                            "construction sequencing should ensure this was initialized this earlier",
                        );

                        ValueOrigin::Output {
                            producer: inner_selector,
                            output: 0,
                        }
                    } else {
                        ValueOrigin::Argument(info.initial_loop_value)
                    }
                } else {
                    ValueOrigin::Argument(loop_value)
                }
            });

        // Construct the shape-selector in the outer region to represent the variable pointer value
        // after the final iteration.
        let outer_region = rvsdg[self.loop_node].region();
        let outer_selector =
            self.construct_shape_selector(rvsdg, loop_value, outer_region, |loop_value| {
                if let Some(info) = self.loop_value_info.get(&loop_value) {
                    if !dep_group.loop_values.contains(&loop_value) {
                        let (_, outer_selector) = info.shape_selectors.expect(
                            "construction sequencing should ensure this initialized this earlier",
                        );

                        ValueOrigin::Output {
                            producer: outer_selector,
                            output: 0,
                        }
                    } else {
                        ValueOrigin::Output {
                            producer: self.loop_node,
                            output: info.initial_loop_value,
                        }
                    }
                } else {
                    ValueOrigin::Output {
                        producer: self.loop_node,
                        output: loop_value,
                    }
                }
            });

        self.loop_value_info
            .get_mut(&loop_value)
            .unwrap()
            .shape_selectors = Some((inner_selector, outer_selector));
    }

    fn construct_shape_selector<F>(
        &self,
        rvsdg: &mut Rvsdg,
        loop_value: u32,
        region: Region,
        mut loop_value_origin: F,
    ) -> Node
    where
        F: FnMut(u32) -> ValueOrigin,
    {
        let info = &self.loop_value_info[&loop_value];
        let dep_group = info
            .dep_group
            .expect("should have been initialized before calling this function");
        let shapes = &self.dep_groups[dep_group].shapes;

        // Create a branch-selector based on the number of shapes, each assigned a consecutive
        // integer as its index. Note that a case-based branch-selector always has a default
        // branch, so we'll need one fewer case than the number of shapes.
        let max_shape_index = shapes.len() as u128 - 1;
        let cases = 0..max_shape_index;
        let shape_index_origin = loop_value_origin(info.shape_index_loop_value);
        let branch_selector = rvsdg.add_op_case_to_branch_selector(
            region,
            ValueInput {
                ty: TY_U32,
                origin: shape_index_origin,
            },
            Int::U32,
            cases,
        );

        // Construct a set of unique shape-roots. Shapes may share a root-pointer, with different
        // access chains. Our switch node will only need one input for each unique root-pointer.
        // We'll use an IndexMap to deduplicate the root-pointers and to assign each root-pointer a
        // unique index.
        let mut shape_roots: IndexMap<u32, Type> = IndexMap::default();
        shape_roots.extend(
            shapes
                .iter()
                .map(|shape| (shape.loop_value, shape.root_ptr_ty)),
        );

        // Now we'll construct the input list for the switch node. The first input is the branch
        // selector. Then follow the inputs for the set of unique root-pointers. Finally, a list
        // of dynamic access parameters for element accesses that use runtime-dynamic index values.
        let mut switch_inputs = vec![ValueInput::output(TY_PREDICATE, branch_selector, 0)];
        switch_inputs.extend(shape_roots.iter().map(|(loop_value, ty)| ValueInput {
            ty: *ty,
            origin: loop_value_origin(*loop_value),
        }));
        switch_inputs.extend(
            info.access_param_loop_values
                .clone()
                .map(|loop_value| ValueInput {
                    ty: TY_U32,
                    origin: loop_value_origin(loop_value),
                }),
        );

        // Construct the switch node. It will have a single output value that represents the
        // normalized pointer-type loop-value.
        let switch_node =
            rvsdg.add_switch(region, switch_inputs, vec![ValueOutput::new(info.ty)], None);

        let access_param_offset = shape_roots.len() as u32;

        for shape in shapes {
            let branch = rvsdg.add_switch_branch(switch_node);
            let root_arg = shape_roots
                .get_index_of(&shape.loop_value)
                .expect("the shape-roots map should contain all loop-values used by the shapes")
                as u32;

            let mut current_value = ValueInput::argument(shape.root_ptr_ty, root_arg);
            let mut current_access_param = access_param_offset;

            for access in &shape.access_chain {
                let node = match access {
                    Access::Field(field) => rvsdg.add_op_field_ptr(branch, current_value, *field),
                    Access::StaticElement(element) => {
                        let index = rvsdg.add_const_u32(branch, *element);

                        rvsdg.add_op_element_ptr(
                            branch,
                            current_value,
                            ValueInput::output(TY_U32, index, 0),
                        )
                    }
                    Access::DynamicElement(_) => {
                        // Note that we don't actually use the origin stored in the Access variant;
                        // this still refers to the origin returned by the pointer-reconstruction
                        // algorithm. For shape-selector switches, access params are passed as
                        // consecutive arguments that start after all root-pointer arguments, one
                        // per dynamic element access.

                        let node = rvsdg.add_op_element_ptr(
                            branch,
                            current_value,
                            ValueInput::argument(TY_U32, current_access_param),
                        );

                        current_access_param += 1;

                        node
                    }
                };

                let output_ty = rvsdg[node].value_outputs()[0].ty;

                current_value = ValueInput::output(output_ty, node, 0);
            }

            rvsdg.reconnect_region_result(branch, 0, current_value.origin)
        }

        switch_node
    }

    fn construct_shape_encoder(&mut self, rvsdg: &mut Rvsdg, loop_value: u32) {
        let loop_region = rvsdg[self.loop_node].expect_loop().loop_region();
        let info = self
            .loop_value_info
            .get_mut(&loop_value)
            .expect("should have been initialized");
        let dep_group = &self.dep_groups[info.dep_group.unwrap()];

        let mut reconstruction_root = info
            .reconstruction_root
            .take()
            .expect("should have been initialized");

        // Specify the values used by each of the leaf nodes and propagate them up through the
        // parent branching nodes. This will help us route values from the loop-region to leaf
        // branches nested inside the shape-encoder sub-graph.
        reconstruction_root.propagate_sub_tree_inputs(|leaf, input_set| {
            let ValueOrigin::Argument(dep_value) = leaf.root_pointer.origin else {
                panic!(
                    "expected all pointers in variable loop-pointer system to originate from \
                        arguments"
                );
            };

            if dep_group.loop_values.contains(&dep_value) {
                // The dependency is part of the same dependency cycle as the current loop-value.
                // We set the serialization of the loop-value for the next iteration, to the
                // serialization of the dependency for the current iteration. Note that this also
                // works if the dependency is the current loop-value itself.

                let dep_info = &self.loop_value_info[&dep_value];

                input_set.insert(ValueOrigin::Argument(dep_info.shape_index_loop_value));

                for access_param in dep_info.access_param_loop_values.clone() {
                    input_set.insert(ValueOrigin::Argument(access_param));
                }
            } else {
                // The dependency is not a circular dependency. We'll need to forward the
                // runtime-dynamic index values to the next iteration. Note that we don't need the
                // base pointer value to be available, as we'll use a constant `u32` value to
                // represent it via its shape index.

                for access in &leaf.access_chain {
                    if let Access::DynamicElement(origin) = access {
                        input_set.insert(*origin);
                    }
                }
            }
        });

        let info = &self.loop_value_info[&loop_value];

        let mut builder = ShapeEncoderBuilder {
            dep_group,
            loop_value_info: &self.loop_value_info,
            rvsdg,
            outer_region: loop_region,
            output_count: info.access_param_loop_values.len() + 1,
        };

        let outputs = builder.visit_reconstruction_node(loop_region, &reconstruction_root, None);

        let shape_index_result = info.shape_index_loop_value + 1;

        rvsdg.reconnect_region_result(loop_region, shape_index_result, outputs[0]);

        for (i, output) in info
            .access_param_loop_values
            .clone()
            .zip(outputs[1..].iter())
        {
            rvsdg.reconnect_region_result(loop_region, i + 1, *output);
        }
    }
}

struct ShapeEncoderBuilder<'a> {
    dep_group: &'a DepGroup,
    loop_value_info: &'a FxHashMap<u32, LoopValueInfo>,
    rvsdg: &'a mut Rvsdg,
    outer_region: Region,
    output_count: usize,
}

impl ShapeEncoderBuilder<'_> {
    fn visit_reconstruction_node(
        &mut self,
        region: Region,
        node: &PointerReconstructionNode,
        input_mapping: Option<&IndexSet<ValueOrigin>>,
    ) -> Vec<ValueOrigin> {
        match node {
            PointerReconstructionNode::Branching(node) => {
                self.visit_branching_node(region, node, input_mapping)
            }
            PointerReconstructionNode::Leaf(node) => {
                self.visit_leaf_node(region, node, input_mapping)
            }
        }
    }

    fn visit_branching_node(
        &mut self,
        region: Region,
        branching_node: &BranchingNode,
        input_mapping: Option<&IndexSet<ValueOrigin>>,
    ) -> Vec<ValueOrigin> {
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

        let mut value_inputs = Vec::with_capacity(branching_node.sub_tree_inputs.len() + 1);

        // Connect the first input to the branch selector predicate.
        value_inputs.push(resolve_input(branching_node.branch_selector, TY_PREDICATE));

        // Connect inputs for the emulation values required by the branching node's child nodes.
        for child_input in branching_node.sub_tree_inputs.iter().copied() {
            let ty = self.rvsdg.value_origin_ty(self.outer_region, child_input);

            value_inputs.push(resolve_input(child_input, ty));
        }

        let value_outputs = vec![ValueOutput::new(TY_U32); self.output_count];
        let switch_node = self
            .rvsdg
            .add_switch(region, value_inputs, value_outputs, None);

        for node in &branching_node.branches {
            let branch = self.rvsdg.add_switch_branch(switch_node);

            let outputs =
                self.visit_reconstruction_node(branch, node, Some(&branching_node.sub_tree_inputs));

            let mut i = 0;

            while i < outputs.len() {
                self.rvsdg
                    .reconnect_region_result(branch, i as u32, outputs[i]);

                i += 1;
            }

            // Shapes may not all use the same number of runtime-dynamic access parameters. Our
            // switch nodes account for the max number of access parameters among all shapes. If
            // this causes any results to remain unconnected, we connect them to fallback values.
            // These may either be optimized away or replaced by a concrete value (typically `0`)
            // later.
            while i < self.output_count {
                let fallback = self.rvsdg.add_const_fallback(branch, TY_U32);

                self.rvsdg.reconnect_region_result(
                    branch,
                    i as u32,
                    ValueOrigin::Output {
                        producer: fallback,
                        output: 0,
                    },
                );

                i += 1;
            }
        }

        (0..self.output_count)
            .map(|i| ValueOrigin::Output {
                producer: switch_node,
                output: i as u32,
            })
            .collect()
    }

    fn visit_leaf_node(
        &mut self,
        region: Region,
        node: &LeafNode,
        input_mapping: Option<&IndexSet<ValueOrigin>>,
    ) -> Vec<ValueOrigin> {
        let resolve_origin = |origin| {
            if let Some(input_mapping) = input_mapping {
                let argument = input_mapping
                    .get_index_of(&origin)
                    .expect("input mapping was not correctly constructed");

                ValueOrigin::Argument(argument as u32)
            } else {
                origin
            }
        };

        let ValueOrigin::Argument(loop_value) = node.root_pointer.origin else {
            panic!(
                "expected all pointers in variable loop-pointer system to originate from arguments"
            );
        };

        if self.dep_group.loop_values.contains(&loop_value) {
            let loop_value_info = &self.loop_value_info[&loop_value];

            let mut outputs = Vec::new();

            let shape_origin = resolve_origin(ValueOrigin::Argument(
                loop_value_info.shape_index_loop_value,
            ));

            outputs.push(shape_origin);

            for access_param in loop_value_info.access_param_loop_values.clone() {
                outputs.push(resolve_origin(ValueOrigin::Argument(access_param)));
            }

            outputs
        } else {
            let shape_index = self
                .dep_group
                .shapes
                .get_index_of(&PointerShape::from_leaf_node(node))
                .expect("every leaf-node should have registered an associated shape");

            let shape_value = self.rvsdg.add_const_u32(region, shape_index as u32);
            let mut outputs = Vec::new();

            outputs.push(ValueOrigin::Output {
                producer: shape_value,
                output: 0,
            });

            for access in &node.access_chain {
                if let Access::DynamicElement(origin) = access {
                    outputs.push(resolve_origin(*origin));
                }
            }

            outputs
        }
    }
}

fn search_vertex(
    index: usize,
    result_analysis: &FxHashMap<u32, LoopValueInfo>,
    vertices: &mut [DepGraphVertex],
    loop_value_vertex_map: &FxHashMap<u32, usize>,
    visit_index: &mut usize,
    stack: &mut Vec<usize>,
    dep_groups: &mut Vec<DepGroup>,
) {
    let vertex_visit_index = *visit_index;

    *visit_index += 1;

    let vertex = &mut vertices[index];
    let loop_value = vertex.loop_value;

    vertex.visit_index = Some(vertex_visit_index);
    vertex.low_link = vertex_visit_index;

    stack.push(index);
    vertex.on_stack = true;

    for dep in &result_analysis[&loop_value].dependencies {
        let dep_index = *loop_value_vertex_map
            .get(dep)
            .expect("invalid loop-value handle");

        if vertices[dep_index].visit_index.is_none() {
            search_vertex(
                dep_index,
                result_analysis,
                vertices,
                loop_value_vertex_map,
                visit_index,
                stack,
                dep_groups,
            );

            vertices[index].update_low_link(vertices[dep_index].low_link);
        } else if vertices[dep_index].on_stack {
            vertices[index].update_low_link(vertices[dep_index].low_link);
        }
    }

    if vertex_visit_index == vertices[index].low_link {
        // Create the dependency group. We also combine all shapes for all group members into a
        // single shared shape-set for the group.

        let mut dep_group = FxHashSet::default();
        let mut shapes = IndexSet::default();
        let mut param_count = 0;

        while let Some(vertex_index) = stack.pop() {
            vertices[vertex_index].on_stack = false;

            let loop_value = vertices[vertex_index].loop_value;

            dep_group.insert(loop_value);

            for shape in &result_analysis[&loop_value].shapes {
                param_count = usize::max(param_count, shape.param_count());

                shapes.insert(shape.clone());
            }

            if vertex_index == index {
                break;
            }
        }

        dep_groups.push(DepGroup {
            loop_values: dep_group,
            shapes,
            param_count,
        });
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::rvsdg::{StateOrigin, ValueUser};
    use crate::ty::{Int, TY_DUMMY, TY_PREDICATE, TY_PTR_U32, TY_U32, TypeKind};
    use crate::{BinaryOperator, FnArg, FnSig, Function, Module, Symbol};

    // Note: the order of shapes is not meant to be explicitly defined, as long as the shape
    // selectors and shape encoders are internally consistent. However, writing tests that are
    // compatible with an arbitrary shape order makes for complicated tests. Since the shape order
    // we produce is deterministic, these tests hard-code the shape order.
    //
    // It is perfectly acceptable to change the code above to produce a different shape order. In
    // that case, these tests should be expected to fail. You will then have to adjust the tests
    // to match the new shape order.

    #[test]
    fn test_normalize_with_variable_pointer_selected_from_invariant_pointers() {
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
                ret_ty: None,
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ptr_0 = rvsdg.add_op_alloca(region, TY_U32);
        let ptr_1 = rvsdg.add_op_alloca(region, TY_U32);
        let value_2 = rvsdg.add_const_u32(region, 2);

        let (loop_node, loop_region) = rvsdg.add_loop(
            region,
            vec![
                ValueInput::output(TY_PTR_U32, ptr_0, 0),
                ValueInput::output(TY_PTR_U32, ptr_1, 0),
                ValueInput::output(TY_PTR_U32, ptr_0, 0),
            ],
            Some(StateOrigin::Argument),
        );

        let case = rvsdg.add_const_u32(loop_region, 0);
        let selector = rvsdg.add_op_case_to_branch_selector(
            loop_region,
            ValueInput::output(TY_U32, case, 0),
            Int::U32,
            [0],
        );
        let switch_node = rvsdg.add_switch(
            loop_region,
            vec![
                ValueInput::output(TY_PREDICATE, selector, 0),
                ValueInput::argument(TY_PTR_U32, 0),
                ValueInput::argument(TY_PTR_U32, 1),
            ],
            vec![ValueOutput::new(TY_PTR_U32)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);
        rvsdg.reconnect_region_result(branch_0, 0, ValueOrigin::Argument(0));

        let branch_1 = rvsdg.add_switch_branch(switch_node);
        rvsdg.reconnect_region_result(branch_1, 0, ValueOrigin::Argument(1));

        let reentry_predicate = rvsdg.add_const_bool(loop_region, false);
        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: reentry_predicate,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(loop_region, 1, ValueOrigin::Argument(0));
        rvsdg.reconnect_region_result(loop_region, 2, ValueOrigin::Argument(1));

        rvsdg.reconnect_region_result(
            loop_region,
            3,
            ValueOrigin::Output {
                producer: switch_node,
                output: 0,
            },
        );

        let inner_store_node = rvsdg.add_op_store(
            loop_region,
            ValueInput::argument(TY_PTR_U32, 2),
            ValueInput::output(TY_U32, case, 0),
            StateOrigin::Argument,
        );
        let outer_store_node = rvsdg.add_op_store(
            region,
            ValueInput::output(TY_PTR_U32, loop_node, 2),
            ValueInput::output(TY_U32, value_2, 0),
            StateOrigin::Node(loop_node),
        );

        let mut normalizer = VariableLoopPointerNormalizer::new(&rvsdg, loop_node);

        normalizer.normalize_loop_value(&mut rvsdg, 2);

        let loop_data = rvsdg[loop_node].expect_loop();

        assert_eq!(loop_data.value_inputs().len(), 5);

        let ValueOrigin::Output {
            producer: initial_shape_index,
            output: 0,
        } = loop_data.value_inputs()[4].origin
        else {
            panic!("the added shape-index loop input should be initialized by a constant")
        };

        assert_eq!(rvsdg[initial_shape_index].expect_const_u32().value(), 2);

        assert_eq!(
            rvsdg[loop_region].value_results()[3].origin,
            ValueOrigin::Argument(2),
            "the normalized pointer loop-value should have become loop-invariant"
        );

        let inner_store_data = rvsdg[inner_store_node].expect_op_store();

        // Verify the inner shape selector

        let ValueOrigin::Output {
            producer: inner_shape_selector,
            output: 0,
        } = inner_store_data.value_inputs()[0].origin
        else {
            panic!("the inner store should use the inner shape-selector output")
        };

        let inner_selector_data = rvsdg[inner_shape_selector].expect_switch();

        assert_eq!(inner_selector_data.value_inputs().len(), 4);

        let ValueOrigin::Output {
            producer: inner_shape_branch_selector,
            output: 0,
        } = inner_selector_data.value_inputs()[0].origin
        else {
            panic!("the inner shape-selector should be driven by a branch selector")
        };

        let inner_shape_branch_selector_data =
            rvsdg[inner_shape_branch_selector].expect_op_case_to_branch_selector();

        assert_eq!(inner_shape_branch_selector_data.cases(), [0, 1]);

        assert_eq!(
            inner_selector_data.value_inputs()[1],
            ValueInput::argument(TY_PTR_U32, 0)
        );
        assert_eq!(
            inner_selector_data.value_inputs()[2],
            ValueInput::argument(TY_PTR_U32, 1)
        );
        assert_eq!(
            inner_selector_data.value_inputs()[3],
            ValueInput::argument(TY_PTR_U32, 3)
        );

        assert_eq!(inner_selector_data.branches().len(), 3);
        assert_eq!(
            rvsdg[inner_selector_data.branches()[0]].value_results()[0].origin,
            ValueOrigin::Argument(0)
        );
        assert_eq!(
            rvsdg[inner_selector_data.branches()[1]].value_results()[0].origin,
            ValueOrigin::Argument(1)
        );
        assert_eq!(
            rvsdg[inner_selector_data.branches()[2]].value_results()[0].origin,
            ValueOrigin::Argument(2)
        );

        assert_eq!(inner_selector_data.value_outputs()[0].ty, TY_PTR_U32);
        assert!(
            inner_selector_data.value_outputs()[0]
                .users
                .contains(&ValueUser::Input {
                    consumer: inner_store_node,
                    input: 0,
                })
        );

        // Verify the outer shape selector

        let outer_store_data = rvsdg[outer_store_node].expect_op_store();

        let ValueOrigin::Output {
            producer: outer_shape_selector,
            output: 0,
        } = outer_store_data.value_inputs()[0].origin
        else {
            panic!("the outer store should use the outer shape-selector output")
        };

        let outer_selector_data = rvsdg[outer_shape_selector].expect_switch();

        assert_eq!(outer_selector_data.value_inputs().len(), 4);

        let ValueOrigin::Output {
            producer: outer_shape_branch_selector,
            output: 0,
        } = outer_selector_data.value_inputs()[0].origin
        else {
            panic!("the outer shape-selector should be driven by a branch selector")
        };

        let outer_shape_branch_selector_data =
            rvsdg[outer_shape_branch_selector].expect_op_case_to_branch_selector();

        assert_eq!(outer_shape_branch_selector_data.cases(), [0, 1]);

        assert_eq!(
            outer_selector_data.value_inputs()[1],
            ValueInput::output(TY_PTR_U32, loop_node, 0)
        );
        assert_eq!(
            outer_selector_data.value_inputs()[2],
            ValueInput::output(TY_PTR_U32, loop_node, 1)
        );
        assert_eq!(
            outer_selector_data.value_inputs()[3],
            ValueInput::output(TY_PTR_U32, loop_node, 3)
        );

        assert_eq!(outer_selector_data.branches().len(), 3);
        assert_eq!(
            rvsdg[outer_selector_data.branches()[0]].value_results()[0].origin,
            ValueOrigin::Argument(0)
        );
        assert_eq!(
            rvsdg[outer_selector_data.branches()[1]].value_results()[0].origin,
            ValueOrigin::Argument(1)
        );
        assert_eq!(
            rvsdg[outer_selector_data.branches()[2]].value_results()[0].origin,
            ValueOrigin::Argument(2)
        );

        assert_eq!(outer_selector_data.value_outputs()[0].ty, TY_PTR_U32);
        assert!(
            outer_selector_data.value_outputs()[0]
                .users
                .contains(&ValueUser::Input {
                    consumer: outer_store_node,
                    input: 0,
                })
        );

        // Verify the shape encoder

        let ValueOrigin::Output {
            producer: shape_encoder,
            output: 0,
        } = rvsdg[loop_region].value_results()[5].origin
        else {
            panic!("the shape-index result should be produced by the shape encoder")
        };
        let shape_encoder_data = rvsdg[shape_encoder].expect_switch();

        assert_eq!(shape_encoder_data.value_inputs().len(), 1);
        assert_eq!(
            shape_encoder_data.value_inputs()[0],
            ValueInput::output(TY_PREDICATE, selector, 0)
        );
        assert_eq!(shape_encoder_data.value_outputs()[0].ty, TY_U32);
        assert!(
            shape_encoder_data.value_outputs()[0]
                .users
                .contains(&ValueUser::Result(5))
        );
        assert_eq!(shape_encoder_data.branches().len(), 2);

        let ValueOrigin::Output {
            producer: shape_index_0,
            output: 0,
        } = rvsdg[shape_encoder_data.branches()[0]].value_results()[0].origin
        else {
            panic!("the first shape-encoder branch should return a constant shape index")
        };

        assert_eq!(rvsdg[shape_index_0].expect_const_u32().value(), 0);

        let ValueOrigin::Output {
            producer: shape_index_1,
            output: 0,
        } = rvsdg[shape_encoder_data.branches()[1]].value_results()[0].origin
        else {
            panic!("the second shape-encoder branch should return a constant shape index")
        };

        assert_eq!(rvsdg[shape_index_1].expect_const_u32().value(), 1);
    }

    #[test]
    fn test_normalize_with_dynamically_refined_variable_pointer() {
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
                ret_ty: None,
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ty_array_u32 = rvsdg.ty().register(TypeKind::Array {
            element_ty: TY_U32,
            count: 10,
            stride: 4,
        });
        let ty_ptr_array_u32 = rvsdg.ty().register(TypeKind::Ptr(ty_array_u32));

        let array_ptr = rvsdg.add_op_alloca(region, ty_array_u32);
        let initial_index = rvsdg.add_const_u32(region, 0);

        let initial_ptr = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(ty_ptr_array_u32, array_ptr, 0),
            ValueInput::output(TY_U32, initial_index, 0),
        );

        let (loop_node, loop_region) = rvsdg.add_loop(
            region,
            vec![
                ValueInput::output(ty_ptr_array_u32, array_ptr, 0),
                ValueInput::output(TY_U32, initial_index, 0),
                ValueInput::output(TY_PTR_U32, initial_ptr, 0),
            ],
            Some(StateOrigin::Argument),
        );

        let store_val = rvsdg.add_const_u32(loop_region, 42);
        let inner_store_node = rvsdg.add_op_store(
            loop_region,
            ValueInput::argument(TY_PTR_U32, 2),
            ValueInput::output(TY_U32, store_val, 0),
            StateOrigin::Argument,
        );

        let next_ptr = rvsdg.add_op_element_ptr(
            loop_region,
            ValueInput::argument(ty_ptr_array_u32, 0),
            ValueInput::argument(TY_U32, 1),
        );

        let index_increment = rvsdg.add_const_u32(loop_region, 1);
        let next_index = rvsdg.add_op_binary(
            loop_region,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 1),
            ValueInput::output(TY_U32, index_increment, 0),
        );

        let reentry_predicate = rvsdg.add_const_bool(loop_region, false);

        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: reentry_predicate,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(loop_region, 1, ValueOrigin::Argument(0));
        rvsdg.reconnect_region_result(
            loop_region,
            2,
            ValueOrigin::Output {
                producer: next_index,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            loop_region,
            3,
            ValueOrigin::Output {
                producer: next_ptr,
                output: 0,
            },
        );

        let mut normalizer = VariableLoopPointerNormalizer::new(&rvsdg, loop_node);

        normalizer.normalize_loop_value(&mut rvsdg, 2);

        let loop_data = rvsdg[loop_node].expect_loop();

        // 3 original, 3 new (initial pointer value, shape index, access param)
        assert_eq!(loop_data.value_inputs().len(), 6);

        // Verify that the original variable pointer-type loop-value is now invariant
        assert_eq!(
            rvsdg[loop_region].value_results()[3].origin,
            ValueOrigin::Argument(2),
        );

        // Verify the shape selector

        let store_ptr = rvsdg[inner_store_node].expect_op_store().value_inputs()[0].origin;
        let ValueOrigin::Output {
            producer: shape_selector,
            ..
        } = store_ptr
        else {
            panic!("store_ptr should be an output from the shape selector")
        };

        let selector_data = rvsdg[shape_selector].expect_switch();

        assert_eq!(
            selector_data.value_inputs().len(),
            4,
            "shape selector should have 4 inputs (selector, array_ptr, initial_ptr, access_param)"
        );

        let ValueOrigin::Output {
            producer: branch_selector,
            ..
        } = selector_data.value_inputs()[0].origin
        else {
            panic!("selector should be driven by a branch selector")
        };

        assert_eq!(
            rvsdg[branch_selector]
                .expect_op_case_to_branch_selector()
                .value_input()
                .origin,
            ValueOrigin::Argument(4),
        );

        assert_eq!(
            selector_data.value_inputs()[1],
            ValueInput::argument(ty_ptr_array_u32, 0)
        );
        assert_eq!(
            selector_data.value_inputs()[2],
            ValueInput::argument(TY_PTR_U32, 3)
        );
        assert_eq!(
            selector_data.value_inputs()[3],
            ValueInput::argument(TY_U32, 5)
        );

        assert_eq!(selector_data.branches().len(), 2,);

        let branch_0 = selector_data.branches()[0];
        let branch_0_output = rvsdg[branch_0].value_results()[0].origin;

        let ValueOrigin::Output {
            producer: branch_0_node,
            ..
        } = branch_0_output
        else {
            panic!("branch 0 should return an output from an OpElementPtr")
        };

        let element_ptr_0 = rvsdg[branch_0_node].expect_op_element_ptr();

        assert_eq!(
            element_ptr_0.value_inputs()[0].origin,
            ValueOrigin::Argument(0)
        );
        assert_eq!(
            element_ptr_0.value_inputs()[1].origin,
            ValueOrigin::Argument(2)
        );

        let branch_1 = selector_data.branches()[1];

        assert_eq!(
            rvsdg[branch_1].value_results()[0].origin,
            ValueOrigin::Argument(1)
        );

        // Verify initial pointer value initialization
        assert_eq!(
            loop_data.value_inputs()[3].origin,
            ValueOrigin::Output {
                producer: initial_ptr,
                output: 0
            },
        );

        // Verify shape index initialization
        let ValueOrigin::Output {
            producer: initial_shape_index_node,
            ..
        } = loop_data.value_inputs()[4].origin
        else {
            panic!("initial shape index should be a constant")
        };
        assert_eq!(
            rvsdg[initial_shape_index_node].expect_const_u32().value(),
            1,
        );

        // Verify shape encoding

        let ValueOrigin::Output {
            producer: encoder_node,
            ..
        } = rvsdg[loop_region].value_results()[5].origin
        else {
            panic!("shape encoder for v2 should be a node")
        };
        assert_eq!(rvsdg[encoder_node].expect_const_u32().value(), 0,);

        assert_eq!(
            rvsdg[loop_region].value_results()[6].origin,
            ValueOrigin::Argument(1),
        );
    }

    #[test]
    fn test_normalize_with_swapping_variable_pointers() {
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
                ret_ty: None,
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ptr_0 = rvsdg.add_op_alloca(region, TY_U32);
        let ptr_1 = rvsdg.add_op_alloca(region, TY_U32);

        let (loop_node, loop_region) = rvsdg.add_loop(
            region,
            vec![
                ValueInput::output(TY_PTR_U32, ptr_0, 0),
                ValueInput::output(TY_PTR_U32, ptr_1, 0),
            ],
            Some(StateOrigin::Argument),
        );

        let reentry_predicate = rvsdg.add_const_bool(loop_region, false);
        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: reentry_predicate,
                output: 0,
            },
        );

        let store_val = rvsdg.add_const_u32(loop_region, 42);
        let inner_store_node_0 = rvsdg.add_op_store(
            loop_region,
            ValueInput::argument(TY_PTR_U32, 0),
            ValueInput::output(TY_U32, store_val, 0),
            StateOrigin::Argument,
        );
        let inner_store_node_1 = rvsdg.add_op_store(
            loop_region,
            ValueInput::argument(TY_PTR_U32, 1),
            ValueInput::output(TY_U32, store_val, 0),
            StateOrigin::Argument,
        );

        // Swap loop values
        rvsdg.reconnect_region_result(loop_region, 1, ValueOrigin::Argument(1));
        rvsdg.reconnect_region_result(loop_region, 2, ValueOrigin::Argument(0));

        let mut normalizer = VariableLoopPointerNormalizer::new(&rvsdg, loop_node);

        // Normalizing 0 should also normalize 1 due to a (circular) dependency
        normalizer.normalize_loop_value(&mut rvsdg, 0);

        let loop_data = rvsdg[loop_node].expect_loop();

        // 2 original, 2 initial pointer values, 2 shape indices
        assert_eq!(loop_data.value_inputs().len(), 6);

        // Verify both original loop-values became loop-invariant
        assert_eq!(
            rvsdg[loop_region].value_results()[1].origin,
            ValueOrigin::Argument(0)
        );
        assert_eq!(
            rvsdg[loop_region].value_results()[2].origin,
            ValueOrigin::Argument(1)
        );

        // Verify shape index initialization

        let ValueOrigin::Output {
            producer: init_0_index_node,
            ..
        } = loop_data.value_inputs()[4].origin
        else {
            panic!("shape index 0 should be initialized by a constant")
        };
        assert_eq!(rvsdg[init_0_index_node].expect_const_u32().value(), 0);

        let ValueOrigin::Output {
            producer: init_1_index_node,
            ..
        } = loop_data.value_inputs()[5].origin
        else {
            panic!("shape index 1 should be initialized by a constant")
        };
        assert_eq!(rvsdg[init_1_index_node].expect_const_u32().value(), 1);

        // Verify the first shape selector

        let store_0_ptr = rvsdg[inner_store_node_0].expect_op_store().value_inputs()[0].origin;

        let ValueOrigin::Output {
            producer: selector_0,
            ..
        } = store_0_ptr
        else {
            panic!("store_0_ptr should be an output from the shape selector")
        };

        let selector_0_data = rvsdg[selector_0].expect_switch();

        let ValueOrigin::Output {
            producer: branch_selector_0,
            ..
        } = selector_0_data.value_inputs()[0].origin
        else {
            panic!("selector_0 should be driven by a branch selector")
        };
        assert_eq!(
            rvsdg[branch_selector_0]
                .expect_op_case_to_branch_selector()
                .value_input()
                .origin,
            ValueOrigin::Argument(4)
        );

        assert_eq!(selector_0_data.value_inputs().len(), 3);
        assert_eq!(
            selector_0_data.value_inputs()[1],
            ValueInput::argument(TY_PTR_U32, 2)
        );
        assert_eq!(
            selector_0_data.value_inputs()[2],
            ValueInput::argument(TY_PTR_U32, 3)
        );

        assert_eq!(
            rvsdg[selector_0_data.branches()[0]].value_results()[0].origin,
            ValueOrigin::Argument(0)
        );
        assert_eq!(
            rvsdg[selector_0_data.branches()[1]].value_results()[0].origin,
            ValueOrigin::Argument(1)
        );

        // Verify the second shape selector

        let store_1_ptr = rvsdg[inner_store_node_1].expect_op_store().value_inputs()[0].origin;
        let ValueOrigin::Output {
            producer: selector_1,
            ..
        } = store_1_ptr
        else {
            panic!("store_1_ptr should be an output from the shape selector")
        };
        let selector_1_data = rvsdg[selector_1].expect_switch();

        let ValueOrigin::Output {
            producer: branch_selector_1,
            ..
        } = selector_1_data.value_inputs()[0].origin
        else {
            panic!("selector_1 should be driven by a branch selector")
        };
        assert_eq!(
            rvsdg[branch_selector_1]
                .expect_op_case_to_branch_selector()
                .value_input()
                .origin,
            ValueOrigin::Argument(5)
        );

        assert_eq!(selector_1_data.value_inputs().len(), 3);
        assert_eq!(
            selector_1_data.value_inputs()[1],
            ValueInput::argument(TY_PTR_U32, 2)
        );
        assert_eq!(
            selector_1_data.value_inputs()[2],
            ValueInput::argument(TY_PTR_U32, 3)
        );

        assert_eq!(
            rvsdg[selector_1_data.branches()[0]].value_results()[0].origin,
            ValueOrigin::Argument(0)
        );
        assert_eq!(
            rvsdg[selector_1_data.branches()[1]].value_results()[0].origin,
            ValueOrigin::Argument(1)
        );

        // Verify shape encoding

        assert_eq!(
            rvsdg[loop_region].value_results()[5].origin,
            ValueOrigin::Argument(5)
        );
        assert_eq!(
            rvsdg[loop_region].value_results()[6].origin,
            ValueOrigin::Argument(4)
        );
    }
}
