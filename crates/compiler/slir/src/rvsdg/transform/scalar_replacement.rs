//! Replaces alloca nodes of aggregate types with multiple alloca nodes, one for each of the
//! aggregate's parts.
//!
//! We'll also refer to this as "splitting" an alloca node.
//!
//! The following types are considered aggregate types: struct types, array types, slice (unsized
//! array) types, and enum types. We don't support alloca nodes for unsized types, so we will not
//! have to consider the case of splitting a slice alloca node; we only ever split alloca nodes
//! for struct types, array types, and enum types.
//!
//! For a struct type, its parts are its fields. For an array type, its parts are its elements. For
//! an enum type, its parts are first its discriminant, then a sequence of each of its variants.
//! Note that a part of an aggregate may itself be an aggregate: a struct field may itself be a
//! struct/array/enum; the elements of an array may be structs/arrays/enums; the variants of an enum
//! are currently always structs. If the splitting of an alloca results in new "part" alloca nodes
//! that are also of aggregate types, then we may split these "part" alloca nodes recursively. We
//! don't do this all at once; rather, scalar-replacement uses a queue system and "part" allocas
//! for aggregate types will be pushed onto the end of the queue to await their turn.
//!
//! # Optimization and legalization
//!
//! Why split alloca nodes? Scalar replacement of aggregates is a common pass in optimizing
//! compilers. While scalar replacement in itself does not typically result in a performance
//! improvement, other compiler passes are much more straightforward to implement against simple
//! scalar values. As such, scalar replacement is a key transformation for enabling some of the
//! other passes in the compiler pipeline.
//!
//! However, for this compiler specifically, we may also need to split alloca nodes for reasons of
//! "legalization". Our primary output target, WGSL, does not allow aggregate types to contain
//! pointers (or "pointer-likes" such as texture handles): their parts may not be of a pointer type
//! or an aggregate type that contains pointer types (recursively). If we encounter such an alloca
//! node, splitting is not optional; we must find a way to split out the pointers it contains into
//! scalar values.
//!
//! # Value-flow splitting
//!
//! We cannot simply split alloca nodes in isolation. Alloca nodes have a single output value that
//! represents a pointer to its data. We split the alloca with the goal of eliminating the original
//! aggregate alloca node and replacing it with its "part" alloca nodes. We must then somehow adjust
//! the users of the original alloca node's output value so that they no longer need to consume the
//! output of the original alloca node, directly or indirectly: the original alloca node's output
//! value must become "userless" (except for [OpOffsetSlice] nodes as we'll expand on below) and its
//! users must be adjusted to now use the "part" alloca output values.
//!
//! If an [OpStore] node was a user of the alloca node's output value, then we replace the [OpStore]
//! node with multiple [OpStore] nodes: one for each part. An [OpStore] node does not have an output
//! value, so no further splitting is required.
//!
//! If an [OpLoad] node was a user of the alloca node's output value, then we replace the [OpLoad]
//! node with multiple [OpLoad] nodes: one for each part. However, the original [OpLoad] node *does*
//! have a value output; it represents a copy of the aggregate value contained in the alloca node.
//! Further splitting is now required: to eliminate the alloca node, we must eliminate all its
//! users, which means eliminating the original [OpLoad] node, which means the original [OpLoad]
//! node must be made "user-less" and its users must be adjusted to now use the outputs of the new
//! "part" [OpLoad] nodes.
//!
//! In general, we cannot simply split the original alloca node or its direct consumers; we have to
//! split the entire value-flow that originates from that alloca node. If a "split value" flows into
//! a [Loop] node, then we must split that loop-input and its corresponding argument, result and
//! output. If a "split value" flows into a [Switch] node, then we must split that switch-input and
//! its corresponding argument, and if it continues to flow to a branch result, then we must split
//! that branch-result and its corresponding output.
//!
//! However, not all node kinds that may consume a pointer to an aggregate value also output a
//! pointer to an aggregate. As discussed above [OpStore] has no output value at all, so no
//! downstream adjustments will need to be made. [OpFieldPtr], [OpElementPtr], [OpExtractField],
//! [OpExtractElement], [OpDiscriminantPtr] and [OpVariantPtr] all output a pointer to a part of the
//! aggregate; we'll call these "part-selector" nodes. These may all be adjusted by connecting their
//! users directly to the appropriate split part value and removing the part-selector node. As the
//! element index for [OpElementPtr] and may be runtime-dynamic, we introduce a new [Switch] node to
//! select the appropriate part, using the index value as the [Switch] node's branch selector. An
//! [OpGetDiscriminant] node may be replaced by an [OpLoad] node that uses the discriminant-part
//! value of an enum alloca's value-flow. An [OpSetDiscriminant] may be replaced by an [OpStore]
//! node that uses the discriminant-part value of an enum alloca's value-flow. All of these node
//! kinds act as terminators of the alloca's value-flow; no further down-stream splitting is
//! required after one of these node kinds.
//!
//! [OpOffsetSlice] and [OpGetSliceOffset] nodes form an exception. We don't split these nodes and
//! instead keep them attached to the original value-flow from the alloca node. We do this because
//! we don't keep track of the slice offset as a separate value, the value-flow chain itself
//! represents the current slice offset. This will be adjusted in the offset-slice-replacement pass,
//! but this pass runs after the memory-promotion-and-legalization pass (of which scalar-replacement
//! is a part) as it relies on variable-pointer-emulation having completed. Since
//! [OpGetSliceOffset] would act as a terminator anyway, this case is simple. [OpOffsetSlice],
//! however, may have users that need to be adjusted. We simply adjust these users up around the
//! [OpOffsetSlice] node, to its input, as if the [OpOffsetSlice] node does not exist. Its purpose
//! is now only to track the offset and no longer to provide the slice pointer value. This means
//! that the original alloca node we're trying to split does not become userless in this case and
//! cannot yet be removed. It will become userless after the offset-slice-replacement pass and can
//! then be removed by a dead-connectible-elimination pass.
//!
//! # Splitting switch node results and output
//!
//! Switch nodes may have (and typically do have) multiple branches. If a result for one branch
//! region needs to be split, then the results for all other branch regions will also need to be
//! split. This makes switch node output splitting a complex task that requires careful handling.
//!
//! There are two cases we handle separately. This first is the case where the type of the output
//! that is to be split is a struct type, an array type, or an enum type. In this case, we know
//! exactly what parts to split the output into, and consequently, what parts to split each of the
//! corresponding branch results into. We add the new "part outputs" to the switch node. This
//! creates corresponding "part results" for each of the branches, which will initially connect to
//! placeholder values. We'll have to replace the placeholder connections with actual connections.
//!
//! There is always one branch that will be the initial branch that triggers the switch node output
//! split. For this branch, part values will already be available, so we can connect the new branch
//! results to their corresponding part values. For the other branches we can use part-selector
//! nodes to provide values for the new part-results:
//!
//! - For a struct value we can add a series of [OpExtractField] nodes, one for each field.
//! - For a pointer-to-a-struct value we can add a series of [OpFieldPtr] nodes, one for each field.
//! - For an array value we can add a series of [OpExtractElement] nodes, one for each element.
//! - For a pointer-to-an-array value we can add a series of [OpElementPtr] nodes, one for each
//!   element.
//! - For a pointer-to-an-enum value we can add an [OpDiscriminantPtr] node for the discriminant
//!   part and a series of [OpVariantPtr] nodes for each of the variant parts.
//!
//! If a later alloca replacement ends up replacing the original value in one of the other branches,
//! then these part-selector nodes will be removed as described above. Note that not all branches
//! necessarily need to end up splitting the original value for that branch: a branch's aggregate
//! value need not originate from an alloca node at all; it may originate from a global binding.
//! This is not a problem, and we don't need to do any special analysis to detect this case, since
//! at no point did we leave any of the branches in an invalid state.
//!
//! The second case is where the switch node output type is a slice pointer. How many parts do we
//! split the switch output into? For the various branches the split value may (and typically will)
//! originate from different originating alloca nodes, that may all have different array lengths.
//! Moreover, the split value may also originate from global bindings, where the array length may
//! be very large and may even be runtime-dynamic (for storage bindings).
//!
//! Fortunately, a constraint on global-binding values helps us avoid the latter scenario: global
//! binding values may not contain pointer values. Though the originating arrays for the various
//! result values associated with a single slice output value may all have different lengths, they
//! must all have the same element type. For element types that don't contain pointer types, scalar
//! replacement is never required for legalization. Conversely, if scalar replacement would be
//! required for the legalization of an alloca node, then we know there can never be a downstream
//! switch output for which another branch originates from a global binding; pointer-containing
//! results always originate from an alloca node, for all branches. For now, we opt to leverage this
//! constraint by never splitting array alloca nodes if they don't contain pointers values, and thus
//! splitting is not required for legalization. This ensures that we never have to split switch
//! output values for which one or more branches originate from global bindings; for all branches
//! the result is always guaranteed to originate from an alloca node. (We may be able to add
//! analysis to detect switch output values for which all results originate from alloca nodes, even
//! if the values don't contain pointers, for potential optimization purposes.)
//!
//! We are then assured to be able to resolve the length of the originating array for each of the
//! branch results. The number of parts to split the output (and each of the results) into is then
//! chosen to be the "max" of these originating array lengths. Rather than run a value-flow analysis
//! to resolve these lengths, we instead adopt a lazy approach. We keep processing alloca nodes for
//! splitting. When the splitting of the value-flow of an alloca node would require us to split
//! a switch branch result, rather than attempt to proceed with a split immediately, we instead
//! "pause" the splitting of that value-flow path by inserting a [Reaggregation] node. This node
//! kind recombines the split values into a single aggregate again, so that we may connect the
//! original branch result to this [Reaggregation] node and keep it unsplit for the time being. We
//! then add the corresponding switch output to a marker table, where we keep track of the number of
//! branches that have pending [Reaggregation] nodes. As we proceed to process more alloca nodes,
//! we'll also insert such [Reaggregation] nodes for the other branches, incrementing the branch
//! count in the marker table each time. When the branch count in the marker table becomes equal to
//! the total branch count of the switch node, we know the switch output is ready to be split. At
//! this point we split the switch output and remove each branch's [Reaggregation] node,
//! reconnecting their inputs to the newly created result values. We then proceed with processing
//! the consumers of the switch output.
//!
//! There is one final problem to address. Since each of the branches may have originating arrays of
//! different lengths, they consequently may have varying numbers of part values. If a branch's part
//! count is equal to the max part count (and thus equal to the number of parts that we split the
//! output/branch-results into), then each new result-part has an obvious corresponding input-part.
//! However, for branches with fewer part values, we're left with result-parts that have no
//! corresponding input-part. What input do we provide for such result-parts? Accessing
//! out-of-bounds array elements is Undefined Behavior. Therefore, any input value of the correct
//! type would be a valid input value. Since all parts of an array have the same type, we opt to
//! simply connect all remaining input-less result-parts to the last element part of the array. It
//! is then up to valid (UB-free) programs to (runtime-dynamically) ensure (e.g., with bounds
//! checks) that values flowing from such result-parts are never actually accessed. In RISL, it
//! would be impossible to write a successfully compiling program that accesses such values without
//! using `unsafe` code (e.g., `slice::get_unchecked`).

use std::collections::VecDeque;
use std::ops::Deref;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::rvsdg::analyse::element_index::ElementIndex;
use crate::rvsdg::transform::enum_replacement::replace_enum_alloca;
use crate::rvsdg::visit::value_flow::ValueFlowVisitor;
use crate::rvsdg::{
    Connectivity, Node, NodeKind, OpAlloca, OpLoad, Region, Rvsdg, SimpleNode, StateOrigin,
    ValueInput, ValueOrigin, ValueOutput, ValueUser, visit,
};
use crate::ty::{TY_PREDICATE, TY_U32, Type, TypeKind, TypeRegistry};
use crate::{Function, Module};

enum SwitchOutputSplitKind {
    Struct,
    Array,
}

/// Whether the given type must be split for reasons of legalization.
///
/// Currently, types that contain pointers must be split as our primary target (WGSL) does not
/// support pointers in aggregate types. Types that contain enums must also be split, as enums are
/// not supported at all in our targets, and splitting is the primary strategy to eliminate enums.
fn ty_must_split(ty_registry: &TypeRegistry, ty: Type) -> bool {
    match ty_registry.kind(ty).deref() {
        TypeKind::Ptr(_) | TypeKind::Enum(_) => true,
        TypeKind::Struct(data) => data
            .fields
            .iter()
            .any(|field| ty_must_split(ty_registry, field.ty)),
        TypeKind::Array { element_ty, .. } => ty_must_split(ty_registry, *element_ty),
        TypeKind::Slice { element_ty, .. } => ty_must_split(ty_registry, *element_ty),
        _ => false,
    }
}

/// Whether an alloca for the given type would be a candidate for splitting.
///
/// Struct and enum types are always candidates for splitting. Array types are only candidates if
/// they contain a type that must be split (see [ty_must_split]). See the "Splitting switch node
/// results and output" section of the module-level documentation for details on why we're
/// conservative when it comes to splitting arrays.
fn alloca_ty_is_candidate(ty_registry: &TypeRegistry, ty: Type) -> bool {
    match ty_registry.kind(ty).deref() {
        TypeKind::Struct(_) | TypeKind::Enum(_) => true,
        TypeKind::Array { element_ty, .. } => ty_must_split(ty_registry, *element_ty),
        _ => false,
    }
}

/// Collects all [OpAlloca] nodes of aggregate types in a region and all sub-regions (e.g. a switch
/// node branch region) into a queue of candidates for scalar replacement.
///
/// Note that this does not yet make any decisions about whether we should perform a scalar
/// replacement transform on a given [OpAlloca] node, this requires further analysis.
struct CandidateAllocaCollector<'a, 'b> {
    rvsdg: &'a Rvsdg,
    candidates: &'b mut VecDeque<Node>,
}

impl CandidateAllocaCollector<'_, '_> {
    fn visit_region(&mut self, region: Region) {
        for node in self.rvsdg[region].nodes() {
            self.visit_node(*node);
        }
    }

    fn visit_node(&mut self, node: Node) {
        match self.rvsdg[node].kind() {
            NodeKind::Simple(SimpleNode::OpAlloca(op)) => {
                if alloca_ty_is_candidate(&self.rvsdg.ty(), op.ty()) {
                    self.candidates.push_back(node);
                }
            }
            NodeKind::Switch(n) => {
                for branch in n.branches() {
                    self.visit_region(*branch);
                }
            }
            NodeKind::Loop(n) => self.visit_region(n.loop_region()),
            _ => {}
        }
    }
}

fn collect_candidate_allocas(rvsdg: &Rvsdg, candidates: &mut VecDeque<Node>, region: Region) {
    CandidateAllocaCollector { rvsdg, candidates }.visit_region(region);
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AnalysisResult {
    Replace,
    NeedsPromotionPass,
    Ignore,
}

/// Analyzes an [OpAlloca] of an aggregate value can be replaced by its parts now, after memory
/// promotion, or should be ignored entirely and never replaced.
///
/// # Escape analysis
///
/// An [OpAlloca] of an aggregate (a struct or array) "escapes" if:
///
/// - The alloca pointer, or the output of an [OpLoad] on the alloca pointer, is passed whole as a
///   call argument input to an [OpCall] node.
/// - The output of an [OpLoad] on the alloca pointer is returned as a result from the local
///   function region.
///
/// For the [OpAlloca] to be found to escape, it must be the whole pointer to the full aggregate, or
/// the whole unsplit loaded value, that is passed to an [OpCall] node or returned as a result.
/// Passing or returning sub-elements of the aggregate, obtained via e.g. an [OpElementPtr], does
/// not constitute an escape, as in these cases scalar replacement will only require local
/// modifications (the [OpElementPtr] can be adjusted such that any [OpCall] user or result user
/// can remain unchanged).
///
/// # Stored-value analysis
///
/// While we can replace cases where a pointer to an alloca of aggregate is used as the "pointer"
/// input to on [OpStore], if it is used as the "value" input to an [OpStore] we cannot. However,
/// such [OpStore] nodes may be "promoted" away by a [memory_promotion_and_legalization] pass, so
/// the [OpAlloca] may be retried later.
struct AggregateAllocaAnalyzer {
    visited: FxHashSet<(Region, ValueUser)>,
    was_loaded: bool,
    has_nonlocal_use: bool,
    is_stored_value: bool,
}

impl AggregateAllocaAnalyzer {
    fn new() -> Self {
        Self {
            visited: FxHashSet::default(),
            was_loaded: false,
            has_nonlocal_use: false,
            is_stored_value: false,
        }
    }

    fn analyze_alloca_node(&mut self, rvsdg: &Rvsdg, alloca_node: Node) -> AnalysisResult {
        // Reset
        self.visited.clear();
        self.has_nonlocal_use = false;
        self.is_stored_value = false;
        self.was_loaded = false;

        // Perform the analysis
        self.visit_value_output(rvsdg, alloca_node, 0);

        // Summarize the analysis result
        if self.has_nonlocal_use {
            AnalysisResult::Ignore
        } else if self.is_stored_value {
            AnalysisResult::NeedsPromotionPass
        } else {
            AnalysisResult::Replace
        }
    }
}

impl ValueFlowVisitor for AggregateAllocaAnalyzer {
    fn should_visit(&mut self, region: Region, user: ValueUser) -> bool {
        // Don't do duplicate visits and we can stop early if we've already found a non-local use.
        !self.has_nonlocal_use && self.visited.insert((region, user))
    }

    fn visit_region_result(&mut self, rvsdg: &Rvsdg, region: Region, result: u32) {
        let owner = rvsdg[region].owner();

        if let NodeKind::Function(_) = rvsdg[owner].kind() {
            self.has_nonlocal_use = true;
        } else {
            visit::value_flow::visit_region_result(self, rvsdg, region, result);
        }
    }

    fn visit_value_input(&mut self, rvsdg: &Rvsdg, node: Node, input: u32) {
        use NodeKind::*;
        use SimpleNode::*;

        let kind = rvsdg[node].kind();

        match kind {
            Simple(OpCall(_)) => {
                self.has_nonlocal_use = true;
            }
            Simple(OpLoad(_)) => {
                // Do propagate the search past an OpLoad for non-local use analysis, but once the
                // value has been loaded, the value should no longer affect stored-value analysis.
                self.was_loaded = true;

                visit::value_flow::visit_value_input(self, rvsdg, node, input);
            }
            Simple(OpStore(_)) if input == 1 && !self.was_loaded => {
                self.is_stored_value = true;

                // We do continue searching for non-local uses, as non-local will cause an alloca
                // to be ignored entirely, which supersedes retrying after memory promotion.
                visit::value_flow::visit_value_input(self, rvsdg, node, input);
            }
            Simple(OpStore(_))
            | Simple(OpExtractField(_))
            | Simple(OpExtractElement(_))
            | Simple(OpFieldPtr(_))
            | Simple(OpElementPtr(_))
            | Simple(OpVariantPtr(_))
            | Simple(OpDiscriminantPtr(_))
            | Simple(OpGetDiscriminant(_))
            | Simple(OpSetDiscriminant(_))
            | Simple(OpGetSliceOffset(_)) => {
                // These operations take a pointer to an aggregate as input, but splitting does not
                // propagate past these node kinds, so we end the search
            }
            Switch(_) | Loop(_) | Simple(OpOffsetSlice(_)) => {
                visit::value_flow::visit_value_input(self, rvsdg, node, input);
            }
            _ => unreachable!(
                "node `{node:?}` ({kind:?}) cannot take (a pointer to) an aggregate value as input"
            ),
        }
    }
}

struct Replacer<'a, 'b> {
    rvsdg: &'a mut Rvsdg,
    candidate_queue: &'b mut VecDeque<Node>,
    ty: TypeRegistry,
}

impl Replacer<'_, '_> {
    fn replace_alloca(&mut self, node: Node) {
        let node_data = &self.rvsdg[node];
        let region = node_data.region();
        let node_data = node_data.expect_op_alloca();
        let ty = node_data.ty();

        let mut replacements = Vec::new();

        match self.rvsdg.ty().clone().kind(ty).deref() {
            TypeKind::Enum(_) => {
                // Enum replacement is handled separately
                return replace_enum_alloca(&mut self.rvsdg, node, |node, ty| {
                    if self.ty.kind(ty).is_aggregate() {
                        self.candidate_queue.push_back(node);
                    }
                });
            }
            TypeKind::Array {
                element_ty: base,
                count,
                ..
            } => {
                let element_ptr_ty = self.ty.register(TypeKind::Ptr(*base));
                let element_is_aggregate = self.ty.kind(*base).is_aggregate();

                for _ in 0..*count {
                    let element_node = self.rvsdg.add_op_alloca(region, *base);

                    replacements.push(ValueInput {
                        ty: element_ptr_ty,
                        origin: ValueOrigin::Output {
                            producer: element_node,
                            output: 0,
                        },
                    });

                    if element_is_aggregate {
                        self.candidate_queue.push_back(element_node);
                    }
                }
            }
            TypeKind::Struct(struct_data) => {
                for field in &struct_data.fields {
                    let field_ty = field.ty;
                    let field_ptr_ty = self.ty.register(TypeKind::Ptr(field_ty));
                    let field_node = self.rvsdg.add_op_alloca(region, field_ty);

                    replacements.push(ValueInput {
                        ty: field_ptr_ty,
                        origin: ValueOrigin::Output {
                            producer: field_node,
                            output: 0,
                        },
                    });

                    if self.ty.kind(field_ty).is_aggregate() {
                        self.candidate_queue.push_back(field_node);
                    }
                }
            }
            _ => unreachable!("type is not an aggregate, node should not have been a candidate"),
        }

        self.visit_users(node, 0, &replacements);

        let _ = self.rvsdg.try_remove_node(node);
    }

    fn visit_users(&mut self, node: Node, output: u32, split_inputs: &[ValueInput]) {
        let region = self.rvsdg[node].region();
        let user_count = self.rvsdg[node].value_outputs()[output as usize]
            .users
            .len();

        // We iterate over users in reverse order, so that users may remove themselves from the user
        // set without disrupting iteration
        for i in (0..user_count).rev() {
            let user = self.rvsdg[node].value_outputs()[output as usize].users[i];

            self.visit_user(region, user, split_inputs);
        }
    }

    fn visit_user(&mut self, region: Region, user: ValueUser, split_input: &[ValueInput]) {
        match user {
            ValueUser::Result(result) => self.split_region_result(region, result, split_input),
            ValueUser::Input { consumer, input } => {
                self.visit_node_input(consumer, input, split_input)
            }
        }
    }

    fn split_region_result(&mut self, region: Region, result: u32, split_input: &[ValueInput]) {
        let owner = self.rvsdg[region].owner();

        match self.rvsdg[owner].kind() {
            NodeKind::Switch(_) => self.split_switch_result(region, result, split_input),
            NodeKind::Loop(_) => self.split_loop_result(region, result, split_input),
            NodeKind::Function(_) => panic!(
                "cannot split function result; non-local-use analyses should have rejected the \
                alloca"
            ),
            _ => unreachable!("node kind cannot be a region owner"),
        }
    }

    fn visit_node_input(&mut self, node: Node, input: u32, split_input: &[ValueInput]) {
        use NodeKind::*;
        use SimpleNode::*;

        match self.rvsdg[node].kind() {
            Switch(_) => self.split_switch_input(node, input, split_input),
            Loop(_) => self.split_loop_input(node, input, split_input),
            Simple(OpLoad(_)) => self.split_op_load(node, split_input),
            Simple(OpStore(_)) => self.split_op_store(node, input, split_input),
            Simple(OpFieldPtr(_)) => self.visit_op_field_ptr(node, split_input),
            Simple(OpElementPtr(_)) => self.visit_op_element_ptr(node, split_input),
            Simple(OpExtractField(_)) => self.visit_op_extract_field(node, split_input),
            Simple(OpExtractElement(_)) => self.visit_op_extract_element(node, split_input),
            Simple(OpOffsetSlice(_)) => self.visit_users(node, 0, split_input),
            Simple(OpGetSliceOffset(_)) => (),
            Simple(ValueProxy(_)) => self.visit_value_proxy(node, split_input),
            _ => unreachable!("node kind cannot take a pointer or aggregate value as input"),
        }
    }

    fn visit_op_field_ptr(&mut self, node: Node, split_input: &[ValueInput]) {
        let node_data = &self.rvsdg[node];
        let region = node_data.region();
        let node_data = node_data.expect_op_field_ptr();
        let field_index = node_data.field_index();
        let new_user_origin = split_input[field_index as usize].origin;

        self.rvsdg.reconnect_value_users(
            region,
            ValueOrigin::Output {
                producer: node,
                output: 0,
            },
            new_user_origin,
        );

        self.rvsdg.remove_node(node);
    }

    fn visit_op_element_ptr(&mut self, node: Node, split_input: &[ValueInput]) {
        let node_data = &self.rvsdg[node];
        let region = node_data.region();
        let node_data = node_data.expect_op_element_ptr();
        let output_ty = node_data.value_output().ty;

        let first_index = ElementIndex::from_origin(self.rvsdg, node_data.index_input().origin);

        let new_user_origin = match first_index {
            ElementIndex::Static(index) => split_input[index as usize].origin,
            ElementIndex::Dynamic(selector) => {
                // The element index is not statically known. We'll have to dynamically select an
                // input at runtime with a switch node.

                let branch_count = split_input.len() as u32;
                let to_predicate = self.rvsdg.add_op_u32_to_branch_selector(
                    region,
                    branch_count,
                    ValueInput {
                        ty: TY_U32,
                        origin: selector,
                    },
                );
                let mut switch_inputs = Vec::with_capacity(split_input.len() + 1);

                switch_inputs.push(ValueInput {
                    ty: TY_PREDICATE,
                    origin: ValueOrigin::Output {
                        producer: to_predicate,
                        output: 0,
                    },
                });
                switch_inputs.extend(split_input.iter().copied());

                let switch = self.rvsdg.add_switch(
                    region,
                    switch_inputs,
                    vec![ValueOutput::new(output_ty)],
                    None,
                );

                for i in 0..split_input.len() {
                    let branch = self.rvsdg.add_switch_branch(switch);
                    let origin = ValueOrigin::Argument(i as u32);

                    self.rvsdg.reconnect_region_result(branch, 0, origin);
                }

                ValueOrigin::Output {
                    producer: switch,
                    output: 0,
                }
            }
        };

        self.rvsdg.reconnect_value_users(
            region,
            ValueOrigin::Output {
                producer: node,
                output: 0,
            },
            new_user_origin,
        );

        // We've reconnected all the node's users now. Consequently, it's now dead and can
        // be removed.
        self.rvsdg.remove_node(node);
    }

    fn visit_op_extract_field(&mut self, node: Node, split_input: &[ValueInput]) {
        let node_data = &self.rvsdg[node];
        let region = node_data.region();
        let node_data = node_data.expect_op_extract_field();
        let field_index = node_data.field_index();
        let new_user_origin = split_input[field_index as usize].origin;

        self.rvsdg.reconnect_value_users(
            region,
            ValueOrigin::Output {
                producer: node,
                output: 0,
            },
            new_user_origin,
        );

        self.rvsdg.remove_node(node);
    }

    fn visit_op_extract_element(&mut self, node: Node, split_input: &[ValueInput]) {
        let node_data = &self.rvsdg[node];
        let region = node_data.region();
        let node_data = node_data.expect_op_extract_element();
        let output_ty = node_data.value_output().ty;

        let first_index = ElementIndex::from_origin(self.rvsdg, node_data.index_input().origin);

        let new_user_origin = match first_index {
            ElementIndex::Static(index) => split_input[index as usize].origin,
            ElementIndex::Dynamic(selector) => {
                // The element index is not statically known. We'll have to dynamically select an
                // input at runtime with a switch node.

                let branch_count = split_input.len() as u32;
                let to_predicate = self.rvsdg.add_op_u32_to_branch_selector(
                    region,
                    branch_count,
                    ValueInput {
                        ty: TY_U32,
                        origin: selector,
                    },
                );
                let mut switch_inputs = Vec::with_capacity(split_input.len() + 1);

                switch_inputs.push(ValueInput {
                    ty: TY_PREDICATE,
                    origin: ValueOrigin::Output {
                        producer: to_predicate,
                        output: 0,
                    },
                });
                switch_inputs.extend(split_input.iter().copied());

                let switch = self.rvsdg.add_switch(
                    region,
                    switch_inputs,
                    vec![ValueOutput::new(output_ty)],
                    None,
                );

                for (i, input) in split_input.iter().enumerate() {
                    let branch = self.rvsdg.add_switch_branch(switch);
                    let origin = ValueOrigin::Argument(i as u32);

                    self.rvsdg.reconnect_region_result(branch, 0, origin);
                }

                ValueOrigin::Output {
                    producer: switch,
                    output: 0,
                }
            }
        };

        self.rvsdg.reconnect_value_users(
            region,
            ValueOrigin::Output {
                producer: node,
                output: 0,
            },
            new_user_origin,
        );

        // We've reconnected all the node's users now. Consequently, it's now dead and can
        // be removed.
        self.rvsdg.remove_node(node);
    }

    fn split_op_load(&mut self, node: Node, split_input: &[ValueInput]) {
        let region = self.rvsdg[node].region();
        let state_origin = self.rvsdg[node]
            .state()
            .expect("load operation should part of state chain")
            .origin;

        // The OpLoad nodes we produce will have to be linked into the state chain. Though the order
        // in which this happens does not semantically to the program, for inspection of the RVSDG
        // it is more intuitive when this matches the input order. Because of the way successively
        // inserting a node with the same state origin behaves, we reverse the iteration order to
        // achieve this.
        let mut split_input = split_input
            .iter()
            .rev()
            .map(|input| {
                let TypeKind::Ptr(output_ty) = *self.rvsdg.ty().kind(input.ty) else {
                    panic!("expected input to load operation to be a pointer");
                };

                let split_node = self.rvsdg.add_op_load(region, *input, state_origin);

                ValueInput {
                    ty: output_ty,
                    origin: ValueOrigin::Output {
                        producer: split_node,
                        output: 0,
                    },
                }
            })
            .collect::<Vec<_>>();

        // We reversed the iteration order when we produced this list. Though again the order of
        // these does not matter semantically to the program, for inspection of the RVSDG it is more
        // intuitive to maintain a consistent order, so we unreverse the order here.
        split_input.reverse();

        self.visit_users(node, 0, &split_input);

        // After visiting the users of the original node's output, all users should have been
        // reconnected to the split nodes and the original node should have no users left; we should
        // be able to remove the node now.
        self.rvsdg.remove_node(node);
    }

    fn split_op_store(&mut self, node: Node, input: u32, split_input: &[ValueInput]) {
        // OpStore presents a more complex case than most of the other cases. This is due to OpStore
        // having two inputs, both of which have to split at the same time (as opposed to e.g. a
        // Switch input, where we can split inputs individually). For whichever input is visited
        // first, we generate final connectivity immediately; for the other, we introduce an
        // intermediate set of [OpFieldPtr]/[OpElementPtr] nodes (for the case of the "pointer"
        // input) or an intermediate set of [OpExtractField]/[OpExtractElement] nodes (for the case
        // of the "value" input).
        //
        // However, it is possible that both inputs originate from a common output somewhere up
        // the DAG. Adding [OpElementPtr]/[OpExtractElement] nodes to upstream user sets runs the
        // risk of modifying a user set concurrently with its iteration. To sidestep this problem,
        // instead of directly connecting the [OpElementPtr]/[OpExtractElement] nodes to the value
        // origin, we first introduce a [ValueProxy] node with [Rvsdg::proxy_origin_user] and then
        // connect the [OpElementPtr]/[OpExtractElement] to the output of this [ValueProxy] node
        // instead.
        //
        // The [ValueProxy] and [OpElementPtr]/[OpExtractElement] nodes introduced by this strategy
        // are in most cases temporary. When later the DAG path that would have arrived at the other
        // input is traversed, the [ValueProxy] and [OpElementPtr]/[OpExtractElement] nodes will
        // typically be eliminated. For cases where the input does not originate from an OpAlloca,
        // [ValueProxy] nodes will have to be cleaned up by a later pass (in these cases the
        // [OpElementPtr]/[OpExtractElement] nodes typically remain necessary).

        assert!(input < 2, "OpStore only has 2 value inputs");

        let node_data = &self.rvsdg[node];
        let region = node_data.region();
        let state_origin = node_data
            .state()
            .expect("store op should have state")
            .origin;
        let node_data = node_data.expect_op_store();
        let mut ptr_input = *node_data.ptr_input();
        let mut value_input = *node_data.value_input();

        if input == 0 {
            // If the provoking input is the pointer input, then proxy the value input
            let proxy = self.rvsdg.proxy_origin_user(
                region,
                value_input.ty,
                value_input.origin,
                ValueUser::Input {
                    consumer: node,
                    input: 1,
                },
            );

            value_input.origin = ValueOrigin::Output {
                producer: proxy,
                output: 0,
            };
        }

        if input == 1 {
            // If the provoking input is the value input, then proxy the pointer input
            let proxy = self.rvsdg.proxy_origin_user(
                region,
                ptr_input.ty,
                ptr_input.origin,
                ValueUser::Input {
                    consumer: node,
                    input: 0,
                },
            );

            ptr_input.origin = ValueOrigin::Output {
                producer: proxy,
                output: 0,
            };
        }

        match self.rvsdg.ty().clone().kind(value_input.ty).deref() {
            TypeKind::Array {
                element_ty: base,
                count,
                ..
            } => {
                // We iterate the element indices in reverse due to the way we link the load and
                // store nodes we create into the state chain: we repeatedly reuse the state_origin
                // from the original unsplit store node. This means that adding the lowest index
                // last, will make it end up the earliest in the state chain. Though the actual
                // order should not matter for the validity of the program, lowest-to-highest index
                // order is the more natural order for human review of the compiler's output.
                for i in (0..*count).rev() {
                    self.add_store_element_nodes(
                        region,
                        input,
                        i as u32,
                        *base,
                        ptr_input,
                        value_input,
                        state_origin,
                        split_input,
                    );
                }
            }
            TypeKind::Struct(struct_data) => {
                let field_count = struct_data.fields.len();

                // We iterate in reverse for the same reason as for the array case, see the comment
                // above.
                for i in (0..field_count).rev() {
                    let field_ty = struct_data.fields[i].ty;

                    self.add_store_field_nodes(
                        region,
                        input,
                        i as u32,
                        field_ty,
                        ptr_input,
                        value_input,
                        state_origin,
                        split_input,
                    );
                }
            }
            _ => unreachable!("type is not an aggregate"),
        }

        self.rvsdg.remove_node(node);
    }

    fn add_store_element_nodes(
        &mut self,
        region: Region,
        provoking_input: u32,
        element_index: u32,
        element_ty: Type,
        ptr_input: ValueInput,
        value_input: ValueInput,
        state_origin: StateOrigin,
        split_input: &[ValueInput],
    ) {
        let ptr_ty = self.rvsdg.ty().register(TypeKind::Ptr(element_ty));
        let index_input = self.rvsdg.add_const_u32(region, element_index);

        let element_ptr_input = if provoking_input == 0 {
            split_input[element_index as usize]
        } else {
            let element_ptr = self.rvsdg.add_op_element_ptr(
                region,
                ptr_input,
                ValueInput::output(TY_U32, index_input, 0),
            );

            ValueInput::output(ptr_ty, element_ptr, 0)
        };

        let element_value_input = if provoking_input == 1 {
            split_input[element_index as usize]
        } else {
            let element_value = self.rvsdg.add_op_extract_element(
                region,
                value_input,
                ValueInput::output(TY_U32, index_input, 0),
            );

            ValueInput::output(element_ty, element_value, 0)
        };

        self.rvsdg
            .add_op_store(region, element_ptr_input, element_value_input, state_origin);
    }

    fn add_store_field_nodes(
        &mut self,
        region: Region,
        provoking_input: u32,
        field_index: u32,
        field_ty: Type,
        ptr_input: ValueInput,
        value_input: ValueInput,
        state_origin: StateOrigin,
        split_input: &[ValueInput],
    ) {
        let ptr_ty = self.rvsdg.ty().register(TypeKind::Ptr(field_ty));

        let field_ptr_input = if provoking_input == 0 {
            split_input[field_index as usize]
        } else {
            let field_ptr = self.rvsdg.add_op_field_ptr(region, ptr_input, field_index);

            ValueInput::output(ptr_ty, field_ptr, 0)
        };

        let field_value_input = if provoking_input == 1 {
            split_input[field_index as usize]
        } else {
            let field_value = self
                .rvsdg
                .add_op_extract_field(region, value_input, field_index);

            ValueInput::output(field_ty, field_value, 0)
        };

        self.rvsdg
            .add_op_store(region, field_ptr_input, field_value_input, state_origin);
    }

    fn split_switch_input(&mut self, node: Node, input: u32, split_input: &[ValueInput]) {
        assert_ne!(input, 0, "the branch selector input is never an aggregate");

        let arg_index = input as usize - 1;
        let node_data = self.rvsdg[node].expect_switch();
        let branch_count = node_data.branches().len();
        let split_args_base = node_data.value_inputs().len() - 1;

        let split_args = split_input
            .iter()
            .enumerate()
            .map(|(i, input)| {
                self.rvsdg.add_switch_input(node, *input);

                ValueInput {
                    ty: input.ty,
                    origin: ValueOrigin::Argument((split_args_base + i) as u32),
                }
            })
            .collect::<Vec<_>>();

        for branch_index in 0..branch_count {
            let branch = self.rvsdg[node].expect_switch().branches()[branch_index];

            self.redirect_region_argument(branch, arg_index as u32, &split_args);
        }
    }

    fn split_switch_result(&mut self, branch: Region, result: u32, split_input: &[ValueInput]) {
        let node = self.rvsdg[branch].owner();
        let node_data = self.rvsdg[node].expect_switch();
        let output_ty = node_data.value_outputs()[result as usize].ty;

        let split_kind = match self.rvsdg.ty().kind(output_ty).deref() {
            TypeKind::Ptr(ty) => match self.rvsdg.ty().kind(*ty).deref() {
                TypeKind::Struct(_) => Ok(SwitchOutputSplitKind::Struct),
                TypeKind::Array { .. } | TypeKind::Slice { .. } => Ok(SwitchOutputSplitKind::Array),
                _ => Err(()),
            },
            TypeKind::Struct(_) => Ok(SwitchOutputSplitKind::Struct),
            TypeKind::Array { .. } | TypeKind::Slice { .. } => Ok(SwitchOutputSplitKind::Array),
            _ => Err(()),
        };

        let Ok(split_kind) = split_kind else {
            panic!(
                "expected a (pointer to a) struct or array type, got `{}`",
                output_ty.to_string(self.rvsdg.ty())
            );
        };

        match split_kind {
            SwitchOutputSplitKind::Struct => {
                self.split_switch_struct_output(node, result, branch, split_input)
            }
            SwitchOutputSplitKind::Array => self.try_split_switch_array_output(node, result),
        }
    }

    fn split_switch_struct_output(
        &mut self,
        node: Node,
        output: u32,
        provoking_branch: Region,
        split_input: &[ValueInput],
    ) {
        let node_data = self.rvsdg[node].expect_switch();
        let branch_count = node_data.branches().len();
        let output_ty = node_data.value_outputs()[output as usize].ty;
        let is_ptr = self.rvsdg.ty().kind(output_ty).is_ptr();

        let part_result_start = node_data.value_outputs().len();
        let part_result_end = part_result_start + split_input.len();

        let split_output = split_input
            .iter()
            .map(|input| {
                let output = self.rvsdg.add_switch_output(node, input.ty);

                ValueInput {
                    ty: input.ty,
                    origin: ValueOrigin::Output {
                        producer: node,
                        output,
                    },
                }
            })
            .collect::<Vec<_>>();

        for branch in 0..branch_count {
            let branch = self.rvsdg[node].expect_switch().branches()[branch];

            if branch == provoking_branch {
                // If the current branch is the provoking branch, then we already have part values
                // available in our `split_input`, so directly connect

                for (part, part_result) in
                    (part_result_start..part_result_end).into_iter().enumerate()
                {
                    self.rvsdg.reconnect_region_result(
                        branch,
                        part_result as u32,
                        split_input[part].origin,
                    );
                }
            } else {
                // If the current branch is not the provoking branch, then use part-selector nodes
                // (OpFieldPtr and OpExtractField) to split the original results input value and
                // provide inputs for the new result-parts. If scalar-replacement ends up visiting
                // this branch for another value-flow path later. Then these parts selectors will
                // "dissolve" and act as terminators for the value-flow splitting. If we never visit
                // this branch in another value-flow path (because the value for this path
                // originates from a global binding), then these part-selectors are the correct
                // final transform for this value-flow path and no further adjustments are needed.

                for (field, part_result) in
                    (part_result_start..part_result_end).into_iter().enumerate()
                {
                    let input = self.rvsdg[branch].value_results()[output as usize];

                    // To ensure we're not interfering with any active traversal over the original
                    // origin's users, insert a ValueProxy node.
                    let proxy = self.rvsdg.proxy_origin_user(
                        branch,
                        input.ty,
                        input.origin,
                        ValueUser::Result(output),
                    );
                    let input = ValueInput::output(input.ty, proxy, 0);

                    let part_node = if is_ptr {
                        self.rvsdg.add_op_field_ptr(branch, input, field as u32)
                    } else {
                        self.rvsdg.add_op_extract_field(branch, input, field as u32)
                    };

                    self.rvsdg.reconnect_region_result(
                        branch,
                        part_result as u32,
                        ValueOrigin::Output {
                            producer: part_node,
                            output: 0,
                        },
                    );
                }
            }
        }

        self.visit_users(node, output, &split_output);
    }

    fn try_split_switch_array_output(&mut self, node: Node, output: u32) {
        use NodeKind::*;
        use SimpleNode::*;

        let node_data = self.rvsdg[node].expect_switch();

        // We'll only split the output if all branches are ready, that is, all corresponding results
        // receive their input from a Reaggregation node. We'll combine that with resolving the part
        // count (the maximum number of parts across all Reaggregation nodes), which saves us
        // having to iterate over the branches again later. The `part_count` below being `Some`
        // afterward indicates that we're ready to proceed with the split.
        let mut part_count = Some(0);
        for branch in node_data.branches() {
            let result_origin = self.rvsdg[*branch].value_results()[output as usize].origin;

            if let ValueOrigin::Output {
                producer,
                output: 0,
            } = result_origin
                && let Simple(Reaggregation(n)) = self.rvsdg[producer].kind()
            {
                part_count = Some(usize::max(part_count.unwrap_or(0), n.parts().len()));
            } else {
                part_count = None;

                break;
            }
        }

        if let Some(part_count) = part_count {
            let branch_count = node_data.branches().len();
            let aggregate_ty = node_data.value_outputs()[output as usize].ty;

            let part_ty = match *self.rvsdg.ty().kind(aggregate_ty) {
                TypeKind::Ptr(ty) => match *self.rvsdg.ty().kind(ty) {
                    TypeKind::Array { element_ty, .. } | TypeKind::Slice { element_ty, .. } => {
                        self.rvsdg.ty().register(TypeKind::Ptr(element_ty))
                    }
                    _ => unreachable!("expected an array or slice type"),
                },
                TypeKind::Array { element_ty, .. } | TypeKind::Slice { element_ty, .. } => {
                    element_ty
                }
                _ => unreachable!("expected an array or slice type"),
            };

            let part_result_start = node_data.value_outputs().len();
            let part_result_end = part_result_start + part_count;

            let split_output = (0..part_count)
                .into_iter()
                .map(|_| {
                    let output = self.rvsdg.add_switch_output(node, part_ty);

                    ValueInput {
                        ty: part_ty,
                        origin: ValueOrigin::Output {
                            producer: node,
                            output,
                        },
                    }
                })
                .collect::<Vec<_>>();

            for branch in 0..branch_count {
                let branch = self.rvsdg[node].expect_switch().branches()[branch];
                let result_origin = self.rvsdg[branch].value_results()[output as usize].origin;
                let ValueOrigin::Output {
                    producer: reaggregation_node,
                    output: 0,
                } = result_origin
                else {
                    panic!("expected the result to be connected to a reaggregation node");
                };
                let reaggregation_count = self.rvsdg[reaggregation_node]
                    .expect_reaggregation()
                    .parts()
                    .len();

                for (part, part_result) in
                    (part_result_start..part_result_end).into_iter().enumerate()
                {
                    // In case this branch has fewer reaggregated parts than new result-parts, clamp
                    // the part index to the number of parts, so that we repeat the last array
                    // element for the excess result-parts (see also the module level
                    // documentation).
                    let part = usize::min(part, reaggregation_count);
                    let origin = self.rvsdg[reaggregation_node]
                        .expect_reaggregation()
                        .parts()[part]
                        .origin;

                    self.rvsdg
                        .reconnect_region_result(branch, part_result as u32, origin);
                }

                // Now that all parts have been connected, dissolve the reaggregation node and
                // reconnect the original aggregate value back to the original result.
                self.rvsdg.dissolve_reaggregation(reaggregation_node);
            }

            self.visit_users(node, output, &split_output);
        }
    }

    fn split_loop_input(&mut self, node: Node, input: u32, split_input: &[ValueInput]) {
        let node_data = self.rvsdg[node].expect_loop();
        let loop_region = node_data.loop_region();
        let prior_input_count = node_data.value_inputs().len();
        let prior_result_count = prior_input_count + 1;

        let mut split_args = Vec::with_capacity(split_input.len());
        let mut split_outputs = Vec::with_capacity(split_input.len());

        // Add inputs/outputs/arguments/results for each element of the original aggregate input
        // and record mappings for both the arguments and the outputs.
        for (i, input) in split_input.iter().enumerate() {
            self.rvsdg.add_loop_input(node, *input);

            split_args.push(ValueInput {
                ty: input.ty,
                origin: ValueOrigin::Argument((prior_input_count + i) as u32),
            });
            split_outputs.push(ValueInput {
                ty: input.ty,
                origin: ValueOrigin::Output {
                    producer: node,
                    output: (prior_input_count + i) as u32,
                },
            });
        }

        // First connect all region results that we've created to the unsplit input that the
        // original result connects to, via [OpElementPtr] nodes or [OpExtractElement] nodes,
        // depending on whether the result type is a pointer or an immediate value. This also
        // disconnects the original result.
        self.redirect_region_result(loop_region, input as usize + 1, prior_result_count);

        // Now redirect the argument using the argument mapping. We do this after redirecting the
        // results, because otherwise this might try to split the original result again; by doing
        // this after result redirection, all the argument's user tree should terminate at the
        // [OpElementPtr]/[OpExtractElement] nodes that were inserted, since nothing should
        // be connected to the original result anymore.
        self.redirect_region_argument(loop_region, input, &split_args);

        // Finally, redirect the value output using the output mapping
        self.visit_users(node, input, &split_outputs);
    }

    fn split_loop_result(&mut self, region: Region, result: u32, split_input: &[ValueInput]) {
        assert_ne!(
            result, 0,
            "the reentry decider result is never an aggregate"
        );

        let owner = self.rvsdg[region].owner();
        let outer_region = self.rvsdg[owner].region();
        let loop_data = self.rvsdg[owner].expect_loop();
        let prior_input_count = loop_data.value_inputs().len();
        let prior_result_count = prior_input_count + 1;

        let input_index = result - 1;
        let value_input = loop_data.value_inputs()[input_index as usize];

        // A loop node's input and outputs must match, so splitting a result also means splitting
        // the corresponding input. However, propagating splits "upwards" runs the risk of
        // concurrently modifying a user set that is currently being traversed. To sidestep this,
        // we insert a proxy between the input and its origin, so that our modification will modify
        // the proxy's user set, which we know is not currently being traversed.
        let proxy = self.rvsdg.proxy_origin_user(
            outer_region,
            value_input.ty,
            value_input.origin,
            ValueUser::Input {
                consumer: owner,
                input: input_index,
            },
        );
        let proxy_input = ValueInput {
            ty: value_input.ty,
            origin: ValueOrigin::Output {
                producer: proxy,
                output: 0,
            },
        };

        let mut split_args = Vec::with_capacity(split_input.len());
        let mut split_outputs = Vec::with_capacity(split_input.len());

        let ty_reg = self.rvsdg.ty().clone();

        // Add inputs for each element of the aggregate and connect them to the proxy via either
        // an OpElementPtr node if the input is of a pointer type, or via an OpExtractElement
        // node otherwise. Also, record an argument mapping in `split_args` and an output mapping in
        // `split_outputs`.
        match ty_reg.kind(value_input.ty).deref() {
            TypeKind::Ptr(pointee_ty) => match ty_reg.kind(*pointee_ty).deref() {
                TypeKind::Array {
                    element_ty: base,
                    count,
                    ..
                } => {
                    for i in 0..*count {
                        let ptr_ty = ty_reg.register(TypeKind::Ptr(*base));
                        let index = self.rvsdg.add_const_u32(outer_region, i as u32);
                        let element = self.rvsdg.add_op_element_ptr(
                            outer_region,
                            proxy_input,
                            ValueInput::output(TY_U32, index, 0),
                        );

                        self.rvsdg
                            .add_loop_input(owner, ValueInput::output(ptr_ty, element, 0));

                        split_args.push(ValueInput::argument(
                            ptr_ty,
                            prior_input_count as u32 + i as u32,
                        ));
                        split_outputs.push(ValueInput::output(
                            ptr_ty,
                            owner,
                            prior_input_count as u32 + i as u32,
                        ));
                    }
                }
                TypeKind::Struct(struct_data) => {
                    for (i, field) in struct_data.fields.iter().enumerate() {
                        let element_ty = field.ty;
                        let ptr_ty = ty_reg.register(TypeKind::Ptr(element_ty));

                        let element =
                            self.rvsdg
                                .add_op_field_ptr(outer_region, proxy_input, i as u32);

                        self.rvsdg
                            .add_loop_input(owner, ValueInput::output(ptr_ty, element, 0));

                        split_args.push(ValueInput::argument(
                            ptr_ty,
                            prior_input_count as u32 + i as u32,
                        ));
                        split_outputs.push(ValueInput::output(
                            ptr_ty,
                            owner,
                            prior_input_count as u32 + i as u32,
                        ));
                    }
                }
                _ => panic!("pointee type is not an aggregate"),
            },
            TypeKind::Array {
                element_ty: base,
                count,
                ..
            } => {
                for i in 0..*count {
                    let index = self.rvsdg.add_const_u32(outer_region, i as u32);
                    let element = self.rvsdg.add_op_extract_element(
                        outer_region,
                        proxy_input,
                        ValueInput::output(TY_U32, index, 0),
                    );

                    self.rvsdg
                        .add_loop_input(owner, ValueInput::output(*base, element, 0));

                    split_args.push(ValueInput::argument(
                        *base,
                        prior_input_count as u32 + i as u32,
                    ));
                    split_outputs.push(ValueInput::output(
                        *base,
                        owner,
                        prior_input_count as u32 + i as u32,
                    ));
                }
            }
            TypeKind::Struct(struct_data) => {
                for (i, field) in struct_data.fields.iter().enumerate() {
                    let element_ty = field.ty;

                    let element =
                        self.rvsdg
                            .add_op_extract_field(outer_region, proxy_input, i as u32);

                    self.rvsdg
                        .add_loop_input(owner, ValueInput::output(element_ty, element, 0));

                    split_args.push(ValueInput::argument(
                        element_ty,
                        prior_input_count as u32 + i as u32,
                    ));
                    split_outputs.push(ValueInput::output(
                        element_ty,
                        owner,
                        prior_input_count as u32 + i as u32,
                    ));
                }
            }
            _ => unreachable!("type is not an aggregate"),
        }

        // Reconnect the results we just created to the `split_input`.
        for (i, input) in split_input.iter().enumerate() {
            let result_index = prior_result_count + i;

            self.rvsdg
                .reconnect_region_result(region, result_index as u32, input.origin);
        }

        // Redirect the argument using the argument mapping
        self.redirect_region_argument(region, input_index, &split_args);

        // Redirect the value output using the output mapping
        self.visit_users(owner, input_index, &split_outputs);
    }

    fn visit_value_proxy(&mut self, node: Node, split_input: &[ValueInput]) {
        self.visit_users(node, 0, split_input);
        let _ = self.rvsdg.try_remove_node(node);
    }

    /// Redirects all users of the `region`'s given `argument` to the `split_input` nodes.
    ///
    /// Leaves the `argument` without any users.
    fn redirect_region_argument(
        &mut self,
        region: Region,
        argument: u32,
        split_input: &[ValueInput],
    ) {
        let arg_index = argument as usize;
        let user_count = self.rvsdg[region].value_arguments()[arg_index].users.len();

        // We iterate over users in reverse order, so that users may more themselves from the user
        // set, without disrupting iteration
        for user_index in (0..user_count).rev() {
            let user = self.rvsdg[region].value_arguments()[arg_index].users[user_index];

            self.visit_user(region, user, split_input)
        }
    }

    /// Redirects the origin for the `region`'s given `result` to a set of "split" results that
    /// start at `split_results_start`, via either [OpElementPtr] or [OpExtractElement] nodes
    /// depending on whether to original input type is a pointer or immediate value.
    ///
    /// Leaves the original result connected to the "placeholder" origin.
    fn redirect_region_result(
        &mut self,
        region: Region,
        original: usize,
        split_results_start: usize,
    ) {
        let original_input = self.rvsdg[region].value_results()[original];
        let ty_reg = self.rvsdg.ty().clone();

        match ty_reg.kind(original_input.ty).deref() {
            TypeKind::Ptr(pointee_ty) => match ty_reg.kind(*pointee_ty).deref() {
                TypeKind::Array { count, .. } => {
                    for i in 0..*count {
                        let index_node = self.rvsdg.add_const_u32(region, i as u32);
                        let split_node = self.rvsdg.add_op_element_ptr(
                            region,
                            original_input,
                            ValueInput::output(TY_U32, index_node, 0),
                        );
                        let result_index = split_results_start + i as usize;

                        self.rvsdg.reconnect_region_result(
                            region,
                            result_index as u32,
                            ValueOrigin::Output {
                                producer: split_node,
                                output: 0,
                            },
                        );
                    }
                }
                TypeKind::Struct(struct_data) => {
                    for i in 0..struct_data.fields.len() {
                        let split_node =
                            self.rvsdg
                                .add_op_field_ptr(region, original_input, i as u32);
                        let result_index = split_results_start + i as usize;

                        self.rvsdg.reconnect_region_result(
                            region,
                            result_index as u32,
                            ValueOrigin::Output {
                                producer: split_node,
                                output: 0,
                            },
                        );
                    }
                }
                _ => unreachable!("pointee type is not an aggregate"),
            },
            TypeKind::Array { count, .. } => {
                for i in 0..*count {
                    let index_node = self.rvsdg.add_const_u32(region, i as u32);
                    let split_node = self.rvsdg.add_op_extract_element(
                        region,
                        original_input,
                        ValueInput::output(TY_U32, index_node, 0),
                    );
                    let result_index = split_results_start + i as usize;

                    self.rvsdg.reconnect_region_result(
                        region,
                        result_index as u32,
                        ValueOrigin::Output {
                            producer: split_node,
                            output: 0,
                        },
                    );
                }
            }
            TypeKind::Struct(struct_data) => {
                for i in 0..struct_data.fields.len() {
                    let split_node =
                        self.rvsdg
                            .add_op_extract_field(region, original_input, i as u32);
                    let result_index = split_results_start + i;

                    self.rvsdg.reconnect_region_result(
                        region,
                        result_index as u32,
                        ValueOrigin::Output {
                            producer: split_node,
                            output: 0,
                        },
                    );
                }
            }
            _ => unreachable!("type is not an aggregate or a pointer to an aggregate"),
        }

        self.rvsdg.disconnect_region_result(region, original as u32);
    }
}

pub struct AggregateReplacementContext {
    analyzer: AggregateAllocaAnalyzer,
    queue: VecDeque<Node>,
    candidates: VecDeque<Node>,
}

impl AggregateReplacementContext {
    pub fn new() -> Self {
        AggregateReplacementContext {
            analyzer: AggregateAllocaAnalyzer::new(),
            queue: VecDeque::new(),
            candidates: VecDeque::new(),
        }
    }

    pub fn for_region(&mut self, rvsdg: &Rvsdg, region: Region) -> RegionReplacementContext {
        self.queue.clear();
        self.candidates.clear();

        collect_candidate_allocas(rvsdg, &mut self.candidates, region);

        RegionReplacementContext { cx: self }
    }

    fn try_enqueue_candidates(&mut self, rvsdg: &Rvsdg, start: usize) {
        let mut i = start;

        while i < self.candidates.len() {
            let node = self.candidates[i];

            match self.analyzer.analyze_alloca_node(rvsdg, node) {
                AnalysisResult::Replace => {
                    self.queue.push_back(node);
                    self.candidates.swap_remove_back(i);

                    // Note that we don't increment `i` here, because we just moved another yet
                    // unprocessed candidate into the `i`th position, which we want to process on
                    // the next iteration.
                }
                AnalysisResult::NeedsPromotionPass => i += 1,
                AnalysisResult::Ignore => {
                    self.candidates.swap_remove_back(i);

                    // Note that we don't increment `i` here, because we just moved another yet
                    // unprocessed candidate into the `i`th position, which we want to process on
                    // the next iteration.
                }
            }
        }
    }
}

pub struct RegionReplacementContext<'a> {
    cx: &'a mut AggregateReplacementContext,
}

impl RegionReplacementContext<'_> {
    pub fn replace(&mut self, rvsdg: &mut Rvsdg) -> bool {
        self.cx.try_enqueue_candidates(rvsdg, 0);

        if self.cx.queue.is_empty() {
            false
        } else {
            let ty_registry = rvsdg.ty().clone();

            while let Some(node) = self.cx.queue.pop_front() {
                let prior_candidate_count = self.cx.candidates.len();

                let mut replacer = Replacer {
                    rvsdg,
                    candidate_queue: &mut self.cx.candidates,
                    ty: ty_registry.clone(),
                };

                replacer.replace_alloca(node);

                // Attempt to add any newly added candidates to the end of the queue *during* this
                // current replacement iteration. This helps minimize the number of
                // promotion/legalization passes we'll have to run.
                self.cx.try_enqueue_candidates(rvsdg, prior_candidate_count);
            }

            true
        }
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::ty::{TY_DUMMY, TY_PREDICATE, TY_PTR_U32};
    use crate::{BinaryOperator, FnArg, FnSig, Symbol, thin_set};

    #[test]
    fn test_scalar_replace_op_field_ptr() {
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
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ty = module.ty.register(TypeKind::Struct(crate::ty::Struct {
            fields: vec![
                crate::ty::StructField {
                    offset: 0,
                    ty: TY_U32,
                    io_binding: None,
                },
                crate::ty::StructField {
                    offset: 4,
                    ty: TY_U32,
                    io_binding: None,
                },
            ],
        }));
        let ptr_ty = module.ty.register(TypeKind::Ptr(ty));
        let field_ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let op_alloca = rvsdg.add_op_alloca(region, ty);
        let op_field_ptr =
            rvsdg.add_op_field_ptr(region, ValueInput::output(ptr_ty, op_alloca, 0), 1);
        let load = rvsdg.add_op_load(
            region,
            ValueInput::output(field_ptr_ty, op_field_ptr, 0),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load,
                output: 0,
            },
        );

        let mut cx = AggregateReplacementContext::new();
        let mut rcx = cx.for_region(&rvsdg, region);

        rcx.replace(&mut rvsdg);

        assert_eq!(
            rvsdg[region].value_results()[0].origin,
            ValueOrigin::Output {
                producer: load,
                output: 0
            }
        );

        let load_origin = rvsdg[load].expect_op_load().ptr_input().origin;
        let ValueOrigin::Output {
            producer: load_origin_node,
            output: 0,
        } = load_origin
        else {
            panic!("load origin should be the first output of a node")
        };
        let field_alloca = rvsdg[load_origin_node].expect_op_alloca();

        assert_eq!(field_alloca.ty(), TY_U32);
        assert_eq!(
            &field_alloca.value_output().users,
            &thin_set![ValueUser::Input {
                consumer: load,
                input: 0
            }]
        );

        assert!(!rvsdg.is_live_node(op_field_ptr));
        assert!(!rvsdg.is_live_node(op_alloca));
    }

    #[test]
    fn test_scalar_replace_op_element_ptr() {
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
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ty = module.ty.register(TypeKind::Array {
            element_ty: TY_PTR_U32,
            count: 2,
            stride: 4,
        });
        let ptr_ty = module.ty.register(TypeKind::Ptr(ty));
        let element_ptr_ty = module.ty.register(TypeKind::Ptr(TY_PTR_U32));

        let op_alloca = rvsdg.add_op_alloca(region, ty);
        let element_index = rvsdg.add_const_u32(region, 1);
        let op_element_ptr = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(ptr_ty, op_alloca, 0),
            ValueInput::output(TY_U32, element_index, 0),
        );
        let load = rvsdg.add_op_load(
            region,
            ValueInput::output(element_ptr_ty, op_element_ptr, 0),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load,
                output: 0,
            },
        );

        let mut cx = AggregateReplacementContext::new();
        let mut rcx = cx.for_region(&rvsdg, region);

        rcx.replace(&mut rvsdg);

        assert_eq!(
            rvsdg[region].value_results()[0].origin,
            ValueOrigin::Output {
                producer: load,
                output: 0
            }
        );

        let load_origin = rvsdg[load].expect_op_load().ptr_input().origin;
        let ValueOrigin::Output {
            producer: load_origin_node,
            output: 0,
        } = load_origin
        else {
            panic!("load origin should be the first output of a node")
        };
        let element_alloca = rvsdg[load_origin_node].expect_op_alloca();

        assert_eq!(element_alloca.ty(), TY_PTR_U32);
        assert_eq!(
            &element_alloca.value_output().users,
            &thin_set![ValueUser::Input {
                consumer: load,
                input: 0
            }]
        );

        assert!(!rvsdg.is_live_node(op_element_ptr));
        assert!(!rvsdg.is_live_node(op_alloca));
    }

    #[test]
    fn test_scalar_replace_op_element_ptr_dynamic_index() {
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

        let ty = module.ty.register(TypeKind::Array {
            element_ty: TY_PTR_U32,
            count: 2,
            stride: 4,
        });
        let ptr_ty = module.ty.register(TypeKind::Ptr(ty));
        let element_ptr_ty = module.ty.register(TypeKind::Ptr(TY_PTR_U32));

        let op_alloca = rvsdg.add_op_alloca(region, ty);
        let op_element_ptr = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(ptr_ty, op_alloca, 0),
            ValueInput::argument(TY_U32, 0),
        );
        let load = rvsdg.add_op_load(
            region,
            ValueInput::output(element_ptr_ty, op_element_ptr, 0),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load,
                output: 0,
            },
        );

        let mut cx = AggregateReplacementContext::new();
        let mut rcx = cx.for_region(&rvsdg, region);

        rcx.replace(&mut rvsdg);

        assert_eq!(
            rvsdg[region].value_results()[0].origin,
            ValueOrigin::Output {
                producer: load,
                output: 0
            }
        );

        let load_origin = rvsdg[load].expect_op_load().ptr_input().origin;
        let ValueOrigin::Output {
            producer: switch_node,
            output: 0,
        } = load_origin
        else {
            panic!("load origin should be the first output of a node")
        };

        let switch = rvsdg[switch_node].expect_switch();

        assert_eq!(switch.value_outputs().len(), 1);

        let switch_output = &switch.value_outputs()[0];

        assert_eq!(switch_output.ty, element_ptr_ty);
        assert_eq!(
            &switch_output.users,
            &thin_set![ValueUser::Input {
                consumer: load,
                input: 0
            }]
        );

        assert_eq!(switch.value_inputs().len(), 3);
        assert_eq!(switch.branches().len(), 2);

        let branch_0 = &rvsdg[switch.branches()[0]];

        assert_eq!(
            &branch_0.value_arguments()[0].users,
            &thin_set![ValueUser::Result(0)]
        );
        assert_eq!(&branch_0.value_arguments()[1].users, &thin_set![]);
        assert_eq!(branch_0.value_results()[0].origin, ValueOrigin::Argument(0));

        let branch_1 = &rvsdg[switch.branches()[1]];

        assert_eq!(&branch_1.value_arguments()[0].users, &thin_set![]);
        assert_eq!(
            &branch_1.value_arguments()[1].users,
            &thin_set![ValueUser::Result(0)]
        );
        assert_eq!(branch_1.value_results()[0].origin, ValueOrigin::Argument(1));

        assert_eq!(switch.value_inputs()[0].ty, TY_PREDICATE);

        let ValueOrigin::Output {
            producer: to_predicate,
            output: 0,
        } = switch.value_inputs()[0].origin
        else {
            panic!("switch input 0's origin should be the first output of a node")
        };

        let to_predicate_data = rvsdg[to_predicate].expect_op_u32_to_branch_selector();

        assert_eq!(to_predicate_data.branch_count(), 2);
        assert_eq!(
            to_predicate_data.value_input().origin,
            ValueOrigin::Argument(0)
        );

        assert_eq!(switch.value_inputs()[1].ty, element_ptr_ty);

        let ValueOrigin::Output {
            producer: element_0_alloca,
            output: 0,
        } = switch.value_inputs()[1].origin
        else {
            panic!("switch input 1's origin should be the first output of a node")
        };

        let element_0_alloca = rvsdg[element_0_alloca].expect_op_alloca();

        assert_eq!(element_0_alloca.ty(), TY_PTR_U32);
        assert_eq!(
            &element_0_alloca.value_output().users,
            &thin_set![ValueUser::Input {
                consumer: switch_node,
                input: 1
            }]
        );

        assert_eq!(switch.value_inputs()[2].ty, element_ptr_ty);

        let ValueOrigin::Output {
            producer: element_1_alloca,
            output: 0,
        } = switch.value_inputs()[2].origin
        else {
            panic!("switch input 2's origin should be the first output of a node")
        };

        let element_1_alloca = rvsdg[element_1_alloca].expect_op_alloca();

        assert_eq!(element_1_alloca.ty(), TY_PTR_U32);
        assert_eq!(
            &element_1_alloca.value_output().users,
            &thin_set![ValueUser::Input {
                consumer: switch_node,
                input: 2
            }]
        );

        assert!(!rvsdg.is_live_node(op_element_ptr));
        assert!(!rvsdg.is_live_node(op_alloca));
    }

    #[test]
    fn test_scalar_replace_op_extract_field() {
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
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ty = module.ty.register(TypeKind::Struct(crate::ty::Struct {
            fields: vec![
                crate::ty::StructField {
                    offset: 0,
                    ty: TY_U32,
                    io_binding: None,
                },
                crate::ty::StructField {
                    offset: 4,
                    ty: TY_U32,
                    io_binding: None,
                },
            ],
        }));
        let ptr_ty = module.ty.register(TypeKind::Ptr(ty));
        let field_ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let op_alloca = rvsdg.add_op_alloca(region, ty);
        let op_load = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, op_alloca, 0),
            StateOrigin::Argument,
        );
        let op_extract_field =
            rvsdg.add_op_extract_field(region, ValueInput::output(ty, op_load, 0), 1);

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: op_extract_field,
                output: 0,
            },
        );

        let mut cx = AggregateReplacementContext::new();
        let mut rcx = cx.for_region(&rvsdg, region);

        rcx.replace(&mut rvsdg);

        let result_input = rvsdg[region].value_results()[0];

        assert_eq!(result_input.ty, TY_U32);

        let ValueOrigin::Output {
            producer: element_load_node,
            output: 0,
        } = result_input.origin
        else {
            panic!("result origin should be the first output of a node")
        };

        let field_load = rvsdg[element_load_node].expect_op_load();

        assert_eq!(field_load.ptr_input().ty, field_ptr_ty);
        assert_eq!(field_load.value_output().ty, TY_U32);
        assert_eq!(
            &field_load.value_output().users,
            &thin_set![ValueUser::Result(0)]
        );

        let field_load_input = field_load.ptr_input();

        let ValueOrigin::Output {
            producer: field_alloca_node,
            output: field_alloca_output,
        } = field_load_input.origin
        else {
            panic!("field load op origin should be the first output of a node")
        };

        assert_eq!(field_alloca_output, 0);

        let field_alloca = rvsdg[field_alloca_node].expect_op_alloca();

        assert_eq!(field_alloca.ty(), TY_U32);
        assert_eq!(field_alloca.value_output().ty, field_ptr_ty);
        assert_eq!(
            &field_alloca.value_output().users,
            &thin_set![ValueUser::Input {
                consumer: element_load_node,
                input: 0
            }]
        );

        assert!(!rvsdg.is_live_node(op_alloca));
        assert!(!rvsdg.is_live_node(op_load));
        assert!(!rvsdg.is_live_node(op_extract_field));
    }

    #[test]
    fn test_scalar_replace_op_store() {
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
                ret_ty: None,
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ty = module.ty.register(TypeKind::Struct(crate::ty::Struct {
            fields: vec![
                crate::ty::StructField {
                    offset: 0,
                    ty: TY_U32,
                    io_binding: None,
                },
                crate::ty::StructField {
                    offset: 4,
                    ty: TY_U32,
                    io_binding: None,
                },
            ],
        }));
        let ptr_ty = module.ty.register(TypeKind::Ptr(ty));
        let field_ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let op_alloca = rvsdg.add_op_alloca(region, ty);
        let op_load = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, op_alloca, 0),
            StateOrigin::Argument,
        );
        let op_store = rvsdg.add_op_store(
            region,
            ValueInput::output(ptr_ty, op_alloca, 0),
            ValueInput::output(ty, op_load, 0),
            StateOrigin::Node(op_load),
        );

        let mut cx = AggregateReplacementContext::new();
        let mut rcx = cx.for_region(&rvsdg, region);

        rcx.replace(&mut rvsdg);

        let StateOrigin::Node(store_field_1_node) = *rvsdg[region].state_result() else {
            panic!("the state origin should be a node output")
        };

        let store_field_1 = rvsdg[store_field_1_node].expect_op_store();

        let StateOrigin::Node(store_field_0_node) = store_field_1.state().unwrap().origin else {
            panic!("the state origin should be a node output")
        };

        let store_field_0 = rvsdg[store_field_0_node].expect_op_store();

        let StateOrigin::Node(load_field_1_node) = store_field_0.state().unwrap().origin else {
            panic!("the state origin should be a node output")
        };

        let load_field_1 = rvsdg[load_field_1_node].expect_op_load();

        let StateOrigin::Node(load_field_0_node) = load_field_1.state().unwrap().origin else {
            panic!("the state origin should be a node output")
        };

        let load_field_0 = rvsdg[load_field_0_node].expect_op_load();

        let ValueOrigin::Output {
            producer: field_0_alloca_node,
            output: 0,
        } = store_field_0.ptr_input().origin
        else {
            panic!("the pointer input to store_field_0 node should be the first output of a node")
        };

        let field_0_alloca = rvsdg[field_0_alloca_node].expect_op_alloca();

        let ValueOrigin::Output {
            producer: field_1_alloca_node,
            output: 0,
        } = store_field_1.ptr_input().origin
        else {
            panic!("the pointer input to store_field_1 node should be the first output of a node")
        };

        let field_1_alloca = rvsdg[field_1_alloca_node].expect_op_alloca();

        assert_eq!(field_0_alloca.ty(), TY_U32);
        assert_eq!(
            &field_0_alloca.value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: store_field_0_node,
                    input: 0
                },
                ValueUser::Input {
                    consumer: load_field_0_node,
                    input: 0
                }
            ]
        );

        assert_eq!(field_1_alloca.ty(), TY_U32);
        assert_eq!(
            &field_1_alloca.value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: store_field_1_node,
                    input: 0
                },
                ValueUser::Input {
                    consumer: load_field_1_node,
                    input: 0
                }
            ]
        );

        assert_eq!(
            load_field_0.ptr_input(),
            &ValueInput {
                ty: field_ptr_ty,
                origin: ValueOrigin::Output {
                    producer: field_0_alloca_node,
                    output: 0,
                },
            }
        );
        assert_eq!(
            &load_field_0.value_output().users,
            &thin_set![ValueUser::Input {
                consumer: store_field_0_node,
                input: 1
            }]
        );

        assert_eq!(
            load_field_1.ptr_input(),
            &ValueInput {
                ty: field_ptr_ty,
                origin: ValueOrigin::Output {
                    producer: field_1_alloca_node,
                    output: 0,
                },
            }
        );
        assert_eq!(
            &load_field_1.value_output().users,
            &thin_set![ValueUser::Input {
                consumer: store_field_1_node,
                input: 1
            }]
        );

        assert_eq!(
            store_field_0.ptr_input(),
            &ValueInput {
                ty: field_ptr_ty,
                origin: ValueOrigin::Output {
                    producer: field_0_alloca_node,
                    output: 0,
                },
            }
        );
        assert_eq!(
            store_field_0.value_input(),
            &ValueInput {
                ty: TY_U32,
                origin: ValueOrigin::Output {
                    producer: load_field_0_node,
                    output: 0,
                },
            }
        );

        assert_eq!(
            store_field_1.ptr_input(),
            &ValueInput {
                ty: field_ptr_ty,
                origin: ValueOrigin::Output {
                    producer: field_1_alloca_node,
                    output: 0,
                },
            }
        );
        assert_eq!(
            store_field_1.value_input(),
            &ValueInput {
                ty: TY_U32,
                origin: ValueOrigin::Output {
                    producer: load_field_1_node,
                    output: 0,
                },
            }
        );

        assert!(!rvsdg.is_live_node(op_alloca));
        assert!(!rvsdg.is_live_node(op_load));
        assert!(!rvsdg.is_live_node(op_store));
    }

    #[test]
    fn test_scalar_replace_switch_input() {
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

        let ty = module.ty.register(TypeKind::Struct(crate::ty::Struct {
            fields: vec![
                crate::ty::StructField {
                    offset: 0,
                    ty: TY_U32,
                    io_binding: None,
                },
                crate::ty::StructField {
                    offset: 4,
                    ty: TY_U32,
                    io_binding: None,
                },
            ],
        }));
        let ptr_ty = module.ty.register(TypeKind::Ptr(ty));
        let field_ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let op_alloca = rvsdg.add_op_alloca(region, ty);
        let switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(ptr_ty, op_alloca, 0),
            ],
            vec![ValueOutput::new(TY_U32)],
            Some(StateOrigin::Argument),
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);
        let branch_0_op_ptr_field_ptr =
            rvsdg.add_op_field_ptr(branch_0, ValueInput::argument(ptr_ty, 0), 0);
        let branch_0_load = rvsdg.add_op_load(
            branch_0,
            ValueInput::output(field_ptr_ty, branch_0_op_ptr_field_ptr, 0),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: branch_0_load,
                output: 0,
            },
        );

        let branch_1 = rvsdg.add_switch_branch(switch_node);
        let branch_1_op_ptr_field_ptr =
            rvsdg.add_op_field_ptr(branch_1, ValueInput::argument(ptr_ty, 0), 1);
        let branch_1_load = rvsdg.add_op_load(
            branch_1,
            ValueInput::output(field_ptr_ty, branch_1_op_ptr_field_ptr, 0),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            branch_1,
            0,
            ValueOrigin::Output {
                producer: branch_1_load,
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

        let mut cx = AggregateReplacementContext::new();
        let mut rcx = cx.for_region(&rvsdg, region);

        rcx.replace(&mut rvsdg);

        let result_input = rvsdg[region].value_results()[0];

        assert_eq!(result_input.ty, TY_U32);
        assert_eq!(
            result_input.origin,
            ValueOrigin::Output {
                producer: switch_node,
                output: 0
            }
        );

        let switch = rvsdg[switch_node].expect_switch();

        assert_eq!(switch.value_inputs().len(), 4);

        let branch_0 = &rvsdg[branch_0];

        assert_eq!(
            branch_0.value_results()[0].origin,
            ValueOrigin::Output {
                producer: branch_0_load,
                output: 0
            }
        );

        assert_eq!(
            rvsdg[branch_0_load].expect_op_load().ptr_input().origin,
            ValueOrigin::Argument(1)
        );

        assert_eq!(branch_0.value_arguments().len(), 3);
        assert_eq!(branch_0.value_arguments()[1].ty, field_ptr_ty);
        assert_eq!(branch_0.value_arguments()[2].ty, field_ptr_ty);
        assert_eq!(
            &branch_0.value_arguments()[1].users,
            &thin_set![ValueUser::Input {
                consumer: branch_0_load,
                input: 0,
            }]
        );
        assert_eq!(&branch_0.value_arguments()[2].users, &thin_set![]);

        let branch_1 = &rvsdg[branch_1];

        assert_eq!(
            branch_1.value_results()[0].origin,
            ValueOrigin::Output {
                producer: branch_1_load,
                output: 0
            }
        );

        assert_eq!(
            rvsdg[branch_1_load].expect_op_load().ptr_input().origin,
            ValueOrigin::Argument(2)
        );

        assert_eq!(branch_1.value_arguments().len(), 3);
        assert_eq!(branch_1.value_arguments()[1].ty, field_ptr_ty);
        assert_eq!(branch_1.value_arguments()[2].ty, field_ptr_ty);
        assert_eq!(&branch_1.value_arguments()[1].users, &thin_set![]);
        assert_eq!(
            &branch_1.value_arguments()[2].users,
            &thin_set![ValueUser::Input {
                consumer: branch_1_load,
                input: 0,
            }]
        );

        assert_eq!(switch.value_inputs()[0].origin, ValueOrigin::Argument(0));

        let ValueOrigin::Output {
            producer: field_0_alloca_node,
            output: 0,
        } = switch.value_inputs()[2].origin
        else {
            panic!("the second input to the switch node should be the first output of a node")
        };

        let field_0_alloca = rvsdg[field_0_alloca_node].expect_op_alloca();

        assert_eq!(field_0_alloca.ty(), TY_U32);
        assert_eq!(
            &field_0_alloca.value_output().users,
            &thin_set![ValueUser::Input {
                consumer: switch_node,
                input: 2,
            }]
        );

        let ValueOrigin::Output {
            producer: field_1_alloca_node,
            output: 0,
        } = switch.value_inputs()[3].origin
        else {
            panic!("the third input to the switch node should be the first output of a node")
        };

        let field_1_alloca = rvsdg[field_1_alloca_node].expect_op_alloca();

        assert_eq!(field_1_alloca.ty(), TY_U32);
        assert_eq!(
            &field_1_alloca.value_output().users,
            &thin_set![ValueUser::Input {
                consumer: switch_node,
                input: 3,
            }]
        );

        assert!(!rvsdg.is_live_node(branch_0_op_ptr_field_ptr));
        assert!(!rvsdg.is_live_node(branch_1_op_ptr_field_ptr));
    }

    #[test]
    fn test_scalar_replace_loop_input() {
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

        let ty = module.ty.register(TypeKind::Struct(crate::ty::Struct {
            fields: vec![
                crate::ty::StructField {
                    offset: 0,
                    ty: TY_U32,
                    io_binding: None,
                },
                crate::ty::StructField {
                    offset: 4,
                    ty: TY_U32,
                    io_binding: None,
                },
            ],
        }));
        let ptr_ty = module.ty.register(TypeKind::Ptr(ty));
        let field_ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let counter = rvsdg.add_const_u32(region, 0);
        let op_alloca = rvsdg.add_op_alloca(region, ty);

        let (loop_node, loop_region) = rvsdg.add_loop(
            region,
            vec![
                ValueInput::output(TY_U32, counter, 0),
                ValueInput::argument(TY_U32, 0),
                ValueInput::output(ptr_ty, op_alloca, 0),
            ],
            None,
        );

        let op_ptr_field_ptr_node =
            rvsdg.add_op_field_ptr(loop_region, ValueInput::argument(ptr_ty, 2), 0);
        let load_node = rvsdg.add_op_load(
            loop_region,
            ValueInput::output(field_ptr_ty, op_ptr_field_ptr_node, 0),
            StateOrigin::Argument,
        );

        let increment_node = rvsdg.add_const_u32(loop_region, 1);
        let add_one_node = rvsdg.add_op_binary(
            loop_region,
            BinaryOperator::Add,
            ValueInput::output(TY_U32, load_node, 0),
            ValueInput::output(TY_U32, increment_node, 0),
        );
        let store_node = rvsdg.add_op_store(
            loop_region,
            ValueInput::output(field_ptr_ty, op_ptr_field_ptr_node, 0),
            ValueInput::output(TY_U32, add_one_node, 0),
            StateOrigin::Node(load_node),
        );

        let counter_increment_node = rvsdg.add_op_binary(
            loop_region,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 0),
            ValueInput::output(TY_U32, increment_node, 0),
        );
        let reentry_test_node = rvsdg.add_op_binary(
            loop_region,
            BinaryOperator::Lt,
            ValueInput::output(TY_U32, counter_increment_node, 0),
            ValueInput::argument(TY_U32, 1),
        );

        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: reentry_test_node,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            loop_region,
            1,
            ValueOrigin::Output {
                producer: counter_increment_node,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(loop_region, 2, ValueOrigin::Argument(1));
        rvsdg.reconnect_region_result(loop_region, 3, ValueOrigin::Argument(2));

        let mut cx = AggregateReplacementContext::new();
        let mut rcx = cx.for_region(&rvsdg, region);

        rcx.replace(&mut rvsdg);

        let loop_data = rvsdg[loop_node].expect_loop();
        let loop_region = loop_data.loop_region();

        assert_eq!(loop_data.value_inputs().len(), 5);
        assert_eq!(loop_data.value_inputs()[0].ty, TY_U32);
        assert_eq!(loop_data.value_inputs()[1].ty, TY_U32);
        assert_eq!(loop_data.value_inputs()[3].ty, field_ptr_ty);
        assert_eq!(loop_data.value_inputs()[4].ty, field_ptr_ty);

        let ValueOrigin::Output {
            producer: field_0_alloca_node,
            output: 0,
        } = loop_data.value_inputs()[3].origin
        else {
            panic!("the third input to the switch node should be the first output of a node")
        };

        let field_0_alloca = rvsdg[field_0_alloca_node].expect_op_alloca();

        assert_eq!(field_0_alloca.ty(), TY_U32);
        assert_eq!(
            &field_0_alloca.value_output().users,
            &thin_set![ValueUser::Input {
                consumer: loop_node,
                input: 3,
            }]
        );

        let ValueOrigin::Output {
            producer: field_1_alloca_node,
            output: 0,
        } = loop_data.value_inputs()[4].origin
        else {
            panic!("the third input to the switch node should be the first output of a node")
        };

        let field_1_alloca = rvsdg[field_1_alloca_node].expect_op_alloca();

        assert_eq!(field_1_alloca.ty(), TY_U32);
        assert_eq!(
            &field_1_alloca.value_output().users,
            &thin_set![ValueUser::Input {
                consumer: loop_node,
                input: 4,
            }]
        );

        let load = rvsdg[load_node].expect_op_load();

        assert_eq!(load.ptr_input().origin, ValueOrigin::Argument(3));
        assert_eq!(
            &load.value_output().users,
            &thin_set![ValueUser::Input {
                consumer: add_one_node,
                input: 0,
            }]
        );

        let arguments = rvsdg[loop_region].value_arguments();

        assert_eq!(arguments.len(), 5);

        assert_eq!(arguments[3].ty, field_ptr_ty);
        assert_eq!(
            &arguments[3].users,
            &thin_set![
                ValueUser::Result(4),
                ValueUser::Input {
                    consumer: store_node,
                    input: 0,
                },
                ValueUser::Input {
                    consumer: load_node,
                    input: 0,
                }
            ]
        );

        assert_eq!(arguments[4].ty, field_ptr_ty);
        assert_eq!(arguments[4].users, thin_set![ValueUser::Result(5)]);

        let results = rvsdg[loop_region].value_results();

        assert_eq!(results.len(), 6);

        assert_eq!(results[4].ty, field_ptr_ty);
        assert_eq!(results[4].origin, ValueOrigin::Argument(3));

        assert_eq!(results[5].ty, field_ptr_ty);
        assert_eq!(results[5].origin, ValueOrigin::Argument(4));

        assert!(!rvsdg.is_live_node(op_ptr_field_ptr_node));
    }
}
