//! Rewrites slice-pointer-type loop-values whose recurrence accumulates a slice offset, such that
//! the slice-pointer-type loop-value itself becomes "offset-free".
//!
//! After the transformation, the accumulated offset is instead carried by a new separate `u32`
//! loop-value.
//!
//! This is a preparatory transform for [loop_pointer_normalization]. One of the assumptions made by
//! [loop_pointer_normalization] is that the loop-values it normalizes do not iteratively
//! self-refine; there exists a fixed set of pointer "shapes" that does not continue to increase
//! as the loop iterates. However, slice-pointer-type loop-values that are refined each iteration
//! by adding additional offset into the slice violate this assumption.
//!
//! The transform splits a slice pointer's accumulated offset into two components:
//!
//! - The *absolute base offset*: the offset already accumulated on the value-flow graph feeding
//!   into the loop node as one of its inputs. This remains unaffected by this transform. The
//!   later [offset_slice_replacement] will process these after the [memory_transform] completes.
//! - The *loop-relative offset*: the offset accumulated since entering the loop. We add a new
//!   `u32` loop-value to represent this value. Its initial input is `0`. At the end of each
//!   iteration, we take the initial offset value for the iteration from the loop-region argument,
//!   and add to it the total offset accumulated during the iteration. We then feed this value into
//!   the loop-region result to represent the offset value at the start of the next iteration.
//!
//! These two components are then recombined through two [OpOffsetSlice] nodes: one inside the
//! loop-region to represent the normalized slice-pointer-value during iteration, and one in the
//! loop node's outer region, to represent the normalized slice-pointer-value after the loop. All
//! users of the original loop-region argument/loop output are then reconnected to the outputs of
//! these new [OpOffsetSlice] nodes.
//!
//! While this transform operates on a single entry loop-value, we may have to normalize other
//! loop-values if such loop-values are reachable by the reverse value-flow from the loop-region
//! result of the entry loop-value (in fact, the entry loop-value does not always itself need
//! normalization). We refer to such other loop-values as the "dependencies" of the entry
//! loop-value. The entry loop-value and its dependencies together form the "reachable" set of
//! loop-values. Not every loop-value in the reachable set needs to be normalized; only if the
//! loop-value has loop-carried slice offset does it need to be normalized. This means that in the
//! reverse value-flow graph, as traced from the loop-region result, must include an [OpOffsetSlice]
//! node, and its input pointer value must originate from a slice-pointer-type loop-region argument.
//! If the pointer originates from a pointer-refining operation (e.g. [OpFieldPtr]), then it cannot
//! carry offset across iterations, as such pointer-refining operations always create fresh slice
//! pointers without any offset (it always points to the start of the (unsized) array); no slice
//! offset was carried over from the previous iteration. We call the set of loop-values that need
//! normalization the "affected" set. Note that the affected set may be empty; in that case this
//! transform does nothing.
//!
//! We do not trace reverse value-flow past [OpLoad] nodes. If a trace reaches the output of an
//! [OpLoad] node, then the loop-value cannot be normalized and we return an error. However, this
//! transform is meant to run on loop-values classified as "variable" by
//! [memory_promotion_and_legalization]. As [memory_promotion_and_legalization] also performs a
//! reverse value-flow analysis, any such [OpLoad] nodes would have resulted in a "blocked"
//! classification rather than a "variable" classification, and thus we should never encounter this
//! situation here.
//!
//! Note that a rewritten base loop-value is not necessarily loop-invariant, it is only guaranteed
//! to be *offset-free*. A recurrence that selects between different pointer values stays variant.
//! It is then up to [loop_pointer_normalization] to rewrite the graph to contain only
//! loop-invariant pointer values.
//!
//! [loop_pointer_normalization]: crate::rvsdg::transform::loop_pointer_normalization
//! [memory_promotion_and_legalization]: crate::rvsdg::transform::memory_promotion_and_legalization
//! [offset_slice_elaboration]: crate::rvsdg::transform::offset_slice_elaboration
//! [offset_slice_replacement]: crate::rvsdg::transform::offset_slice_replacement
use indexmap::IndexSet;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::BinaryOperator;
use crate::rvsdg::transform::pointer_reconstruction::PointerReconstructionError;
use crate::rvsdg::{
    Connectivity, Node, NodeKind, Region, Rvsdg, SimpleNode, ValueInput, ValueOrigin, ValueUser,
};
use crate::ty::{TY_U32, Type};

fn is_loop_value_invariant(rvsdg: &Rvsdg, loop_region: Region, loop_value: u32) -> bool {
    let result_index = loop_value + 1;
    let result = rvsdg[loop_region].value_results()[result_index as usize];

    result.origin == ValueOrigin::Argument(loop_value)
}

/// Reconnects all users of the `original_origin` to the `new_origin`, except when the user is a
/// value-input on the `excluded_node`.
fn reconnect_value_users_except(
    rvsdg: &mut Rvsdg,
    region: Region,
    original_origin: ValueOrigin,
    new_origin: ValueOrigin,
    excluded_node: Node,
) {
    let user_count = match original_origin {
        ValueOrigin::Argument(argument) => rvsdg[region].value_arguments()[argument as usize]
            .users
            .len(),
        ValueOrigin::Output { producer, output } => {
            rvsdg[producer].value_outputs()[output as usize].users.len()
        }
    };

    for i in (0..user_count).rev() {
        let user = match original_origin {
            ValueOrigin::Argument(argument) => {
                rvsdg[region].value_arguments()[argument as usize].users[i]
            }
            ValueOrigin::Output { producer, output } => {
                rvsdg[producer].value_outputs()[output as usize].users[i]
            }
        };

        if matches!(user, ValueUser::Input { consumer, .. } if consumer == excluded_node) {
            continue;
        }

        rvsdg.reconnect_value_user(region, user, new_origin);
    }
}

/// The loop-relative offset value accumulated by (a segment of) a loop-region result's reverse
/// value-flow graph.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RelativeOffset {
    Zero,
    Value(ValueOrigin),
}

/// The result of analyzing the reverse value-flow graph of a loop-region result.
struct RecurrenceAnalysis {
    loop_node: Node,

    /// Whether the value-flow concerns a slice pointer to which "offset" is applied.
    carries_offset: bool,

    /// The set of all loop-region arguments reached by the reverse value-flow of the loop-region
    /// result, including arguments reached through pointer-refining operations.
    ///
    /// Pointer-refining operations ([OpElementPtr], [OpFieldPtr]) always produce "fresh" pointer
    /// values that do not carry any slice offset. Therefore, if the loop-region result can only
    /// reach a given argument via a pointer-refining operation, this implies that this argument
    /// cannot contribute slice offset to this recurrence.
    reached_arguments: FxHashSet<u32>,

    /// The set of loop-region arguments reached by the reverse value-flow without passing through
    /// pointer-refining operations ([OpElementPtr], [OpFieldPtr]).
    ///
    /// This sub-set of the [reached_arguments] could potentially carry slice offset from a previous
    /// iteration into the next iteration.
    reached_arguments_with_offset: FxHashSet<u32>,

    /// The set of `(region, origin, on-chain)` triples already visited by this analysis.
    ///
    /// Once an origin has been visited in a given mode, its contributions to `carries_offset`,
    /// `chain_arguments` and `reached_arguments` have been recorded, so revisiting it can be
    /// skipped.
    ///
    /// Note that this is not a minor optimization: the unconditional revisiting of reverse
    /// value-flow sub-graphs from multiple users can lead to a combinatorial explosion, potentially
    /// resulting in exponential time complexity relative to graph depth.
    visited: FxHashSet<(Region, ValueOrigin, bool)>,
}

impl RecurrenceAnalysis {
    fn for_loop_value(
        rvsdg: &Rvsdg,
        loop_node: Node,
        loop_value: u32,
    ) -> Result<Self, PointerReconstructionError> {
        let mut analysis = RecurrenceAnalysis {
            loop_node,
            carries_offset: false,
            reached_arguments: Default::default(),
            reached_arguments_with_offset: Default::default(),
            visited: Default::default(),
        };

        let loop_region = rvsdg[loop_node].expect_loop().loop_region();
        let result = rvsdg[loop_region].value_results()[loop_value as usize + 1];

        analysis.analyze_value_origin(rvsdg, loop_region, result.origin, true)?;

        Ok(analysis)
    }

    fn analyze_value_origin(
        &mut self,
        rvsdg: &Rvsdg,
        region: Region,
        origin: ValueOrigin,
        carries_offset: bool,
    ) -> Result<(), PointerReconstructionError> {
        use NodeKind::*;
        use SimpleNode::*;

        if !self.visited.insert((region, origin, carries_offset)) {
            return Ok(());
        }

        match origin {
            ValueOrigin::Argument(argument) => {
                let owner = rvsdg[region].owner();

                if owner == self.loop_node {
                    self.reached_arguments.insert(argument);

                    if carries_offset {
                        self.reached_arguments_with_offset.insert(argument);
                    }

                    return Ok(());
                }

                // The walk never descends into nested loop regions, and never ascends past the
                // provoking loop's region, so if the owner node is not the provoking loop node,
                // then it must be a switch node; continue the trace from the switch node's
                // corresponding value-input.
                let outer_region = rvsdg[owner].region();
                let input_origin = rvsdg[owner].value_inputs()[argument as usize + 1].origin;

                self.analyze_value_origin(rvsdg, outer_region, input_origin, carries_offset)
            }
            ValueOrigin::Output { producer, output } => match rvsdg[producer].kind() {
                Switch(_) => {
                    let branch_count = rvsdg[producer].expect_switch().branches().len();

                    for i in 0..branch_count {
                        let branch = rvsdg[producer].expect_switch().branches()[i];
                        let branch_origin = rvsdg[branch].value_results()[output as usize].origin;

                        self.analyze_value_origin(rvsdg, branch, branch_origin, carries_offset)?;
                    }

                    Ok(())
                }
                Loop(_) => {
                    let inner_loop_region = rvsdg[producer].expect_loop().loop_region();

                    if is_loop_value_invariant(rvsdg, inner_loop_region, output) {
                        // The inner loop-value is loop-invariant, so its output value equals its
                        // input value; pass through to the input origin.
                        let input_origin = rvsdg[producer].value_inputs()[output as usize].origin;

                        self.analyze_value_origin(rvsdg, region, input_origin, carries_offset)
                    } else {
                        Err(PointerReconstructionError::NeedsLoopPointerNormalization {
                            loop_node: producer,
                            loop_value: output,
                        })
                    }
                }
                Simple(OpOffsetSlice(_)) => {
                    if carries_offset {
                        self.carries_offset = true;
                    }

                    let ptr_origin = rvsdg[producer].expect_op_offset_slice().ptr_input().origin;

                    self.analyze_value_origin(rvsdg, region, ptr_origin, carries_offset)
                }
                Simple(OpFieldPtr(_)) => {
                    // A pointer-refining node: this always produces a fresh pointer value that does
                    // not yet carry any offset. This implies that this value-flow path has no
                    // loop-carried offset.
                    let ptr_origin = rvsdg[producer].expect_op_field_ptr().ptr_input().origin;

                    self.analyze_value_origin(rvsdg, region, ptr_origin, false)
                }
                Simple(OpElementPtr(_)) => {
                    // See the OpFieldPtr arm above.
                    let ptr_origin = rvsdg[producer].expect_op_element_ptr().ptr_input().origin;

                    self.analyze_value_origin(rvsdg, region, ptr_origin, false)
                }
                Simple(OpVariantPtr(_) | OpExtractField(_) | OpExtractElement(_)) => Ok(()),
                Simple(OpLoad(_)) => Err(PointerReconstructionError::NeedsPromotion(producer)),
                Simple(OpAlloca(_) | ConstFallback(_)) => panic!(
                    "pointers to values created inside the loop-region scope should not recur; \
                        a value cannot outlive its scope"
                ),
                _ => panic!("node kind cannot output a pointer"),
            },
        }
    }
}

/// Represents a rewrite job for an affected loop-value.
struct Rewrite {
    /// The loop-value that is to be rewritten.
    loop_value: u32,

    /// The type of the loop-value.
    ty: Type,

    /// The loop-value that was created to represent the loop-relative offset.
    offset_loop_value: u32,
}

struct SliceOffsetNormalizer {
    loop_node: Node,
    loop_region: Region,
    relative_offset_cache: FxHashMap<(Region, ValueOrigin), RelativeOffset>,
    base_pointer_cache: FxHashMap<(Region, ValueOrigin), ValueOrigin>,
}

impl SliceOffsetNormalizer {
    fn new(rvsdg: &Rvsdg, loop_node: Node) -> Self {
        let loop_region = rvsdg[loop_node].expect_loop().loop_region();

        Self {
            loop_node,
            loop_region,
            relative_offset_cache: FxHashMap::default(),
            base_pointer_cache: FxHashMap::default(),
        }
    }

    fn normalize(
        &mut self,
        rvsdg: &mut Rvsdg,
        loop_value: u32,
    ) -> Result<bool, PointerReconstructionError> {
        // Find the set of loop-values that need to be co-normalized. If the analysis finds that
        // loop-values for a nested loop must be normalized first, we early exit with a
        // NeedsLoopPointerNormalization error. The loop-pointer-normalization machinery will then
        // retry normalization for the current loop node later.
        let affected = self.collect_affected_loop_values(rvsdg, loop_value)?;

        if affected.is_empty() {
            return Ok(false);
        }

        let outer_region = rvsdg[self.loop_node].region();

        // Add a zero-initialized `u32` offset loop-value and an in-loop OpOffsetSlice node that
        // uses this loop-value to apply the appropriate slice offset on each iteration. We then
        // reconnect all original users of the original loop-value's loop-region argument to the
        // output of this new OpOffsetSlice node.
        //
        // Note that we must first do this for every affected loop-value, before we can start
        // resolving the new "offset" and "stripped base" result values; if there are dependencies
        // between loop-values, then we want to ensure we trace to the output of the new
        // OpOffsetSlice node so that the dependent value properly incorporates the loop-carried
        // offset of its dependency.
        let mut rewrites = Vec::with_capacity(affected.len());

        for loop_value in affected {
            let ty = rvsdg[self.loop_region].value_arguments()[loop_value as usize].ty;

            let zero_node = rvsdg.add_const_u32(outer_region, 0);
            let offset_loop_value =
                rvsdg.add_loop_input(self.loop_node, ValueInput::output(TY_U32, zero_node, 0));

            let op_offset_slice = rvsdg.add_op_offset_slice(
                self.loop_region,
                ValueInput::argument(ty, loop_value),
                ValueInput::argument(TY_U32, offset_loop_value),
            );

            reconnect_value_users_except(
                rvsdg,
                self.loop_region,
                ValueOrigin::Argument(loop_value),
                ValueOrigin::Output {
                    producer: op_offset_slice,
                    output: 0,
                },
                op_offset_slice,
            );

            rewrites.push(Rewrite {
                loop_value,
                ty,
                offset_loop_value,
            });
        }

        // We can now proceed to resolve the offset recurrence and the base pointer recurrence for
        // every affected loop-value.
        for rewrite in &rewrites {
            let result_origin =
                rvsdg[self.loop_region].value_results()[rewrite.loop_value as usize + 1].origin;

            let offset = self.resolve_relative_offset(rvsdg, self.loop_region, result_origin);

            let offset_origin = match offset {
                RelativeOffset::Zero => {
                    let zero_node = rvsdg.add_const_u32(self.loop_region, 0);

                    ValueOrigin::Output {
                        producer: zero_node,
                        output: 0,
                    }
                }
                RelativeOffset::Value(origin) => origin,
            };

            rvsdg.reconnect_region_result(
                self.loop_region,
                rewrite.offset_loop_value + 1,
                offset_origin,
            );

            let base_pointer_origin =
                self.resolve_base_pointer(rvsdg, self.loop_region, result_origin);

            rvsdg.reconnect_region_result(
                self.loop_region,
                rewrite.loop_value + 1,
                base_pointer_origin,
            );
        }

        // Lastly, reconnect all users of the original loop-value output to a new OpSliceOffset
        // node that applies our new loop-carried offset value after the last iteration.
        for rewrite in &rewrites {
            let op_offset_slice = rvsdg.add_op_offset_slice(
                outer_region,
                ValueInput::output(rewrite.ty, self.loop_node, rewrite.loop_value),
                ValueInput::output(TY_U32, self.loop_node, rewrite.offset_loop_value),
            );

            reconnect_value_users_except(
                rvsdg,
                outer_region,
                ValueOrigin::Output {
                    producer: self.loop_node,
                    output: rewrite.loop_value,
                },
                ValueOrigin::Output {
                    producer: op_offset_slice,
                    output: 0,
                },
                op_offset_slice,
            );
        }

        Ok(true)
    }

    /// Collects the set of loop-values that must be co-normalized before the given `loop_value` can
    /// be normalized by the loop-pointer-normalization transform.
    fn collect_affected_loop_values(
        &self,
        rvsdg: &Rvsdg,
        loop_value: u32,
    ) -> Result<IndexSet<u32>, PointerReconstructionError> {
        let mut analyses = Vec::new();
        let mut dep_group = FxHashSet::default();
        let mut stack = vec![loop_value];

        dep_group.insert(loop_value);

        while let Some(loop_value) = stack.pop() {
            let result = rvsdg[self.loop_region].value_results()[loop_value as usize + 1];

            if result.origin == ValueOrigin::Argument(loop_value) {
                // If the loop-value is loop-invariant, then it cannot carry offset and it does not
                // have any dependencies, so we can ignore it.
                continue;
            }

            let analysis = RecurrenceAnalysis::for_loop_value(rvsdg, self.loop_node, loop_value)?;

            // Extend the dependency group with the newly reached arguments. Sort them first, so
            // that the order in which we'll end up creating the new "offset" loop-values is
            // predictable (for testing).
            let mut extension = analysis
                .reached_arguments
                .iter()
                .copied()
                .collect::<Vec<_>>();

            extension.sort_unstable();

            for argument in extension {
                if dep_group.insert(argument) {
                    stack.push(argument);
                }
            }

            analyses.push((loop_value, analysis));
        }

        let mut affected_set = IndexSet::default();

        // A member of the dependency group is "affected" if that loop-value's single iteration
        // value-flow graph "carries offset".
        for (loop_value, analysis) in &analyses {
            if analysis.carries_offset {
                affected_set.insert(*loop_value);
            }
        }

        // A member of the dependency group is also "affected" if it reaches any loop-region
        // argument for another loop-value that "carries offset". We'll iteratively add more
        // affected loop-values to the affected set until the set stabilizes.
        let mut changed = true;

        while changed {
            changed = false;

            for (loop_value, analysis) in &analyses {
                if !affected_set.contains(loop_value)
                    && analysis
                        .reached_arguments_with_offset
                        .iter()
                        .any(|argument| affected_set.contains(argument))
                {
                    affected_set.insert(*loop_value);

                    changed = true;
                }
            }
        }

        Ok(affected_set)
    }

    /// Resolves a loop-relative offset value for the given pointer origin.
    fn resolve_relative_offset(
        &mut self,
        rvsdg: &mut Rvsdg,
        region: Region,
        ptr_origin: ValueOrigin,
    ) -> RelativeOffset {
        if let Some(offset) = self.relative_offset_cache.get(&(region, ptr_origin)) {
            *offset
        } else {
            let offset = self.relative_offset_for_origin(rvsdg, region, ptr_origin);

            self.relative_offset_cache
                .insert((region, ptr_origin), offset);

            offset
        }
    }

    fn relative_offset_for_origin(
        &mut self,
        rvsdg: &mut Rvsdg,
        region: Region,
        ptr_origin: ValueOrigin,
    ) -> RelativeOffset {
        match ptr_origin {
            ValueOrigin::Argument(argument) => {
                self.relative_offset_for_argument(rvsdg, region, argument)
            }
            ValueOrigin::Output { producer, output } => {
                self.relative_offset_for_output(rvsdg, producer, output)
            }
        }
    }

    fn relative_offset_for_argument(
        &mut self,
        rvsdg: &mut Rvsdg,
        region: Region,
        ptr_argument: u32,
    ) -> RelativeOffset {
        let owner = rvsdg[region].owner();

        if owner == self.loop_node {
            // Arguments of the loop being processed contribute no loop-relative offset; the
            // absolute offset of the value-flow feeding into the loop is resolved by the
            // offset-slice-replacement transform (after the memory-transform).
            return RelativeOffset::Zero;
        }

        match rvsdg[owner].kind() {
            NodeKind::Switch(_) => {
                self.relative_offset_for_switch_branch_argument(rvsdg, region, ptr_argument)
            }
            _ => unreachable!("the walk can only visit the loop region and switch branches"),
        }
    }

    fn relative_offset_for_output(
        &mut self,
        rvsdg: &mut Rvsdg,
        node: Node,
        ptr_output: u32,
    ) -> RelativeOffset {
        use NodeKind::*;
        use SimpleNode::*;

        match rvsdg[node].kind() {
            Switch(_) => self.relative_offset_for_switch_output(rvsdg, node, ptr_output),
            Loop(_) => self.relative_offset_for_loop_output(rvsdg, node, ptr_output),
            Simple(OpOffsetSlice(_)) => self.relative_offset_for_op_offset_slice(rvsdg, node),
            Simple(
                OpAlloca(_) | OpFieldPtr(_) | OpElementPtr(_) | OpVariantPtr(_) | OpExtractField(_)
                | OpExtractElement(_) | ConstFallback(_),
            ) => RelativeOffset::Zero,
            _ => panic!("unexpected node kind"),
        }
    }

    fn relative_offset_for_op_offset_slice(
        &mut self,
        rvsdg: &mut Rvsdg,
        node: Node,
    ) -> RelativeOffset {
        let region = rvsdg[node].region();
        let data = rvsdg[node].expect_op_offset_slice();
        let ptr_origin = data.ptr_input().origin;
        let added_offset_origin = data.offset_input().origin;
        let prior_offset = self.resolve_relative_offset(rvsdg, region, ptr_origin);

        match prior_offset {
            RelativeOffset::Zero => RelativeOffset::Value(added_offset_origin),
            RelativeOffset::Value(prior_offset_origin) => {
                let combined = rvsdg.add_op_binary(
                    region,
                    BinaryOperator::Add,
                    ValueInput {
                        ty: TY_U32,
                        origin: prior_offset_origin,
                    },
                    ValueInput {
                        ty: TY_U32,
                        origin: added_offset_origin,
                    },
                );

                RelativeOffset::Value(ValueOrigin::Output {
                    producer: combined,
                    output: 0,
                })
            }
        }
    }

    fn relative_offset_for_switch_output(
        &mut self,
        rvsdg: &mut Rvsdg,
        switch_node: Node,
        ptr_output: u32,
    ) -> RelativeOffset {
        let branch_count = rvsdg[switch_node].expect_switch().branches().len();

        let mut branch_offsets = Vec::with_capacity(branch_count);

        for i in 0..branch_count {
            let branch = rvsdg[switch_node].expect_switch().branches()[i];
            let ptr_origin = rvsdg[branch].value_results()[ptr_output as usize].origin;

            branch_offsets.push(self.resolve_relative_offset(rvsdg, branch, ptr_origin));
        }

        // If no branch contributes an offset, we don't need an offset output on this switch node.
        if branch_offsets
            .iter()
            .all(|offset| *offset == RelativeOffset::Zero)
        {
            return RelativeOffset::Zero;
        }

        let offset_output = rvsdg.add_switch_output(switch_node, TY_U32);

        for (i, branch_offset) in branch_offsets.iter().enumerate() {
            let branch = rvsdg[switch_node].expect_switch().branches()[i];

            let offset_origin = match branch_offset {
                RelativeOffset::Zero => {
                    let zero = rvsdg.add_const_u32(branch, 0);

                    ValueOrigin::Output {
                        producer: zero,
                        output: 0,
                    }
                }
                RelativeOffset::Value(offset_origin) => *offset_origin,
            };

            rvsdg.reconnect_region_result(branch, offset_output, offset_origin);
        }

        RelativeOffset::Value(ValueOrigin::Output {
            producer: switch_node,
            output: offset_output,
        })
    }

    fn relative_offset_for_switch_branch_argument(
        &mut self,
        rvsdg: &mut Rvsdg,
        branch: Region,
        ptr_argument: u32,
    ) -> RelativeOffset {
        let ptr_input = ptr_argument + 1;
        let switch_node = rvsdg[branch].owner();
        let outer_region = rvsdg[switch_node].region();
        let ptr_origin = rvsdg[switch_node].value_inputs()[ptr_input as usize].origin;
        let outer_offset = self.resolve_relative_offset(rvsdg, outer_region, ptr_origin);

        let inner_offset = match outer_offset {
            RelativeOffset::Zero => RelativeOffset::Zero,
            RelativeOffset::Value(origin) => {
                let input = rvsdg.add_switch_input(switch_node, ValueInput { ty: TY_U32, origin });

                RelativeOffset::Value(ValueOrigin::Argument(input - 1))
            }
        };

        // Make sure we only do this for the first branch we encounter by adding a cache entry for
        // every branch; other branches should short-circuit to using this cached value.
        for branch in rvsdg[switch_node].expect_switch().branches() {
            self.relative_offset_cache
                .insert((*branch, ValueOrigin::Argument(ptr_argument)), inner_offset);
        }

        inner_offset
    }

    fn relative_offset_for_loop_output(
        &mut self,
        rvsdg: &mut Rvsdg,
        loop_node: Node,
        ptr_output: u32,
    ) -> RelativeOffset {
        let region = rvsdg[loop_node].region();
        let loop_region = rvsdg[loop_node].expect_loop().loop_region();

        assert!(
            is_loop_value_invariant(rvsdg, loop_region, ptr_output),
            "the analysis phase should have reported a non-invariant nested loop-value"
        );

        // The inner loop-value is loop-invariant, so its output value equals its input value; pass
        // through to the input origin.
        let input_origin = rvsdg[loop_node].value_inputs()[ptr_output as usize].origin;

        self.resolve_relative_offset(rvsdg, region, input_origin)
    }

    /// Resolves a base pointer value that does not carry any relative offset.
    fn resolve_base_pointer(
        &mut self,
        rvsdg: &mut Rvsdg,
        region: Region,
        ptr_origin: ValueOrigin,
    ) -> ValueOrigin {
        if let Some(origin) = self.base_pointer_cache.get(&(region, ptr_origin)) {
            *origin
        } else {
            let origin = self.base_pointer_for_origin(rvsdg, region, ptr_origin);

            self.base_pointer_cache.insert((region, ptr_origin), origin);

            origin
        }
    }

    fn base_pointer_for_origin(
        &mut self,
        rvsdg: &mut Rvsdg,
        region: Region,
        ptr_origin: ValueOrigin,
    ) -> ValueOrigin {
        match ptr_origin {
            ValueOrigin::Argument(argument) => {
                self.base_pointer_for_argument(rvsdg, region, argument)
            }
            ValueOrigin::Output { producer, output } => {
                self.base_pointer_for_output(rvsdg, producer, output)
            }
        }
    }

    fn base_pointer_for_argument(
        &mut self,
        rvsdg: &mut Rvsdg,
        region: Region,
        ptr_argument: u32,
    ) -> ValueOrigin {
        let owner = rvsdg[region].owner();

        if owner == self.loop_node {
            // Arguments of the loop being processed are offset-free roots.
            return ValueOrigin::Argument(ptr_argument);
        }

        match rvsdg[owner].kind() {
            NodeKind::Switch(_) => {
                self.base_pointer_for_switch_branch_argument(rvsdg, region, ptr_argument)
            }
            _ => unreachable!("the walk can only visit the loop region and switch branches"),
        }
    }

    fn base_pointer_for_output(
        &mut self,
        rvsdg: &mut Rvsdg,
        node: Node,
        ptr_output: u32,
    ) -> ValueOrigin {
        use NodeKind::*;
        use SimpleNode::*;

        match rvsdg[node].kind() {
            Switch(_) => self.base_pointer_for_switch_output(rvsdg, node, ptr_output),
            Loop(_) => self.base_pointer_for_loop_output(rvsdg, node, ptr_output),
            Simple(OpOffsetSlice(_)) => {
                let region = rvsdg[node].region();
                let ptr_origin = rvsdg[node].expect_op_offset_slice().ptr_input().origin;

                self.resolve_base_pointer(rvsdg, region, ptr_origin)
            }
            Simple(
                OpAlloca(_) | OpFieldPtr(_) | OpElementPtr(_) | OpVariantPtr(_) | OpExtractField(_)
                | OpExtractElement(_) | ConstFallback(_),
            ) => ValueOrigin::Output {
                producer: node,
                output: ptr_output,
            },
            _ => panic!("unexpected node kind"),
        }
    }

    fn base_pointer_for_switch_output(
        &mut self,
        rvsdg: &mut Rvsdg,
        switch_node: Node,
        ptr_output: u32,
    ) -> ValueOrigin {
        let branch_count = rvsdg[switch_node].expect_switch().branches().len();

        let mut branch_origins = Vec::with_capacity(branch_count);

        for i in 0..branch_count {
            let branch = rvsdg[switch_node].expect_switch().branches()[i];
            let ptr_origin = rvsdg[branch].value_results()[ptr_output as usize].origin;

            branch_origins.push(self.resolve_base_pointer(rvsdg, branch, ptr_origin));
        }

        // Pass-through collapsing: if every branch resolves to the branch argument of one and the
        // same switch input, then we don't need a new switch output; we can use this input's origin
        // in the outer region directly.
        if let ValueOrigin::Argument(argument) = branch_origins[0]
            && branch_origins
                .iter()
                .all(|origin| *origin == branch_origins[0])
        {
            return rvsdg[switch_node].value_inputs()[argument as usize + 1].origin;
        }

        let ty = rvsdg[switch_node].value_outputs()[ptr_output as usize].ty;
        let stripped_output = rvsdg.add_switch_output(switch_node, ty);

        for (i, branch_origin) in branch_origins.iter().enumerate() {
            let branch = rvsdg[switch_node].expect_switch().branches()[i];

            rvsdg.reconnect_region_result(branch, stripped_output, *branch_origin);
        }

        ValueOrigin::Output {
            producer: switch_node,
            output: stripped_output,
        }
    }

    fn base_pointer_for_switch_branch_argument(
        &mut self,
        rvsdg: &mut Rvsdg,
        branch: Region,
        ptr_argument: u32,
    ) -> ValueOrigin {
        let ptr_input = ptr_argument + 1;
        let switch_node = rvsdg[branch].owner();
        let outer_region = rvsdg[switch_node].region();
        let input = rvsdg[switch_node].value_inputs()[ptr_input as usize];
        let outer_base_pointer = self.resolve_base_pointer(rvsdg, outer_region, input.origin);

        let inner_origin = if outer_base_pointer == input.origin {
            // The input pointer is already offset-free, so the existing branch argument is its base
            // pointer.
            ValueOrigin::Argument(ptr_argument)
        } else {
            let new_input = rvsdg.add_switch_input(
                switch_node,
                ValueInput {
                    ty: input.ty,
                    origin: outer_base_pointer,
                },
            );

            ValueOrigin::Argument(new_input - 1)
        };

        // Make sure we only do this for the first branch we encounter by adding a cache entry for
        // every branch; other branches should short-circuit to using this cached value.
        for branch in rvsdg[switch_node].expect_switch().branches() {
            self.base_pointer_cache
                .insert((*branch, ValueOrigin::Argument(ptr_argument)), inner_origin);
        }

        inner_origin
    }

    fn base_pointer_for_loop_output(
        &mut self,
        rvsdg: &mut Rvsdg,
        loop_node: Node,
        ptr_output: u32,
    ) -> ValueOrigin {
        let region = rvsdg[loop_node].region();
        let loop_region = rvsdg[loop_node].expect_loop().loop_region();

        assert!(
            is_loop_value_invariant(rvsdg, loop_region, ptr_output),
            "the analysis phase should have reported a non-invariant inner loop-value"
        );

        // The inner loop-value is loop-invariant, so its output value equals its input value; pass
        // through to the input origin.
        let input_origin = rvsdg[loop_node].value_inputs()[ptr_output as usize].origin;

        self.resolve_base_pointer(rvsdg, region, input_origin)
    }
}

/// Normalizes slice-pointer-type loop-values in the dependency group of the given `loop_value` of
/// the given `loop_node` that carry slice offset across iterations.
///
/// See the module-level documentation for details.
pub fn normalize_loop_slice_offsets(
    rvsdg: &mut Rvsdg,
    loop_node: Node,
    loop_value: u32,
) -> Result<bool, PointerReconstructionError> {
    SliceOffsetNormalizer::new(rvsdg, loop_node).normalize(rvsdg, loop_value)
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::rvsdg::ValueOutput;
    use crate::ty::{Int, TY_DUMMY, TY_PREDICATE, TypeKind};
    use crate::{FnSig, Function, Module, Symbol};

    #[test]
    fn test_self_recurrence_through_switch() {
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

        let array_ty = module.ty.register(TypeKind::Array {
            element_ty: TY_U32,
            count: 4,
            stride: 4,
        });
        let array_ptr_ty = module.ty.register(TypeKind::Ptr(array_ty));
        let slice_ty = module.ty.register(TypeKind::Slice {
            element_ty: TY_U32,
            stride: 4,
        });
        let slice_ptr_ty = module.ty.register(TypeKind::Ptr(slice_ty));

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let array_node = rvsdg.add_op_alloca(region, array_ty);
        let offset_node = rvsdg.add_const_u32(region, 0);
        let initial_slice_node = rvsdg.add_op_offset_slice(
            region,
            ValueInput::output(array_ptr_ty, array_node, 0),
            ValueInput::output(TY_U32, offset_node, 0),
        );

        let (loop_node, loop_region) = rvsdg.add_loop(
            region,
            vec![ValueInput::output(slice_ptr_ty, initial_slice_node, 0)],
            None,
        );

        let case_node = rvsdg.add_const_u32(loop_region, 0);
        let selector_node = rvsdg.add_op_case_to_branch_selector(
            loop_region,
            ValueInput::output(TY_U32, case_node, 0),
            Int::U32,
            [0],
        );
        let switch_node = rvsdg.add_switch(
            loop_region,
            vec![
                ValueInput::output(TY_PREDICATE, selector_node, 0),
                ValueInput::argument(slice_ptr_ty, 0),
            ],
            vec![ValueOutput::new(slice_ptr_ty)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);
        let advance_offset_node = rvsdg.add_const_u32(branch_0, 1);
        let advance_node = rvsdg.add_op_offset_slice(
            branch_0,
            ValueInput::argument(slice_ptr_ty, 0),
            ValueInput::output(TY_U32, advance_offset_node, 0),
        );

        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: advance_node,
                output: 0,
            },
        );

        let branch_1 = rvsdg.add_switch_branch(switch_node);

        rvsdg.reconnect_region_result(branch_1, 0, ValueOrigin::Argument(0));

        let reentry_predicate_node = rvsdg.add_const_bool(loop_region, false);

        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: reentry_predicate_node,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            loop_region,
            1,
            ValueOrigin::Output {
                producer: switch_node,
                output: 0,
            },
        );

        let get_offset_node =
            rvsdg.add_op_get_ptr_offset(region, ValueInput::output(slice_ptr_ty, loop_node, 0));

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: get_offset_node,
                output: 0,
            },
        );

        let did_normalize = normalize_loop_slice_offsets(&mut rvsdg, loop_node, 0).unwrap();

        assert!(did_normalize);

        // A new zero-initialized `u32` offset loop-value should have been added.
        assert_eq!(rvsdg[loop_node].expect_loop().value_inputs().len(), 2);
        assert_eq!(rvsdg[loop_node].expect_loop().value_inputs()[1].ty, TY_U32);

        let ValueOrigin::Output {
            producer: initial_offset_node,
            output: 0,
        } = rvsdg[loop_node].expect_loop().value_inputs()[1].origin
        else {
            panic!("the offset loop-value's input should connect to a node output");
        };

        assert_eq!(rvsdg[initial_offset_node].expect_const_u32().value(), 0);

        // The switch node's pointer input should now connect to a new in-loop OpOffsetSlice node,
        // which recombines the base pointer and the loop-relative offset argument.
        let ValueOrigin::Output {
            producer: re_anchor_node,
            output: 0,
        } = rvsdg[switch_node].value_inputs()[1].origin
        else {
            panic!("the switch node's pointer input should connect to a node output");
        };

        let re_anchor_data = rvsdg[re_anchor_node].expect_op_offset_slice();

        assert_eq!(re_anchor_data.ptr_input().origin, ValueOrigin::Argument(0));
        assert_eq!(
            re_anchor_data.offset_input().origin,
            ValueOrigin::Argument(1)
        );

        // The offset walk should have hoisted the offset argument into the switch node, and the
        // base pointer walk should have hoisted the base pointer argument.
        assert_eq!(rvsdg[switch_node].value_inputs().len(), 4);
        assert_eq!(
            rvsdg[switch_node].value_inputs()[2].origin,
            ValueOrigin::Argument(1)
        );
        assert_eq!(
            rvsdg[switch_node].value_inputs()[3].origin,
            ValueOrigin::Argument(0)
        );

        // The base pointer walk should have collapsed the pass-through recurrence, so the switch
        // node should only have gained a single output (for the loop-relative offset).
        assert_eq!(rvsdg[switch_node].value_outputs().len(), 2);
        assert_eq!(rvsdg[switch_node].value_outputs()[1].ty, TY_U32);

        // The base pointer loop-value should now be loop-invariant, with the offset recurrence
        // carried by the new offset loop-value.
        assert_eq!(
            rvsdg[loop_region].value_results()[1].origin,
            ValueOrigin::Argument(0)
        );
        assert_eq!(
            rvsdg[loop_region].value_results()[2].origin,
            ValueOrigin::Output {
                producer: switch_node,
                output: 1,
            }
        );

        // The first branch should add `1` to the hoisted offset argument.
        let ValueOrigin::Output {
            producer: add_node,
            output: 0,
        } = rvsdg[branch_0].value_results()[1].origin
        else {
            panic!("the first branch's offset result should connect to a node output");
        };

        let add_data = rvsdg[add_node].expect_op_binary();

        assert_eq!(add_data.operator(), BinaryOperator::Add);
        assert_eq!(add_data.lhs_input().origin, ValueOrigin::Argument(1));
        assert_eq!(
            add_data.rhs_input().origin,
            ValueOrigin::Output {
                producer: advance_offset_node,
                output: 0,
            }
        );

        // The second branch should pass the hoisted offset argument through unmodified.
        assert_eq!(
            rvsdg[branch_1].value_results()[1].origin,
            ValueOrigin::Argument(1)
        );

        // The user of the loop's pointer output should have been reconnected to the output
        // re-anchor, which recombines the loop's base pointer output and offset output.
        let ValueOrigin::Output {
            producer: output_re_anchor_node,
            output: 0,
        } = rvsdg[get_offset_node]
            .expect_op_get_slice_offset()
            .ptr_input()
            .origin
        else {
            panic!("the get-offset node's pointer input should connect to a node output");
        };

        let output_re_anchor_data = rvsdg[output_re_anchor_node].expect_op_offset_slice();

        assert_eq!(
            output_re_anchor_data.ptr_input().origin,
            ValueOrigin::Output {
                producer: loop_node,
                output: 0,
            }
        );
        assert_eq!(
            output_re_anchor_data.offset_input().origin,
            ValueOrigin::Output {
                producer: loop_node,
                output: 1,
            }
        );

        // The original offset chain should have been left intact.
        assert!(rvsdg.is_live_node(advance_node));
        assert_eq!(
            rvsdg[branch_0].value_results()[0].origin,
            ValueOrigin::Output {
                producer: advance_node,
                output: 0,
            }
        );
    }

    #[test]
    fn test_single_dependency_no_cycle() {
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

        let array_ty = module.ty.register(TypeKind::Array {
            element_ty: TY_U32,
            count: 4,
            stride: 4,
        });
        let array_ptr_ty = module.ty.register(TypeKind::Ptr(array_ty));
        let slice_ty = module.ty.register(TypeKind::Slice {
            element_ty: TY_U32,
            stride: 4,
        });
        let slice_ptr_ty = module.ty.register(TypeKind::Ptr(slice_ty));

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let array_0_node = rvsdg.add_op_alloca(region, array_ty);
        let offset_0_node = rvsdg.add_const_u32(region, 0);
        let initial_slice_0_node = rvsdg.add_op_offset_slice(
            region,
            ValueInput::output(array_ptr_ty, array_0_node, 0),
            ValueInput::output(TY_U32, offset_0_node, 0),
        );
        let array_1_node = rvsdg.add_op_alloca(region, array_ty);
        let offset_1_node = rvsdg.add_const_u32(region, 0);
        let initial_slice_1_node = rvsdg.add_op_offset_slice(
            region,
            ValueInput::output(array_ptr_ty, array_1_node, 0),
            ValueInput::output(TY_U32, offset_1_node, 0),
        );

        let (loop_node, loop_region) = rvsdg.add_loop(
            region,
            vec![
                ValueInput::output(slice_ptr_ty, initial_slice_0_node, 0),
                ValueInput::output(slice_ptr_ty, initial_slice_1_node, 0),
            ],
            None,
        );

        // Loop-value `1` advances itself; loop-value `0` depends on loop-value `1`'s argument.
        let advance_offset_node = rvsdg.add_const_u32(loop_region, 1);
        let advance_node = rvsdg.add_op_offset_slice(
            loop_region,
            ValueInput::argument(slice_ptr_ty, 1),
            ValueInput::output(TY_U32, advance_offset_node, 0),
        );

        let reentry_predicate_node = rvsdg.add_const_bool(loop_region, false);

        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: reentry_predicate_node,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(loop_region, 1, ValueOrigin::Argument(1));
        rvsdg.reconnect_region_result(
            loop_region,
            2,
            ValueOrigin::Output {
                producer: advance_node,
                output: 0,
            },
        );

        let get_offset_node =
            rvsdg.add_op_get_ptr_offset(region, ValueInput::output(slice_ptr_ty, loop_node, 0));

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: get_offset_node,
                output: 0,
            },
        );

        let did_normalize = normalize_loop_slice_offsets(&mut rvsdg, loop_node, 0).unwrap();

        assert!(did_normalize);

        // Both loop-values should have been rewritten: loop-value `1` carries an offset directly,
        // and loop-value `0` reaches loop-value `1`'s argument. Offset loop-value `2` belongs to
        // loop-value `1` and offset loop-value `3` belongs to loop-value `0`.
        assert_eq!(rvsdg[loop_node].expect_loop().value_inputs().len(), 4);

        // The advance node's pointer input should now connect to loop-value `1`'s new OpOffsetSlice
        // node.
        let ValueOrigin::Output {
            producer: re_anchor_node,
            output: 0,
        } = rvsdg[advance_node]
            .expect_op_offset_slice()
            .ptr_input()
            .origin
        else {
            panic!("the advance node's pointer input should connect to a node output");
        };

        let re_anchor_data = rvsdg[re_anchor_node].expect_op_offset_slice();

        assert_eq!(re_anchor_data.ptr_input().origin, ValueOrigin::Argument(1));
        assert_eq!(
            re_anchor_data.offset_input().origin,
            ValueOrigin::Argument(2)
        );

        // Loop-value `0`'s base pointer result should still select loop-value `1`'s argument, with
        // its offset result selecting loop-value `1`'s offset argument.
        assert_eq!(
            rvsdg[loop_region].value_results()[1].origin,
            ValueOrigin::Argument(1)
        );
        assert_eq!(
            rvsdg[loop_region].value_results()[4].origin,
            ValueOrigin::Argument(2)
        );

        // Loop-value `1`'s base pointer should now be loop-invariant.
        assert_eq!(
            rvsdg[loop_region].value_results()[2].origin,
            ValueOrigin::Argument(1)
        );

        // Additional offset should be added to loop-value `1`'s new offset value each iteration.
        let ValueOrigin::Output {
            producer: add_node,
            output: 0,
        } = rvsdg[loop_region].value_results()[3].origin
        else {
            panic!("loop-value `1`'s offset result should connect to a node output");
        };

        let add_data = rvsdg[add_node].expect_op_binary();

        assert_eq!(add_data.operator(), BinaryOperator::Add);
        assert_eq!(add_data.lhs_input().origin, ValueOrigin::Argument(2));
        assert_eq!(
            add_data.rhs_input().origin,
            ValueOrigin::Output {
                producer: advance_offset_node,
                output: 0,
            }
        );
    }

    #[test]
    fn test_two_value_dependency_cycle() {
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

        let array_ty = module.ty.register(TypeKind::Array {
            element_ty: TY_U32,
            count: 4,
            stride: 4,
        });
        let array_ptr_ty = module.ty.register(TypeKind::Ptr(array_ty));
        let slice_ty = module.ty.register(TypeKind::Slice {
            element_ty: TY_U32,
            stride: 4,
        });
        let slice_ptr_ty = module.ty.register(TypeKind::Ptr(slice_ty));

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let array_0_node = rvsdg.add_op_alloca(region, array_ty);
        let offset_0_node = rvsdg.add_const_u32(region, 0);
        let initial_slice_0_node = rvsdg.add_op_offset_slice(
            region,
            ValueInput::output(array_ptr_ty, array_0_node, 0),
            ValueInput::output(TY_U32, offset_0_node, 0),
        );
        let array_1_node = rvsdg.add_op_alloca(region, array_ty);
        let offset_1_node = rvsdg.add_const_u32(region, 0);
        let initial_slice_1_node = rvsdg.add_op_offset_slice(
            region,
            ValueInput::output(array_ptr_ty, array_1_node, 0),
            ValueInput::output(TY_U32, offset_1_node, 0),
        );

        let (loop_node, loop_region) = rvsdg.add_loop(
            region,
            vec![
                ValueInput::output(slice_ptr_ty, initial_slice_0_node, 0),
                ValueInput::output(slice_ptr_ty, initial_slice_1_node, 0),
            ],
            None,
        );

        let advance_offset_node = rvsdg.add_const_u32(loop_region, 1);
        let advance_node = rvsdg.add_op_offset_slice(
            loop_region,
            ValueInput::argument(slice_ptr_ty, 1),
            ValueInput::output(TY_U32, advance_offset_node, 0),
        );

        let reentry_predicate_node = rvsdg.add_const_bool(loop_region, false);

        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: reentry_predicate_node,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            loop_region,
            1,
            ValueOrigin::Output {
                producer: advance_node,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(loop_region, 2, ValueOrigin::Argument(0));

        let get_offset_node =
            rvsdg.add_op_get_ptr_offset(region, ValueInput::output(slice_ptr_ty, loop_node, 0));

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: get_offset_node,
                output: 0,
            },
        );

        let did_normalize = normalize_loop_slice_offsets(&mut rvsdg, loop_node, 0).unwrap();

        assert!(did_normalize);

        // Both loop-values should have been rewritten; offset loop-value `2` belongs to
        // loop-value `0` and offset loop-value `3` belongs to loop-value `1`.
        assert_eq!(rvsdg[loop_node].expect_loop().value_inputs().len(), 4);

        // Verify the base pointer and loop-relative offset value-flow for the first loop-value.

        assert_eq!(
            rvsdg[loop_region].value_results()[1].origin,
            ValueOrigin::Argument(1)
        );

        let ValueOrigin::Output {
            producer: add_node,
            output: 0,
        } = rvsdg[loop_region].value_results()[3].origin
        else {
            panic!("loop-value `0`'s offset result should connect to a node output");
        };

        let add_data = rvsdg[add_node].expect_op_binary();

        assert_eq!(add_data.operator(), BinaryOperator::Add);
        assert_eq!(add_data.lhs_input().origin, ValueOrigin::Argument(3));
        assert_eq!(
            add_data.rhs_input().origin,
            ValueOrigin::Output {
                producer: advance_offset_node,
                output: 0,
            }
        );

        // Verify the base pointer and loop-relative offset value-flow for the second loop-value.

        assert_eq!(
            rvsdg[loop_region].value_results()[2].origin,
            ValueOrigin::Argument(0)
        );
        assert_eq!(
            rvsdg[loop_region].value_results()[4].origin,
            ValueOrigin::Argument(2)
        );
    }

    #[test]
    fn test_flow_through_invariant_inner_loop_value() {
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

        let array_ty = module.ty.register(TypeKind::Array {
            element_ty: TY_U32,
            count: 4,
            stride: 4,
        });
        let array_ptr_ty = module.ty.register(TypeKind::Ptr(array_ty));
        let slice_ty = module.ty.register(TypeKind::Slice {
            element_ty: TY_U32,
            stride: 4,
        });
        let slice_ptr_ty = module.ty.register(TypeKind::Ptr(slice_ty));

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let array_node = rvsdg.add_op_alloca(region, array_ty);
        let offset_node = rvsdg.add_const_u32(region, 0);
        let initial_slice_node = rvsdg.add_op_offset_slice(
            region,
            ValueInput::output(array_ptr_ty, array_node, 0),
            ValueInput::output(TY_U32, offset_node, 0),
        );

        let (outer_loop_node, outer_loop_region) = rvsdg.add_loop(
            region,
            vec![ValueInput::output(slice_ptr_ty, initial_slice_node, 0)],
            None,
        );

        let advance_offset_node = rvsdg.add_const_u32(outer_loop_region, 1);
        let advance_node = rvsdg.add_op_offset_slice(
            outer_loop_region,
            ValueInput::argument(slice_ptr_ty, 0),
            ValueInput::output(TY_U32, advance_offset_node, 0),
        );

        let (inner_loop_node, inner_loop_region) = rvsdg.add_loop(
            outer_loop_region,
            vec![ValueInput::output(slice_ptr_ty, advance_node, 0)],
            None,
        );

        let inner_reentry_predicate_node = rvsdg.add_const_bool(inner_loop_region, false);

        rvsdg.reconnect_region_result(
            inner_loop_region,
            0,
            ValueOrigin::Output {
                producer: inner_reentry_predicate_node,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(inner_loop_region, 1, ValueOrigin::Argument(0));

        let outer_reentry_predicate_node = rvsdg.add_const_bool(outer_loop_region, false);

        rvsdg.reconnect_region_result(
            outer_loop_region,
            0,
            ValueOrigin::Output {
                producer: outer_reentry_predicate_node,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            outer_loop_region,
            1,
            ValueOrigin::Output {
                producer: inner_loop_node,
                output: 0,
            },
        );

        let get_offset_node = rvsdg
            .add_op_get_ptr_offset(region, ValueInput::output(slice_ptr_ty, outer_loop_node, 0));

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: get_offset_node,
                output: 0,
            },
        );

        let did_normalize = normalize_loop_slice_offsets(&mut rvsdg, outer_loop_node, 0).unwrap();

        assert!(did_normalize);

        assert_eq!(rvsdg[outer_loop_node].expect_loop().value_inputs().len(), 2);

        let ValueOrigin::Output {
            producer: re_anchor_node,
            output: 0,
        } = rvsdg[advance_node]
            .expect_op_offset_slice()
            .ptr_input()
            .origin
        else {
            panic!("the advance node's pointer input should connect to a node output");
        };

        let re_anchor_data = rvsdg[re_anchor_node].expect_op_offset_slice();

        assert_eq!(re_anchor_data.ptr_input().origin, ValueOrigin::Argument(0));
        assert_eq!(
            re_anchor_data.offset_input().origin,
            ValueOrigin::Argument(1)
        );

        // Verify the value-flow of the base pointer value

        assert_eq!(
            rvsdg[outer_loop_region].value_results()[1].origin,
            ValueOrigin::Argument(0)
        );

        // Verify the value-flow of the loop-relative offset value

        let ValueOrigin::Output {
            producer: add_node,
            output: 0,
        } = rvsdg[outer_loop_region].value_results()[2].origin
        else {
            panic!("the offset result should connect to a node output");
        };

        let add_data = rvsdg[add_node].expect_op_binary();

        assert_eq!(add_data.operator(), BinaryOperator::Add);
        assert_eq!(add_data.lhs_input().origin, ValueOrigin::Argument(1));
        assert_eq!(
            add_data.rhs_input().origin,
            ValueOrigin::Output {
                producer: advance_offset_node,
                output: 0,
            }
        );

        // Verify that the inner loop node is unchanged

        assert_eq!(rvsdg[inner_loop_node].expect_loop().value_inputs().len(), 1);
        assert_eq!(
            rvsdg[inner_loop_node].expect_loop().value_inputs()[0].origin,
            ValueOrigin::Output {
                producer: advance_node,
                output: 0,
            }
        );
        assert_eq!(
            rvsdg[inner_loop_region].value_results()[1].origin,
            ValueOrigin::Argument(0)
        );
    }

    #[test]
    fn test_value_flow_without_offset_is_noop() {
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

        let array_ty = module.ty.register(TypeKind::Array {
            element_ty: TY_U32,
            count: 4,
            stride: 4,
        });
        let array_ptr_ty = module.ty.register(TypeKind::Ptr(array_ty));
        let slice_ty = module.ty.register(TypeKind::Slice {
            element_ty: TY_U32,
            stride: 4,
        });
        let slice_ptr_ty = module.ty.register(TypeKind::Ptr(slice_ty));

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let array_node = rvsdg.add_op_alloca(region, array_ty);
        let offset_node = rvsdg.add_const_u32(region, 0);
        let initial_slice_node = rvsdg.add_op_offset_slice(
            region,
            ValueInput::output(array_ptr_ty, array_node, 0),
            ValueInput::output(TY_U32, offset_node, 0),
        );

        let (loop_node, loop_region) = rvsdg.add_loop(
            region,
            vec![ValueInput::output(slice_ptr_ty, initial_slice_node, 0)],
            None,
        );

        let case_node = rvsdg.add_const_u32(loop_region, 0);
        let selector_node = rvsdg.add_op_case_to_branch_selector(
            loop_region,
            ValueInput::output(TY_U32, case_node, 0),
            Int::U32,
            [0],
        );
        let switch_node = rvsdg.add_switch(
            loop_region,
            vec![
                ValueInput::output(TY_PREDICATE, selector_node, 0),
                ValueInput::argument(slice_ptr_ty, 0),
            ],
            vec![ValueOutput::new(slice_ptr_ty)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);

        rvsdg.reconnect_region_result(branch_0, 0, ValueOrigin::Argument(0));

        let branch_1 = rvsdg.add_switch_branch(switch_node);

        rvsdg.reconnect_region_result(branch_1, 0, ValueOrigin::Argument(0));

        let reentry_predicate_node = rvsdg.add_const_bool(loop_region, false);

        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: reentry_predicate_node,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            loop_region,
            1,
            ValueOrigin::Output {
                producer: switch_node,
                output: 0,
            },
        );

        let get_offset_node =
            rvsdg.add_op_get_ptr_offset(region, ValueInput::output(slice_ptr_ty, loop_node, 0));

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: get_offset_node,
                output: 0,
            },
        );

        let did_normalize = normalize_loop_slice_offsets(&mut rvsdg, loop_node, 0).unwrap();

        assert!(!did_normalize);

        // The graph should have been left untouched.
        assert_eq!(rvsdg[loop_node].expect_loop().value_inputs().len(), 1);
        assert_eq!(rvsdg[switch_node].value_inputs().len(), 2);
        assert_eq!(rvsdg[switch_node].value_outputs().len(), 1);
        assert_eq!(
            rvsdg[loop_region].value_results()[1].origin,
            ValueOrigin::Output {
                producer: switch_node,
                output: 0,
            }
        );
    }

    #[test]
    fn test_dependency_via_element_ptr() {
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

        let array_ty = module.ty.register(TypeKind::Array {
            element_ty: TY_U32,
            count: 4,
            stride: 4,
        });
        let array_ptr_ty = module.ty.register(TypeKind::Ptr(array_ty));
        let slice_ty = module.ty.register(TypeKind::Slice {
            element_ty: TY_U32,
            stride: 4,
        });
        let slice_ptr_ty = module.ty.register(TypeKind::Ptr(slice_ty));
        let elem_ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let array_node = rvsdg.add_op_alloca(region, array_ty);
        let offset_node = rvsdg.add_const_u32(region, 0);
        let initial_slice_node = rvsdg.add_op_offset_slice(
            region,
            ValueInput::output(array_ptr_ty, array_node, 0),
            ValueInput::output(TY_U32, offset_node, 0),
        );
        let initial_index_node = rvsdg.add_const_u32(region, 0);
        let initial_element_node = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(slice_ptr_ty, initial_slice_node, 0),
            ValueInput::output(TY_U32, initial_index_node, 0),
        );

        let (loop_node, loop_region) = rvsdg.add_loop(
            region,
            vec![
                ValueInput::output(elem_ptr_ty, initial_element_node, 0),
                ValueInput::output(slice_ptr_ty, initial_slice_node, 0),
            ],
            None,
        );

        // Loop-value `1` carries loop-relative offset; loop-value `0` is dependent via an element
        // pointer.
        let advance_offset_node = rvsdg.add_const_u32(loop_region, 1);
        let advance_node = rvsdg.add_op_offset_slice(
            loop_region,
            ValueInput::argument(slice_ptr_ty, 1),
            ValueInput::output(TY_U32, advance_offset_node, 0),
        );

        let index_node = rvsdg.add_const_u32(loop_region, 0);
        let element_node = rvsdg.add_op_element_ptr(
            loop_region,
            ValueInput::argument(slice_ptr_ty, 1),
            ValueInput::output(TY_U32, index_node, 0),
        );

        let reentry_predicate_node = rvsdg.add_const_bool(loop_region, false);

        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: reentry_predicate_node,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            loop_region,
            1,
            ValueOrigin::Output {
                producer: element_node,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            loop_region,
            2,
            ValueOrigin::Output {
                producer: advance_node,
                output: 0,
            },
        );

        let get_offset_node =
            rvsdg.add_op_get_ptr_offset(region, ValueInput::output(slice_ptr_ty, loop_node, 1));

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: get_offset_node,
                output: 0,
            },
        );

        // Request normalization for loop-value `0`. It is not itself of a slice-pointer type and
        // carries no offset from its own single-iteration viewpoint. However, it receives offset
        // from previous iterations of loop-value `1`.
        let did_normalize = normalize_loop_slice_offsets(&mut rvsdg, loop_node, 0).unwrap();

        assert!(did_normalize);

        // Only loop-value `1` should have been rewritten; its offset loop-value is loop-value `2`.
        assert_eq!(rvsdg[loop_node].expect_loop().value_inputs().len(), 3);
        assert_eq!(rvsdg[loop_node].expect_loop().value_inputs()[2].ty, TY_U32);

        // Both the element-pointer node and the advance node should now observe loop-value `1`'s
        // argument through a new OpOffsetSlice new.
        let ValueOrigin::Output {
            producer: re_anchor_node,
            output: 0,
        } = rvsdg[element_node]
            .expect_op_element_ptr()
            .ptr_input()
            .origin
        else {
            panic!("the element-pointer node's pointer input should connect to a node output");
        };

        let re_anchor_data = rvsdg[re_anchor_node].expect_op_offset_slice();

        assert_eq!(re_anchor_data.ptr_input().origin, ValueOrigin::Argument(1));
        assert_eq!(
            re_anchor_data.offset_input().origin,
            ValueOrigin::Argument(2)
        );
        assert_eq!(
            rvsdg[advance_node]
                .expect_op_offset_slice()
                .ptr_input()
                .origin,
            ValueOrigin::Output {
                producer: re_anchor_node,
                output: 0,
            }
        );

        // Loop-value `0`'s value-flow should have been left untouched.
        assert_eq!(
            rvsdg[loop_region].value_results()[1].origin,
            ValueOrigin::Output {
                producer: element_node,
                output: 0,
            }
        );

        // Verify the value-flow of loop-value `1`'s base pointer value

        assert_eq!(
            rvsdg[loop_region].value_results()[2].origin,
            ValueOrigin::Argument(1)
        );

        // Verify the value-flow of loop-value `1`'s loop-relative offset value

        let ValueOrigin::Output {
            producer: add_node,
            output: 0,
        } = rvsdg[loop_region].value_results()[3].origin
        else {
            panic!("loop-value `1`'s offset result should connect to a node output");
        };

        let add_data = rvsdg[add_node].expect_op_binary();

        assert_eq!(add_data.operator(), BinaryOperator::Add);
        assert_eq!(add_data.lhs_input().origin, ValueOrigin::Argument(2));
        assert_eq!(
            add_data.rhs_input().origin,
            ValueOrigin::Output {
                producer: advance_offset_node,
                output: 0,
            }
        );
    }
}
