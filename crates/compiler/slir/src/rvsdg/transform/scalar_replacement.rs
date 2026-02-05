use std::collections::VecDeque;
use std::ops::Deref;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::rvsdg::analyse::element_index::ElementIndex;
use crate::rvsdg::transform::enum_replacement::replace_enum_alloca;
use crate::rvsdg::visit::value_flow::ValueFlowVisitor;
use crate::rvsdg::{
    Connectivity, Node, NodeKind, OpAlloca, OpLoad, ProxyKind, Region, Rvsdg, SimpleNode,
    StateOrigin, ValueInput, ValueOrigin, ValueOutput, ValueUser, visit,
};
use crate::ty::{TY_PREDICATE, TY_U32, Type, TypeKind, TypeRegistry};
use crate::{Function, Module};

enum Job {
    Alloca(Node),
    SwitchOutput(Node),
}

/// Collects all [OpAlloca] nodes of aggregate types in a region and all sub-regions (e.g. a switch
/// node branch region) into a queue of candidates for scalar replacement.
///
/// Note that this does not yet make any decisions about whether we should perform a scalar
/// replacement transform on a given [OpAlloca] node, this requires further analysis.
struct CandidateAllocaCollector<'a, 'b> {
    rvsdg: &'a Rvsdg,
    candidates: &'b mut VecDeque<Job>,
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
                if self.rvsdg.ty().kind(op.ty()).is_aggregate() {
                    self.candidates.push_back(Job::Alloca(node));
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

fn collect_candidate_allocas(rvsdg: &Rvsdg, candidates: &mut VecDeque<Job>, region: Region) {
    CandidateAllocaCollector { rvsdg, candidates }.visit_region(region);
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AnalysisResult {
    Replace,
    NeedsPromotionPass,
    NeedsSwitchOutputReplacement,
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
/// Passing or returning sub-elements of the aggregate, obtained via e.g. an [OpPtrElementPtr], does
/// not constitute an escape, as in these cases scalar replacement will only require local
/// modifications (the [OpPtrElementPtr] can be adjusted such that any [OpCall] user or result user
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
    needs_switch_output_replacement: bool,
}

impl AggregateAllocaAnalyzer {
    fn new() -> Self {
        Self {
            visited: FxHashSet::default(),
            was_loaded: false,
            has_nonlocal_use: false,
            is_stored_value: false,
            needs_switch_output_replacement: false,
        }
    }

    fn analyze_alloca_node(&mut self, rvsdg: &Rvsdg, alloca_node: Node) -> AnalysisResult {
        // Reset
        self.visited.clear();
        self.has_nonlocal_use = false;
        self.is_stored_value = false;
        self.was_loaded = false;
        self.needs_switch_output_replacement = false;

        // Perform the analysis
        self.visit_value_output(rvsdg, alloca_node, 0);

        // Summarize the analysis result
        if self.has_nonlocal_use {
            AnalysisResult::Ignore
        } else if self.is_stored_value {
            AnalysisResult::NeedsPromotionPass
        } else if self.needs_switch_output_replacement {
            AnalysisResult::NeedsSwitchOutputReplacement
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
            Simple(Reaggregation(_)) => {
                self.needs_switch_output_replacement = true;
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

struct SplitBranchState {
    branch_count: u32,
    split_branch_count: u32,
}

struct Replacer<'a, 'b, 'c> {
    rvsdg: &'a mut Rvsdg,
    candidate_queue: &'b mut VecDeque<Job>,
    switch_output_state: &'c mut FxHashMap<Node, SplitBranchState>,
    ty: TypeRegistry,
}

impl Replacer<'_, '_, '_> {
    fn perform_job(&mut self, job: Job) {
        match job {
            Job::Alloca(node) => self.replace_alloca(node),
            Job::SwitchOutput(node) => self.replace_switch_output(node),
        }
    }

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
                        self.candidate_queue.push_back(Job::Alloca(node));
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
                        self.candidate_queue.push_back(Job::Alloca(element_node));
                    }
                }
            }
            TypeKind::Struct(struct_data) => {
                for field in &struct_data.fields {
                    let field_ty = field.ty;
                    let field_ptr_ty = self.ty.register(TypeKind::Ptr(field_ty));
                    let element_node = self.rvsdg.add_op_alloca(region, field_ty);

                    replacements.push(ValueInput {
                        ty: field_ptr_ty,
                        origin: ValueOrigin::Output {
                            producer: element_node,
                            output: 0,
                        },
                    });

                    if self.ty.kind(field_ty).is_aggregate() {
                        self.candidate_queue.push_back(Job::Alloca(element_node));
                    }
                }
            }
            _ => unreachable!("type is not an aggregate, node should not have been a candidate"),
        }

        self.visit_users(node, 0, &replacements);

        let _ = self.rvsdg.try_remove_node(node);
    }

    fn replace_switch_output(&mut self, proxy_node: Node) {
        let ValueOrigin::Output {
            producer: switch_node,
            output,
        } = self.rvsdg[proxy_node].expect_value_proxy().input().origin
        else {
            panic!("the value-proxy marker node should connect to a switch output")
        };

        let data = self.rvsdg[switch_node].expect_switch();
        let prior_result_count = data.value_outputs().len();
        let branch_count = data.branches().len();

        assert!(branch_count > 0);

        let first_branch = data.branches()[0];

        let mut max_part_count = 0;

        for branch in data.branches() {
            let ValueOrigin::Output {
                producer: reaggregation_node,
                output: 0,
            } = self.rvsdg[*branch].value_results()[output as usize].origin
            else {
                panic!("all branches should have been processed into a reaggregation node earlier")
            };
            let part_count = self.rvsdg[reaggregation_node]
                .expect_reaggregation()
                .parts()
                .len();

            max_part_count = usize::max(max_part_count, part_count);
        }

        let mut split_output = Vec::with_capacity(max_part_count);

        // Add the new outputs for the replacement values and record them in `split_output`. We'll
        // derive the output value types from the outputs of the reaggregation node in the first
        // branch.
        for i in 0..max_part_count {
            let ValueOrigin::Output {
                producer: reaggregation_node,
                output: 0,
            } = self.rvsdg[first_branch].value_results()[output as usize].origin
            else {
                panic!("all branches should have been processed into a reaggregation node earlier")
            };
            let inputs = self.rvsdg[reaggregation_node]
                .expect_reaggregation()
                .parts();
            let input_count = inputs.len();
            // If the part count for the first branch is not equal to the `max_part_count`, then
            // we may assume we're dealing with a slice. We'll simply repeat the final input to
            // derive the extra outputs, as every part of a slice will have the same type.
            let input_index = usize::max(i, input_count - 1);
            let input = inputs[input_index];

            split_output.push(ValueInput {
                ty: input.ty,
                origin: ValueOrigin::Output {
                    producer: switch_node,
                    output: i as u32,
                },
            });
        }

        // Now that we've created the new outputs/results, connect them to the inputs of the
        // reaggregation nodes in each branch.
        for b in 0..branch_count {
            let branch = self.rvsdg[switch_node].expect_switch().branches()[b];
            let ValueOrigin::Output {
                producer: reaggregation_node,
                output: 0,
            } = self.rvsdg[branch].value_results()[output as usize].origin
            else {
                panic!("all branches should have been processed into a reaggregation node earlier")
            };

            for i in 0..max_part_count {
                let inputs = self.rvsdg[reaggregation_node]
                    .expect_reaggregation()
                    .parts();
                let input_count = inputs.len();
                // If the part count for the branch is not equal to the `max_part_count`, then
                // we may assume we're dealing with a slice. Since accessing those values would be
                // UB (and should never happen), we're allowed to connect any correctly typed value
                // to produce a valid RVSDG. We'll opt to simply repeat the final value.
                let input_index = usize::max(i, input_count - 1);
                let input = inputs[input_index];
                let result_index = prior_result_count + i;

                self.rvsdg
                    .reconnect_region_result(branch, result_index as u32, input.origin);
            }
            let original_origin = self.rvsdg[reaggregation_node]
                .expect_reaggregation()
                .original()
                .origin;

            self.rvsdg
                .reconnect_region_result(branch, output, original_origin);
            self.rvsdg.remove_node(reaggregation_node);
        }

        self.rvsdg.dissolve_value_proxy(proxy_node);
        self.visit_users(switch_node, output, &split_output);
    }

    fn visit_users(&mut self, node: Node, output: u32, split_inputs: &[ValueInput]) {
        let region = self.rvsdg[node].region();
        let user_count = self.rvsdg[node].value_outputs()[output as usize]
            .users
            .len();

        // We iterate over users in reverse order, so that users may more themselves from the user
        // set, without disrupting iteration
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
                ProxyKind::Generic,
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
                ProxyKind::Generic,
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
        let outer_region = self.rvsdg[node].region();
        let node_data = self.rvsdg[node].expect_switch();
        let branch_count = node_data.branches().len() as u32;
        let original = self.rvsdg[branch].value_results()[result as usize];

        self.rvsdg
            .add_reaggregation(branch, original, split_input.iter().copied());

        let output_users = &self.rvsdg[node].value_outputs()[result as usize].users;

        // We proxy the switch output that is to be split with a proxy node, then use that proxy
        // node to keep track of the output that is to be split later. Doing it this way, rather
        // than storing the output index, allows us to remove switch outputs in between now and
        // when the actual output splitting happens; if we tried to keep track of the output by its
        // index, then the removal of other switch outputs might make our tracking data invalid.
        let output_proxy = if output_users.len() == 1
            && let ValueUser::Input { consumer, input: 0 } = output_users[0]
            && self.rvsdg[consumer].is_switch_output_replacement_marker()
        {
            // We already seem to have added the proxy when we visited a previous branch: use that
            // proxy node.
            consumer
        } else {
            self.rvsdg.proxy_origin_users(
                outer_region,
                original.ty,
                ValueOrigin::Output {
                    producer: node,
                    output: result,
                },
                ProxyKind::SwitchOutputReplacementMarker,
            )
        };

        if let Some(state) = self.switch_output_state.get_mut(&output_proxy) {
            state.split_branch_count += 1;
        } else {
            self.switch_output_state.insert(
                output_proxy,
                SplitBranchState {
                    branch_count,
                    split_branch_count: 1,
                },
            );
            self.candidate_queue
                .push_back(Job::SwitchOutput(output_proxy))
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
        // original result connects to, via [OpPtrElementPtr] nodes or [OpExtractElement] nodes,
        // depending on whether the result type is a pointer or an immediate value. This also
        // disconnects the original result.
        self.redirect_region_result(loop_region, input as usize + 1, prior_result_count);

        // Now redirect the argument using the argument mapping. We do this after redirecting the
        // results, because otherwise this might try to split the original result again; by doing
        // this after result redirection, all the argument's user tree should terminate at the
        // [OpPtrElementPtr]/[OpExtractElement] nodes that were inserted, since nothing should
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
            ProxyKind::Generic,
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

        // Add inputs for each element of the aggregate, and connect them to the proxy via either
        // an OpPtrElementPtr node if the input is of a pointer type, or via an OpExtractElement
        // node otherwise. Also record an argument mapping in `split_args` and an output mapping in
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
    /// start at `split_results_start`, via either [OpPtrElementPtr] or [OpExtractElement] nodes
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
    queue: VecDeque<Job>,
    candidates: VecDeque<Job>,

    /// Tracks for a switch node result/output pair how many branches have been processed into a
    /// [Reaggregation] node.
    ///
    /// The [SplitBranchState] value also stores the total number of branches for the switch node,
    /// so that we can cheaply check if all branches have been processed without having to look up
    /// the actual switch node in the RVSDG.
    ///
    /// The [Node] we use as a key is not the [Switch] node itself. It is instead a [ValueProxy]
    /// node that has been attached to the switch output that we intend to split. The [ValueOrigin]
    /// of that [ValueProxy] node's input identifies the actual [Switch] node and the output of
    /// interest. We do this so that way may remove switch outputs without invalidating this
    /// tracking information; if we instead tracked the output as e.g. a
    /// `(switch_node, output_index)` pair, then removing any of the switch node's outputs might
    /// invalidate our mapping.
    switch_output_state: FxHashMap<Node, SplitBranchState>,
}

impl AggregateReplacementContext {
    pub fn new() -> Self {
        AggregateReplacementContext {
            analyzer: AggregateAllocaAnalyzer::new(),
            queue: VecDeque::new(),
            candidates: VecDeque::new(),
            switch_output_state: Default::default(),
        }
    }

    pub fn for_region(&mut self, rvsdg: &Rvsdg, region: Region) -> RegionReplacementContext {
        self.queue.clear();
        self.candidates.clear();
        self.switch_output_state.clear();

        collect_candidate_allocas(rvsdg, &mut self.candidates, region);

        RegionReplacementContext { cx: self }
    }

    fn try_enqueue_candidates(&mut self, rvsdg: &Rvsdg, start: usize) {
        let mut i = start;

        while i < self.candidates.len() {
            match self.candidates[i] {
                Job::Alloca(node) => match self.analyzer.analyze_alloca_node(rvsdg, node) {
                    AnalysisResult::Replace => {
                        self.queue.push_back(Job::Alloca(node));
                        self.candidates.swap_remove_back(i);
                    }
                    AnalysisResult::NeedsPromotionPass => i += 1,
                    AnalysisResult::NeedsSwitchOutputReplacement => i += 1,
                    AnalysisResult::Ignore => {
                        self.candidates.swap_remove_back(i);
                    }
                },
                Job::SwitchOutput(marker) => {
                    let state = self.switch_output_state.get(&marker).expect(
                        "job should not have been in candidate queue without a \
                        corresponding state entry",
                    );

                    if state.split_branch_count == state.branch_count {
                        self.queue.push_back(Job::SwitchOutput(marker));
                        self.candidates.swap_remove_back(i);
                    } else {
                        i += 1;
                    }
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

            while let Some(job) = self.cx.queue.pop_front() {
                let prior_candidate_count = self.cx.candidates.len();

                let mut replacer = Replacer {
                    rvsdg,
                    candidate_queue: &mut self.cx.candidates,
                    switch_output_state: &mut self.cx.switch_output_state,
                    ty: ty_registry.clone(),
                };

                replacer.perform_job(job);

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
    use crate::ty::{TY_DUMMY, TY_PREDICATE};
    use crate::{BinaryOperator, FnArg, FnSig, Symbol, thin_set};

    #[test]
    fn test_scalar_replace_op_ptr_element_ptr() {
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
            element_ty: TY_U32,
            count: 2,
            stride: 4,
        });
        let ptr_ty = module.ty.register(TypeKind::Ptr(ty));
        let element_ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let op_alloca = rvsdg.add_op_alloca(region, ty);
        let element_index = rvsdg.add_const_u32(region, 1);
        let op_ptr_element_ptr = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(ptr_ty, op_alloca, 0),
            ValueInput::output(TY_U32, element_index, 0),
        );
        let load = rvsdg.add_op_load(
            region,
            ValueInput::output(element_ptr_ty, op_ptr_element_ptr, 0),
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

        assert_eq!(element_alloca.ty(), TY_U32);
        assert_eq!(
            &element_alloca.value_output().users,
            &thin_set![ValueUser::Input {
                consumer: load,
                input: 0
            }]
        );

        assert!(!rvsdg.is_live_node(op_ptr_element_ptr));
        assert!(!rvsdg.is_live_node(op_alloca));
    }

    #[test]
    fn test_scalar_replace_op_ptr_element_ptr_dynamic_index() {
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
            element_ty: TY_U32,
            count: 2,
            stride: 4,
        });
        let ptr_ty = module.ty.register(TypeKind::Ptr(ty));
        let element_ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let op_alloca = rvsdg.add_op_alloca(region, ty);
        let op_ptr_element_ptr = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(ptr_ty, op_alloca, 0),
            ValueInput::argument(TY_U32, 0),
        );
        let load = rvsdg.add_op_load(
            region,
            ValueInput::output(element_ptr_ty, op_ptr_element_ptr, 0),
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

        assert_eq!(element_0_alloca.ty(), TY_U32);
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

        assert_eq!(element_1_alloca.ty(), TY_U32);
        assert_eq!(
            &element_1_alloca.value_output().users,
            &thin_set![ValueUser::Input {
                consumer: switch_node,
                input: 2
            }]
        );

        assert!(!rvsdg.is_live_node(op_ptr_element_ptr));
        assert!(!rvsdg.is_live_node(op_alloca));
    }

    #[test]
    fn test_scalar_replace_op_extract_element() {
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
            element_ty: TY_U32,
            count: 2,
            stride: 4,
        });
        let ptr_ty = module.ty.register(TypeKind::Ptr(ty));
        let element_ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let op_alloca = rvsdg.add_op_alloca(region, ty);
        let op_load = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, op_alloca, 0),
            StateOrigin::Argument,
        );
        let element_index = rvsdg.add_const_u32(region, 1);
        let op_extract_element = rvsdg.add_op_extract_element(
            region,
            ValueInput::output(ty, op_load, 0),
            ValueInput::output(TY_U32, element_index, 0),
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: op_extract_element,
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

        let element_load = rvsdg[element_load_node].expect_op_load();

        assert_eq!(element_load.ptr_input().ty, element_ptr_ty);
        assert_eq!(element_load.value_output().ty, TY_U32);
        assert_eq!(
            &element_load.value_output().users,
            &thin_set![ValueUser::Result(0)]
        );

        let element_load_input = element_load.ptr_input();

        let ValueOrigin::Output {
            producer: element_alloca_node,
            output: element_alloca_output,
        } = element_load_input.origin
        else {
            panic!("element load op origin should be the first output of a node")
        };

        assert_eq!(element_alloca_output, 0);

        let element_alloca = rvsdg[element_alloca_node].expect_op_alloca();

        assert_eq!(element_alloca.ty(), TY_U32);
        assert_eq!(element_alloca.value_output().ty, element_ptr_ty);
        assert_eq!(
            &element_alloca.value_output().users,
            &thin_set![ValueUser::Input {
                consumer: element_load_node,
                input: 0
            }]
        );

        assert!(!rvsdg.is_live_node(op_alloca));
        assert!(!rvsdg.is_live_node(op_load));
        assert!(!rvsdg.is_live_node(op_extract_element));
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

        let ty = module.ty.register(TypeKind::Array {
            element_ty: TY_U32,
            count: 2,
            stride: 4,
        });
        let ptr_ty = module.ty.register(TypeKind::Ptr(ty));
        let element_ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

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

        let StateOrigin::Node(store_element_1_node) = *rvsdg[region].state_result() else {
            panic!("the state origin should be a node output")
        };

        let store_element_1 = rvsdg[store_element_1_node].expect_op_store();

        let StateOrigin::Node(store_element_0_node) = store_element_1.state().unwrap().origin
        else {
            panic!("the state origin should be a node output")
        };

        let store_element_0 = rvsdg[store_element_0_node].expect_op_store();

        let StateOrigin::Node(load_element_1_node) = store_element_0.state().unwrap().origin else {
            panic!("the state origin should be a node output")
        };

        let load_element_1 = rvsdg[load_element_1_node].expect_op_load();

        let StateOrigin::Node(load_element_0_node) = load_element_1.state().unwrap().origin else {
            panic!("the state origin should be a node output")
        };

        let load_element_0 = rvsdg[load_element_0_node].expect_op_load();

        let ValueOrigin::Output {
            producer: alloca_element_0_node,
            output: 0,
        } = store_element_0.ptr_input().origin
        else {
            panic!("the pointer input to store_element_0 node should be the first output of a node")
        };

        let alloca_element_0 = rvsdg[alloca_element_0_node].expect_op_alloca();

        let ValueOrigin::Output {
            producer: alloca_element_1_node,
            output: 0,
        } = store_element_1.ptr_input().origin
        else {
            panic!("the pointer input to store_element_1 node should be the first output of a node")
        };

        let alloca_element_1 = rvsdg[alloca_element_1_node].expect_op_alloca();

        assert_eq!(alloca_element_0.ty(), TY_U32);
        assert_eq!(
            &alloca_element_0.value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: store_element_0_node,
                    input: 0
                },
                ValueUser::Input {
                    consumer: load_element_0_node,
                    input: 0
                }
            ]
        );

        assert_eq!(alloca_element_1.ty(), TY_U32);
        assert_eq!(
            &alloca_element_1.value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: store_element_1_node,
                    input: 0
                },
                ValueUser::Input {
                    consumer: load_element_1_node,
                    input: 0
                }
            ]
        );

        assert_eq!(
            load_element_0.ptr_input(),
            &ValueInput {
                ty: element_ptr_ty,
                origin: ValueOrigin::Output {
                    producer: alloca_element_0_node,
                    output: 0,
                },
            }
        );
        assert_eq!(
            &load_element_0.value_output().users,
            &thin_set![ValueUser::Input {
                consumer: store_element_0_node,
                input: 1
            }]
        );

        assert_eq!(
            load_element_1.ptr_input(),
            &ValueInput {
                ty: element_ptr_ty,
                origin: ValueOrigin::Output {
                    producer: alloca_element_1_node,
                    output: 0,
                },
            }
        );
        assert_eq!(
            &load_element_1.value_output().users,
            &thin_set![ValueUser::Input {
                consumer: store_element_1_node,
                input: 1
            }]
        );

        assert_eq!(
            store_element_0.ptr_input(),
            &ValueInput {
                ty: element_ptr_ty,
                origin: ValueOrigin::Output {
                    producer: alloca_element_0_node,
                    output: 0,
                },
            }
        );
        assert_eq!(
            store_element_0.value_input(),
            &ValueInput {
                ty: TY_U32,
                origin: ValueOrigin::Output {
                    producer: load_element_0_node,
                    output: 0,
                },
            }
        );

        assert_eq!(
            store_element_1.ptr_input(),
            &ValueInput {
                ty: element_ptr_ty,
                origin: ValueOrigin::Output {
                    producer: alloca_element_1_node,
                    output: 0,
                },
            }
        );
        assert_eq!(
            store_element_1.value_input(),
            &ValueInput {
                ty: TY_U32,
                origin: ValueOrigin::Output {
                    producer: load_element_1_node,
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

        let ty = module.ty.register(TypeKind::Array {
            element_ty: TY_U32,
            count: 2,
            stride: 4,
        });
        let ptr_ty = module.ty.register(TypeKind::Ptr(ty));
        let element_ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

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
        let branch_0_index = rvsdg.add_const_u32(branch_0, 0);
        let branch_0_op_ptr_element_ptr = rvsdg.add_op_element_ptr(
            branch_0,
            ValueInput::argument(ptr_ty, 0),
            ValueInput::output(TY_U32, branch_0_index, 0),
        );
        let branch_0_load = rvsdg.add_op_load(
            branch_0,
            ValueInput::output(element_ptr_ty, branch_0_op_ptr_element_ptr, 0),
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
        let branch_1_index = rvsdg.add_const_u32(branch_1, 1);
        let branch_1_op_ptr_element_ptr = rvsdg.add_op_element_ptr(
            branch_1,
            ValueInput::argument(ptr_ty, 0),
            ValueInput::output(TY_U32, branch_1_index, 0),
        );
        let branch_1_load = rvsdg.add_op_load(
            branch_1,
            ValueInput::output(element_ptr_ty, branch_1_op_ptr_element_ptr, 0),
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
        assert_eq!(branch_0.value_arguments()[1].ty, element_ptr_ty);
        assert_eq!(branch_0.value_arguments()[2].ty, element_ptr_ty);
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
        assert_eq!(branch_1.value_arguments()[1].ty, element_ptr_ty);
        assert_eq!(branch_1.value_arguments()[2].ty, element_ptr_ty);
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
            producer: element_0_alloca_node,
            output: 0,
        } = switch.value_inputs()[2].origin
        else {
            panic!("the second input to the switch node should be the first output of a node")
        };

        let element_0_alloca = rvsdg[element_0_alloca_node].expect_op_alloca();

        assert_eq!(element_0_alloca.ty(), TY_U32);
        assert_eq!(
            &element_0_alloca.value_output().users,
            &thin_set![ValueUser::Input {
                consumer: switch_node,
                input: 2,
            }]
        );

        let ValueOrigin::Output {
            producer: element_1_alloca_node,
            output: 0,
        } = switch.value_inputs()[3].origin
        else {
            panic!("the third input to the switch node should be the first output of a node")
        };

        let element_1_alloca = rvsdg[element_1_alloca_node].expect_op_alloca();

        assert_eq!(element_1_alloca.ty(), TY_U32);
        assert_eq!(
            &element_1_alloca.value_output().users,
            &thin_set![ValueUser::Input {
                consumer: switch_node,
                input: 3,
            }]
        );

        assert!(!rvsdg.is_live_node(branch_0_op_ptr_element_ptr));
        assert!(!rvsdg.is_live_node(branch_1_op_ptr_element_ptr));
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

        let ty = module.ty.register(TypeKind::Array {
            element_ty: TY_U32,
            count: 2,
            stride: 4,
        });
        let ptr_ty = module.ty.register(TypeKind::Ptr(ty));
        let element_ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

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

        let element_index = rvsdg.add_const_u32(loop_region, 0);
        let op_ptr_element_ptr_node = rvsdg.add_op_element_ptr(
            loop_region,
            ValueInput::argument(ptr_ty, 2),
            ValueInput::output(TY_U32, element_index, 0),
        );
        let load_node = rvsdg.add_op_load(
            loop_region,
            ValueInput::output(element_ptr_ty, op_ptr_element_ptr_node, 0),
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
            ValueInput::output(element_ptr_ty, op_ptr_element_ptr_node, 0),
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
        assert_eq!(loop_data.value_inputs()[3].ty, element_ptr_ty);
        assert_eq!(loop_data.value_inputs()[4].ty, element_ptr_ty);

        let ValueOrigin::Output {
            producer: element_0_alloca_node,
            output: 0,
        } = loop_data.value_inputs()[3].origin
        else {
            panic!("the third input to the switch node should be the first output of a node")
        };

        let element_0_alloca = rvsdg[element_0_alloca_node].expect_op_alloca();

        assert_eq!(element_0_alloca.ty(), TY_U32);
        assert_eq!(
            &element_0_alloca.value_output().users,
            &thin_set![ValueUser::Input {
                consumer: loop_node,
                input: 3,
            }]
        );

        let ValueOrigin::Output {
            producer: element_1_alloca_node,
            output: 0,
        } = loop_data.value_inputs()[4].origin
        else {
            panic!("the third input to the switch node should be the first output of a node")
        };

        let element_1_alloca = rvsdg[element_1_alloca_node].expect_op_alloca();

        assert_eq!(element_1_alloca.ty(), TY_U32);
        assert_eq!(
            &element_1_alloca.value_output().users,
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

        assert_eq!(arguments[3].ty, element_ptr_ty);
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

        assert_eq!(arguments[4].ty, element_ptr_ty);
        assert_eq!(arguments[4].users, thin_set![ValueUser::Result(5)]);

        let results = rvsdg[loop_region].value_results();

        assert_eq!(results.len(), 6);

        assert_eq!(results[4].ty, element_ptr_ty);
        assert_eq!(results[4].origin, ValueOrigin::Argument(3));

        assert_eq!(results[5].ty, element_ptr_ty);
        assert_eq!(results[5].origin, ValueOrigin::Argument(4));

        assert!(!rvsdg.is_live_node(op_ptr_element_ptr_node));
    }
}
