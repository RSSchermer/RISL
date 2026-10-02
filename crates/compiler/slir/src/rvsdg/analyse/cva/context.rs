use indexmap::{IndexMap, IndexSet};

use super::domain::AbstractPredicate;
use super::engine::Engine;
use super::routing::contains_region;
use super::{AbstractValue, ContextId, QueryCache, ValueKey, branches_matching_selector};
use crate::rvsdg::analyse::scalar_constant::ScalarConstant;
use crate::rvsdg::{Node, NodeKind, Region, Rvsdg, SimpleNode, ValueOrigin};
use crate::ty::{TY_I32, TY_U32};

pub const MAX_SATURATION_ROUNDS: usize = 8;

#[derive(Default, PartialEq, Eq, Debug)]
pub struct ConstraintFrame {
    /// Additional constraints on values that are a consequence of the assumptions made by this
    /// frame's control-flow context.
    ///
    /// Each entry associates a value in the RVSDG with the abstract set of values it is constrained
    /// to. When a value is queried, the constraints from this frame are combined with the
    /// constraints recorded for the value by this frame's ancestors. This map therefore contains
    /// only the information contributed by this frame, rather than a complete copy of all
    /// constraints that apply in the current context.
    ///
    /// If a value does not have an entry, then this frame places no additional constraints on that
    /// value.
    pub values: IndexMap<ValueKey, AbstractValue>,
}

pub struct Context {
    /// The function body or innermost loop body whose execution is being queried.
    pub scope: Region,

    /// This context's parent region, if the context is not the function's root body region.
    pub parent: Option<ContextId>,

    /// If this context was created as a child context via
    /// [super::ContextualValueAnalysis::assume_branch], then this records the
    /// `(switch_node, branch)` pair that was assumed.
    pub branch_assumption: Option<(Node, usize)>,

    pub constraint_frame: ConstraintFrame,
}

/// Short-lived summary of a context stack.
///
/// Context stacks are implemented as linked lists, where each context identifies its parent
/// context. Queries would have to repeatedly search this linked list to accumulate the information
/// they require, which is inefficient. Instead, we accumulate the information once at the start of
/// the query in this "environment" and then search the environment instead of the context stack
/// itself.
#[derive(Clone)]
pub struct Environment<'a> {
    pub scope: Region,
    pub branch_assumptions: Vec<(Node, usize)>,
    pub frames: Vec<&'a ConstraintFrame>,
}

impl<'a> Environment<'a> {
    pub fn init(contexts: &'a [Context], id: ContextId) -> Self {
        let scope = contexts[id.index].scope;

        let mut branch_assumptions = Vec::new();
        let mut frames = Vec::new();
        let mut current = Some(id);

        while let Some(id) = current {
            let context = &contexts[id.index];

            if let Some(branch) = context.branch_assumption {
                branch_assumptions.push(branch);
            }

            frames.push(&context.constraint_frame);
            current = context.parent;
        }

        branch_assumptions.reverse();
        frames.reverse();

        Self {
            scope,
            branch_assumptions,
            frames,
        }
    }

    pub fn with_constraint_frame<'b>(
        &'b self,
        constraint_frame: &'b ConstraintFrame,
    ) -> Environment<'b> {
        let mut frames = self.frames.clone();

        frames.push(constraint_frame);

        Environment {
            scope: self.scope,
            branch_assumptions: self.branch_assumptions.clone(),
            frames,
        }
    }

    pub fn with_branch_assumption(&self, node: Node, branch: usize) -> Self {
        let mut result = self.clone();

        result.branch_assumptions.push((node, branch));

        result
    }

    /// Extends the environment with both an explicit branch assumption and its derived constraints.
    pub fn with_assumed_branch<'b>(
        &'b self,
        node: Node,
        branch: usize,
        constraint_frame: &'b ConstraintFrame,
    ) -> Environment<'b> {
        let mut branch_assumptions = self.branch_assumptions.clone();
        let mut frames = self.frames.clone();

        branch_assumptions.push((node, branch));
        frames.push(constraint_frame);

        Environment {
            scope: self.scope,
            branch_assumptions,
            frames,
        }
    }

    /// If, within this context environment, a branch was assumed for the `switch` node, returns
    /// the branch index.
    ///
    /// Otherwise, returns `None` indicating that no branch was assumed.
    pub fn assumed_branch(&self, switch: Node) -> Option<usize> {
        self.branch_assumptions
            .iter()
            .find_map(|&(node, branch)| (node == switch).then_some(branch))
    }

    pub fn recorded_switch_output_keys(&self, switch: Node) -> IndexSet<ValueKey> {
        let mut keys = IndexSet::new();

        for frame in &self.frames {
            for &key in frame.values.keys() {
                if matches!(key.origin, ValueOrigin::Output { producer, .. } if producer == switch)
                {
                    keys.insert(key);
                }
            }
        }

        keys
    }

    /// Accumulates constraints derived from this context's assumptions during construction,
    /// without performing any new inference.
    pub fn assumption_derived_constraints(&self, graph: &Rvsdg, key: ValueKey) -> AbstractValue {
        let mut result = AbstractValue::top(key.ty(graph));

        for frame in &self.frames {
            if let Some(bound) = frame.values.get(&key) {
                result = result.refine(bound);
            }
        }

        result
    }

    /// Returns whether `region` is inside the current control-flow context.
    ///
    /// If the context assumes certain branch selections, then the region is only "reachable" if it
    /// is inside the assumed regions.
    pub fn is_in_context(&self, graph: &Rvsdg, mut region: Region) -> bool {
        loop {
            if region == graph.global_region() {
                return false;
            }

            let owner = graph[region].owner();

            match graph[owner].kind() {
                NodeKind::Function(_) => return contains_region(graph, region, self.scope),
                NodeKind::Loop(_) if !contains_region(graph, region, self.scope) => return false,
                NodeKind::Switch(switch) => {
                    if self
                        .branch_assumptions
                        .iter()
                        .any(|&(node, branch)| node == owner && switch.branches()[branch] != region)
                    {
                        return false;
                    }
                }
                _ => {}
            }

            region = graph[owner].region();
        }
    }
}

/// Builds a constraint-frame for the selected branch or returns `None` if the assumption leads to
/// a contradiction.
pub fn constraint_frame_for_assumed_branch(
    engine: &mut Engine<'_>,
    parent: &Environment,
    switch: Node,
    branch: usize,
) -> Option<ConstraintFrame> {
    if parent
        .assumed_branch(switch)
        .is_some_and(|assumed| assumed != branch)
    {
        return None;
    }

    let assumed_parent = parent.with_branch_assumption(switch, branch);
    let builder = ContextBuilder::for_assumed_branch(engine, &assumed_parent, switch, branch);

    (!builder.contradictory).then_some(builder.constraint_frame)
}

struct ContextBuilder<'a, 'graph, 'frames> {
    engine: &'a mut Engine<'graph>,
    parent: &'a Environment<'frames>,
    constraint_frame: ConstraintFrame,
    contradictory: bool,
    cache: QueryCache,
}

impl<'a, 'graph, 'frames> ContextBuilder<'a, 'graph, 'frames> {
    fn for_assumed_branch(
        engine: &'a mut Engine<'graph>,
        parent: &'a Environment<'frames>,
        switch: Node,
        branch: usize,
    ) -> Self {
        let mut builder = Self {
            engine,
            parent,
            constraint_frame: ConstraintFrame::default(),
            contradictory: false,
            cache: QueryCache::default(),
        };

        let selector = ValueKey::for_node_input(builder.engine.rvsdg, switch, 0);

        builder.record_value(
            selector,
            AbstractValue::from_scalar_constant(ScalarConstant::Predicate(branch as u32)),
        );
        builder.saturate();

        builder
    }

    fn record_value(&mut self, key: ValueKey, constraints: AbstractValue) -> bool {
        let Some(key) = key.canonicalize(self.engine.rvsdg) else {
            return false;
        };

        let env = self.parent.with_constraint_frame(&self.constraint_frame);
        let base = env.assumption_derived_constraints(self.engine.rvsdg, key);
        let refined = base.refine(&constraints);

        if refined.is_bottom() {
            self.contradictory = true;
        }

        if refined == base {
            return false;
        }

        self.constraint_frame.values.insert(key, refined);
        self.cache.clear();

        true
    }

    fn saturate(&mut self) {
        let mut keys = IndexSet::new();
        let mut switches = IndexSet::new();
        let mut implications = Vec::new();

        for _ in 0..MAX_SATURATION_ROUNDS {
            if self.contradictory || self.engine.remaining_work_budget == 0 {
                return;
            }

            let env = self.parent.with_constraint_frame(&self.constraint_frame);

            keys.clear();
            switches.clear();

            for frame in &env.frames {
                keys.extend(frame.values.keys().copied());
            }

            for &key in &keys {
                if !env.is_in_context(self.engine.rvsdg, key.region) {
                    continue;
                }

                if self
                    .engine
                    .evaluate_value(key, &env, &mut self.cache)
                    .is_bottom()
                {
                    self.contradictory = true;

                    return;
                }

                if let ValueOrigin::Output { producer, .. } = key.origin {
                    if self.engine.rvsdg[producer].is_switch() {
                        // We defer backward propagation from switch outputs into/past switch nodes
                        // until we've processed all keys for this round, as these may include
                        // multiple output values for the same switch node. We only want to process
                        // each switch node once per round, aggregating the implications for the
                        // switch node's branch-selector value all at once.
                        switches.insert(producer);
                    } else {
                        collect_simple_output_implications(
                            self.engine,
                            key,
                            &env,
                            &mut self.cache,
                            &mut implications,
                        );
                    }
                }
            }

            // Now process the switch nodes we deferred earlier.
            for &switch in &switches {
                if !env.is_in_context(self.engine.rvsdg, self.engine.rvsdg[switch].region()) {
                    continue;
                }

                if collect_switch_output_implications(
                    self.engine,
                    &mut self.cache,
                    switch,
                    &env,
                    &mut implications,
                ) {
                    self.contradictory = true;

                    return;
                }
            }

            // Apply the constraints inferred during this round. If any constraint changes, the
            // next round evaluates all constraints again using the updated frame.
            let mut changed = false;

            for (key, value) in implications.drain(..) {
                changed |= self.record_value(key, value);

                if self.contradictory {
                    return;
                }
            }

            if !changed {
                return;
            }
        }
    }
}

/// Collects the implications of the constraints on a simple operation's output values for the
/// node's input value(s).
fn collect_simple_output_implications(
    engine: &mut Engine<'_>,
    key: ValueKey,
    env: &Environment,
    cache: &mut QueryCache,
    implications: &mut Vec<(ValueKey, AbstractValue)>,
) {
    let ValueOrigin::Output { producer, output } = key.origin else {
        return;
    };

    let output_constraints = env.assumption_derived_constraints(engine.rvsdg, key);

    match (engine.rvsdg[producer].kind(), output) {
        (NodeKind::Simple(SimpleNode::OpBoolToBranchSelector(_)), 0) => {
            let AbstractValue::Predicate(predicate) = &output_constraints else {
                unreachable!("Boolean selector output is a predicate");
            };

            if !predicate.is_top() {
                implications.push((
                    ValueKey::for_node_input(engine.rvsdg, producer, 0),
                    predicate.abstract_to_bool().into(),
                ));
            }
        }
        (NodeKind::Simple(SimpleNode::OpCaseToBranchSelector(selector)), 0) => {
            let AbstractValue::Predicate(predicate) = &output_constraints else {
                unreachable!("case selector output is a predicate");
            };

            if !predicate.is_top() {
                let source = ValueKey::for_node_input(engine.rvsdg, producer, 0);
                let value = engine.evaluate_value(source, env, cache);

                let refined = match &value {
                    AbstractValue::I32(value) => {
                        predicate.abstract_to_i32(value, selector.cases()).into()
                    }
                    AbstractValue::U32(value) => {
                        predicate.abstract_to_u32(value, selector.cases()).into()
                    }
                    _ => unreachable!("case selector input is an integer"),
                };

                implications.push((source, refined));
            }
        }
        (NodeKind::Simple(SimpleNode::OpUnary(op)), 0) => {
            let input = ValueKey::for_node_input(engine.rvsdg, producer, 0);
            let value = engine.evaluate_value(input, env, cache);
            let refined = value.abstract_unary_op_inv(op.operator(), &output_constraints);

            if refined != value {
                implications.push((input, refined));
            }
        }
        (NodeKind::Simple(SimpleNode::OpBinary(op)), 0) => {
            let op = op.operator();
            let lhs_input = ValueKey::for_node_input(engine.rvsdg, producer, 0);
            let rhs_input = ValueKey::for_node_input(engine.rvsdg, producer, 1);
            let lhs = engine.evaluate_value(lhs_input, env, cache);
            let rhs = engine.evaluate_value(rhs_input, env, cache);
            let (refined_lhs, refined_rhs) =
                lhs.abstract_binary_op_inv(op, &rhs, &output_constraints);

            if refined_lhs != lhs {
                implications.push((lhs_input, refined_lhs));
            }

            if refined_rhs != rhs {
                implications.push((rhs_input, refined_rhs));
            }
        }
        (NodeKind::Simple(SimpleNode::OpConvertToU32(_) | SimpleNode::OpConvertToI32(_)), 0) => {
            let input = ValueKey::for_node_input(engine.rvsdg, producer, 0);

            if matches!(input.ty(engine.rvsdg), TY_U32 | TY_I32) {
                let input_ty = input.ty(engine.rvsdg);

                let required = match (&output_constraints, input_ty) {
                    (AbstractValue::I32(value), TY_I32) => value.clone().into(),
                    (AbstractValue::I32(value), TY_U32) => value.to_abstract_u32().into(),
                    (AbstractValue::U32(value), TY_I32) => value.to_abstract_i32().into(),
                    (AbstractValue::U32(value), TY_U32) => value.clone().into(),
                    _ => AbstractValue::top(input_ty),
                };

                implications.push((input, required));
            }
        }
        _ => {}
    }
}

/// Collects the implications of the constraints on a switch node's output values for the switch
/// node's branch-selector value and for the switch node's branch result values.
///
/// If saturation reaches a switch node output, we'll derive the implications for that switch node's
/// branch-selector value. We'll also propagate the implications for the switch output to its
/// corresponding branch result, but only if we find that exactly one branch is feasible.
///
/// Back-propagating constraints on branch-region arguments to their corresponding switch input
/// value from multiple feasible branches would require joining rather than intersecting the
/// constraints, but we don't currently have a way to handle this. Therefore, for the time being, we
/// side-step the issue by only propagating saturation into a switch node's branches if we've proven
/// only a single feasible branch.
///
/// Returns `true` if the implications lead to a contradiction where none of the switch node's
/// branches are feasible.
fn collect_switch_output_implications(
    engine: &mut Engine<'_>,
    cache: &mut QueryCache,
    switch: Node,
    env: &Environment,
    implications: &mut Vec<(ValueKey, AbstractValue)>,
) -> bool {
    let selector_key = ValueKey::for_node_input(engine.rvsdg, switch, 0);
    let selector = engine.evaluate_value(selector_key, env, cache);
    let count = engine.rvsdg[switch].expect_switch().branches().len();
    let keys = env.recorded_switch_output_keys(switch);

    let mut choices = branches_matching_selector(&selector, count);

    choices.retain(|&branch| {
        keys.iter().all(|&key| {
            let ValueOrigin::Output { output, .. } = key.origin else {
                unreachable!()
            };

            let value = engine.branch_result_constraints(switch, output, branch, env, cache);

            !value
                .refine(&env.assumption_derived_constraints(engine.rvsdg, key))
                .is_bottom()
        })
    });

    if choices.is_empty() {
        return true;
    }

    implications.push((
        selector_key,
        AbstractPredicate::from_values(choices.iter().map(|&branch| branch as u32)).into(),
    ));

    if choices.len() == 1 {
        let branch = choices[0];

        for key in keys {
            let ValueOrigin::Output { output, .. } = key.origin else {
                unreachable!()
            };

            let result_key = ValueKey::for_branch_result(engine.rvsdg, switch, output, branch);

            implications.push((
                result_key,
                env.assumption_derived_constraints(engine.rvsdg, key),
            ));
        }
    }

    false
}
