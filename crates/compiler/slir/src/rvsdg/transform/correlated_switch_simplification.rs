//! Simplifies nested switch nodes using constraints implied by enclosing switch branches.
//!
//! Entering branch `n` of a switch node implies that the switch node's selector predicate was `n`.
//! We can propagate this constraining fact further up through the reverse value-flow that reaches
//! the switch node's branch-selector input, thus potentially constraining other values upstream in
//! the value-flow. If the value-flow from any such constrained values also reaches switch inputs
//! other than the branch-selector, then the corresponding branch arguments may inherit these
//! constraints. We can then use these constraints to simplify dependent values within the switch
//! branch.
//!
//! Propagating all constraints implied by branch-selection to all nodes within the branch would be
//! computationally expensive; this pass does not attempt to do so. Instead, we focus on a narrow
//! but high-impact special case: an "inner" switch node is nested inside a branch of an "outer"
//! switch node. The "outer" switch node's branch-selector value originates from an "upstream"
//! switch node. The "inner" switch node's branch-selector value originates from a different output
//! value on the same "upstream" switch node. The branch that contains the "inner" switch node
//! constrains the "outer" switch node's branch-selector value; it must have been a value that
//! mapped to that specific branch. This in turn constrains the branches that the "upstream" switch
//! node could have taken; only branches that produce a value that is compatible with the "outer"
//! switch node's branch-selector constraint will have been "feasible". This set of "feasible
//! branches" then finally constrains the value of the "inner" switch node's branch-selector. For
//! example, if each of the remaining feasible branches produce a constant value result, then we can
//! prune branches on the "inner" switch node that we can now prove to be unreachable. If only one
//! of the "inner" switch node's branches remains reachable, we can inline that branch and eliminate
//! the "inner" switch node entirely.
//!
//! In the scenario described above, we say that the "inner" switch node's branch-selector is
//! correlated with the "outer" switch node's branch-selector. The above scenario may seem very
//! specific and contrived. However, such scenarios actually turn out to be very common in SLIR that
//! was derived from RISL code: Rust `enum` values commonly produce this pattern, and typical Rust
//! code makes extensive use of `enum` values.
//!
//! Note that this transform assumes the RVSDG is in "predicate continuation form" as produced by
//! the [branch_selector_normalization] transform.
//!
//! [branch_selector_normalization]: crate::rvsdg::transform::branch_selector_normalization

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::rvsdg::transform::region_replication::inline_switch_branch;
use crate::rvsdg::transform::switch_branch_pruning::retain_switch_branches;
use crate::rvsdg::{Connectivity, Node, NodeKind, Region, Rvsdg, SimpleNode, ValueOrigin};
use crate::{Function, Module};

const MAX_EVAL_DEPTH: usize = 16;
const MAX_NOT_IN_VALUES: usize = 32;

type ValueKey = (Region, ValueOrigin);
type FeasibleBranches = SmallVec<[usize; 4]>;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ScalarConstant {
    U32(u32),
    I32(i32),
    F32(u32),
    Bool(bool),
    Predicate(u32),
}

impl ScalarConstant {
    fn from_node(rvsdg: &Rvsdg, node: Node, output: u32) -> Option<Self> {
        if output != 0 {
            return None;
        }

        match rvsdg[node].kind() {
            NodeKind::Simple(SimpleNode::ConstU32(value)) => Some(Self::U32(value.value())),
            NodeKind::Simple(SimpleNode::ConstI32(value)) => Some(Self::I32(value.value())),
            NodeKind::Simple(SimpleNode::ConstF32(value)) => {
                Some(Self::F32(value.value().to_bits()))
            }
            NodeKind::Simple(SimpleNode::ConstBool(value)) => Some(Self::Bool(value.value())),
            NodeKind::Simple(SimpleNode::ConstPredicate(value)) => {
                Some(Self::Predicate(value.value()))
            }
            _ => None,
        }
    }

    fn integer_encoding(self) -> Option<u128> {
        match self {
            Self::U32(value) => Some(value as u128),
            Self::I32(value) => Some(value as u32 as u128),
            _ => None,
        }
    }
}

/// A fact known about a value while traversing a particular switch branch.
///
/// Constraints range from no knowledge ([`Unknown`](Self::Unknown)) to a concrete constant or a
/// bounded set of excluded integer encodings. [`Impossible`](Self::Impossible) represents
/// contradictory facts and is used to identify infeasible branches.
#[derive(Clone, PartialEq, Eq, Debug)]
enum ValueConstraint {
    /// The value has no known constraints.
    Unknown,

    /// The value is known to equal a constant.
    Const(ScalarConstant),

    /// The value is known not to be in the given set of values.
    NotIn(Vec<u128>),

    /// The value would reflect an impossibility.
    ///
    /// In particular, if the [meet] of constraints on a switch branch result value produces an
    /// "impossible" constraint, then the branch cannot have been taken.
    Impossible,
}

impl ValueConstraint {
    /// Combines two constraints into a new constraint that is at least as strong as the individual
    /// constraints.
    ///
    /// `Unknown` leaves the other constraint unchanged, while `Impossible` remains impossible.
    /// Equal constants remain constant and unequal constants are impossible. A constant combined
    /// with `NotIn` is impossible when its integer encoding is excluded and otherwise remains
    /// constant. Two `NotIn` constraints combine their exclusions; if their union would exceed
    /// [`MAX_NOT_IN_VALUES`], the existing exclusions are retained as a conservative result.
    fn meet(&self, other: &Self) -> Self {
        use ValueConstraint::*;

        match (self, other) {
            (Impossible, _) | (_, Impossible) => Impossible,
            (Unknown, value) | (value, Unknown) => value.clone(),
            (Const(left), Const(right)) => {
                if left == right {
                    Const(*left)
                } else {
                    Impossible
                }
            }
            (Const(value), NotIn(values)) | (NotIn(values), Const(value)) => {
                if value
                    .integer_encoding()
                    .is_some_and(|value| values.contains(&value))
                {
                    Impossible
                } else {
                    Const(*value)
                }
            }
            (NotIn(left), NotIn(right)) => {
                let mut values = left.clone();

                for value in right {
                    if !values.contains(value) {
                        if values.len() == MAX_NOT_IN_VALUES {
                            return NotIn(left.clone());
                        }

                        values.push(*value);
                    }
                }

                NotIn(values)
            }
        }
    }

    fn is_impossible(&self) -> bool {
        matches!(self, Self::Impossible)
    }
}

#[derive(Default)]
struct UndoLog {
    values: Vec<(ValueKey, Option<ValueConstraint>)>,
    known_branches: Vec<Node>,
}

/// Records constraining facts about values and switch branch-selection, scoped by the branch
/// region that produced the provoking branch-selector constraint.
///
/// Whenever we enter a switch branch, a constraint is introduced on the switch nodes branch
/// selector, from which additional constraints on the upstream value-flow can be inferred (see the
/// module-level documentation for details). When entering a branch for a nested switch node,
/// additional constraints can be added. When leaving a branch, the branch-selector constraint and
/// all upstream constraints inferred from it must be reverted so that the fact environment does not
/// become inaccurate when entering a sibling branch or a branch for another switch node.
///
/// To achieve this, we use a fact environment "scope stack". When entering a branch, a new scope
/// should be added to the stack by calling [push_scope], before recording facts for this branch.
/// When leaving a branch, the scope should be popped from the stack by calling [pop_scope]. For
/// each scope we maintain an "undo log". When the scope is popped, we use this log to revert the
/// fact environment to the state it was in before the scope was pushed.
#[derive(Default)]
struct FactEnv {
    values: FxHashMap<ValueKey, ValueConstraint>,

    /// Secondary index over [values]: maps a producer node to the region that contains it and the
    /// outputs of that node for which [values] currently records a fact.
    ///
    /// Kept in sync with [values] by [constrain_value] (which adds entries for newly recorded
    /// output facts) and [pop_scope] (which removes entries when the corresponding facts are
    /// undone), so that output facts can be looked up by producer node without scanning the whole
    /// fact map.
    constrained_outputs: FxHashMap<Node, (Region, SmallVec<[u32; 4]>)>,

    known_branches: FxHashMap<Node, usize>,
    scopes: Vec<UndoLog>,
}

impl FactEnv {
    fn clear(&mut self) {
        self.values.clear();
        self.constrained_outputs.clear();
        self.known_branches.clear();
        self.scopes.clear();
    }

    /// Adds a new scope to the fact environment.
    fn push_scope(&mut self) {
        self.scopes.push(UndoLog::default());
    }

    /// Removes the top scope and all facts recorded within it.
    ///
    /// This undoes all facts recorded for the top scope, loosening value and switch branch
    /// constraints back to what they were before the scope was entered.
    fn pop_scope(&mut self) {
        let scope = self.scopes.pop().expect("fact scope should be active");

        for (key, previous) in scope.values.into_iter().rev() {
            if let Some(previous) = previous {
                self.values.insert(key, previous);
            } else {
                self.values.remove(&key);

                if let (_, ValueOrigin::Output { producer, output }) = key {
                    let (_, outputs) = self
                        .constrained_outputs
                        .get_mut(&producer)
                        .expect("constrained-output index should contain the fact being undone");

                    outputs.retain(|o| *o != output);

                    if outputs.is_empty() {
                        self.constrained_outputs.remove(&producer);
                    }
                }
            }
        }

        for switch in scope.known_branches.into_iter().rev() {
            self.known_branches.remove(&switch);
        }
    }

    /// For the current scope, records the fact that the given value is subject to the given
    /// constraint.
    ///
    /// If the value was already constrained previously, the new constraint will be the "meet" (see
    /// [ValueConstraint::meet]) of the previous constraint and the new constraint.
    ///
    /// Returns `true` if the constraint was strengthened, `false` otherwise.
    ///
    /// Any strengthening of the constraint is also recorded into the top scope's "undo log", so
    /// that the strengthening can be undone when the scope is popped.
    fn constrain_value(&mut self, value: ValueKey, constraint: ValueConstraint) -> bool {
        let previous = self.values.get(&value).cloned();
        let combined = previous
            .as_ref()
            .unwrap_or(&ValueConstraint::Unknown)
            .meet(&constraint);

        if previous.as_ref() == Some(&combined)
            || previous.is_none() && combined == ValueConstraint::Unknown
        {
            return false;
        }

        if previous.is_none()
            && let (region, ValueOrigin::Output { producer, output }) = value
        {
            let (index_region, outputs) = self
                .constrained_outputs
                .entry(producer)
                .or_insert_with(|| (region, SmallVec::new()));

            debug_assert_eq!(*index_region, region);
            debug_assert!(!outputs.contains(&output));

            outputs.push(output);
        }

        self.values.insert(value, combined);
        self.scopes
            .last_mut()
            .expect("fact scope should be active")
            .values
            .push((value, previous));

        true
    }

    /// For the current scope, records the knowledge that the given switch node's control-flow must
    /// flow through the given branch.
    ///
    /// If the fact environment did not already contain the knowledge that the switch node's
    /// control-flow must flow through the given branch, then we also add an entry to the top
    /// scope's "undo log", so that this knowledge can be undone when the scope is popped.
    fn record_known_branch(&mut self, switch: Node, branch: usize) -> bool {
        if let Some(&known) = self.known_branches.get(&switch) {
            debug_assert_eq!(known, branch);

            return false;
        }

        self.known_branches.insert(switch, branch);
        self.scopes
            .last_mut()
            .expect("fact scope should be active")
            .known_branches
            .push(switch);

        true
    }

    /// Maps each output value of the given switch node to its accumulated constraint for the
    /// current scope.
    fn switch_output_constraints(&self, switch: Node) -> SmallVec<[(u32, ValueConstraint); 4]> {
        let Some((region, outputs)) = self.constrained_outputs.get(&switch) else {
            return SmallVec::new();
        };

        outputs
            .iter()
            .map(|&output| {
                let key = (
                    *region,
                    ValueOrigin::Output {
                        producer: switch,
                        output,
                    },
                );
                let fact = self.values[&key].clone();

                (output, fact)
            })
            .collect()
    }
}

/// Derives the constraint on the value that produces the given switch node's branch selector, as
/// implied by entering the given branch.
///
/// This assumes that the RVSDG is in predicate continuation form and that the switch node's
/// branch-selector input connects directly to an [OpBoolToBranchSelector] or
/// [OpCaseToBranchSelector] node. The constraint implied on its input depends on the node kind:
///
/// - If the branch-selector is produced by an [OpBoolToBranchSelector] node, then branch `0`
///   implies that the input value is `true`; branch `1` implies that the input value is `false`.
/// - If the branch-selector is produced by an [OpCaseToBranchSelector] node, then if the branch
///   maps to a case, the input value is equal to the case value; if the branch is the default
///   (last) branch, then the input value cannot be equal to any of the case values.
///
fn branch_selector_constraint(
    rvsdg: &Rvsdg,
    switch: Node,
    branch: usize,
) -> Option<(ValueOrigin, ValueConstraint)> {
    let ValueOrigin::Output {
        producer,
        output: 0,
    } = rvsdg[switch].expect_switch().branch_selector().origin
    else {
        return None;
    };

    match rvsdg[producer].kind() {
        NodeKind::Simple(SimpleNode::OpBoolToBranchSelector(_)) => {
            let fact = match branch {
                0 => ValueConstraint::Const(ScalarConstant::Bool(true)),
                1 => ValueConstraint::Const(ScalarConstant::Bool(false)),
                _ => return None,
            };

            Some((rvsdg[producer].value_inputs()[0].origin, fact))
        }
        NodeKind::Simple(SimpleNode::OpCaseToBranchSelector(selector)) => {
            let fact = if let Some(&case) = selector.cases().get(branch) {
                let constant = if selector.encoding().signed {
                    ScalarConstant::I32(case as u32 as i32)
                } else {
                    ScalarConstant::U32(case as u32)
                };

                ValueConstraint::Const(constant)
            } else if branch == selector.cases().len() {
                ValueConstraint::NotIn(selector.cases().to_vec())
            } else {
                return None;
            };

            Some((rvsdg[producer].value_inputs()[0].origin, fact))
        }
        _ => None,
    }
}

fn canonicalize(rvsdg: &Rvsdg, mut region: Region, mut origin: ValueOrigin) -> ValueKey {
    loop {
        match origin {
            ValueOrigin::Output {
                producer,
                output: 0,
            } if rvsdg.is_live_node(producer) => {
                if let NodeKind::Simple(SimpleNode::ValueProxy(proxy)) = rvsdg[producer].kind() {
                    region = rvsdg[producer].region();
                    origin = proxy.value_inputs()[0].origin;

                    continue;
                }
            }
            ValueOrigin::Argument(argument)
                if region != rvsdg.global_region() && rvsdg.is_live_region(region) =>
            {
                let owner = rvsdg[region].owner();

                if !rvsdg.is_live_node(owner) {
                    break;
                }

                let outer_region = rvsdg[owner].region();

                match rvsdg[owner].kind() {
                    NodeKind::Switch(switch) => {
                        origin = switch.value_inputs()[argument as usize + 1].origin;
                        region = outer_region;

                        continue;
                    }
                    NodeKind::Loop(loop_node) => {
                        let loop_region = loop_node.loop_region();
                        let result = argument as usize + 1;

                        // We only resolve a loop argument to an outer value if the loop-value is
                        // loop-invariant.
                        if rvsdg[loop_region].value_results()[result].origin
                            == ValueOrigin::Argument(argument)
                        {
                            origin = loop_node.value_inputs()[argument as usize].origin;
                            region = outer_region;

                            continue;
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }

        break;
    }

    (region, origin)
}

enum BranchResultSummary {
    Const(ScalarConstant),
    Fallback,
    Alias(ValueKey),
}

impl BranchResultSummary {
    fn summarize(rvsdg: &Rvsdg, switch: Node, output: u32, branch: usize) -> Self {
        let branch_region = rvsdg[switch].expect_switch().branches()[branch];
        let origin = rvsdg[branch_region].value_results()[output as usize].origin;

        if let ValueOrigin::Output {
            producer,
            output: producer_output,
        } = origin
        {
            if let Some(constant) = ScalarConstant::from_node(rvsdg, producer, producer_output) {
                return Self::Const(constant);
            }

            if producer_output == 0
                && matches!(
                    rvsdg[producer].kind(),
                    NodeKind::Simple(SimpleNode::ConstFallback(_))
                )
            {
                return Self::Fallback;
            }
        }

        Self::Alias(canonicalize(rvsdg, branch_region, origin))
    }
}

pub struct CorrelatedSwitchSimplifier {
    env: FactEnv,
    visited: FxHashSet<ValueKey>,
}

impl CorrelatedSwitchSimplifier {
    pub fn new() -> Self {
        Self {
            env: FactEnv::default(),
            visited: FxHashSet::default(),
        }
    }

    pub fn simplify_in_fn(
        &mut self,
        module: &mut Module,
        rvsdg: &mut Rvsdg,
        function: Function,
    ) -> bool {
        let function_node = rvsdg
            .get_function_node(function)
            .expect("function not registered");
        let body = rvsdg[function_node].expect_function().body_region();

        self.env.clear();
        self.visit_region(module, rvsdg, body)
    }

    fn visit_region(&mut self, module: &mut Module, rvsdg: &mut Rvsdg, region: Region) -> bool {
        let nodes = rvsdg[region]
            .nodes()
            .iter()
            .copied()
            .filter(|&node| matches!(rvsdg[node].kind(), NodeKind::Loop(_) | NodeKind::Switch(_)))
            .collect::<Vec<_>>();

        let mut changed = false;

        for node in nodes {
            match rvsdg[node].kind() {
                NodeKind::Loop(loop_node) => {
                    changed |= self.visit_region(module, rvsdg, loop_node.loop_region());
                }
                NodeKind::Switch(_) => {
                    changed |= self.visit_switch(module, rvsdg, node);
                }
                _ => {}
            }
        }

        changed
    }

    fn visit_switch(&mut self, module: &mut Module, rvsdg: &mut Rvsdg, switch: Node) -> bool {
        let region = rvsdg[switch].region();
        let branch_count = rvsdg[switch].expect_switch().branches().len();
        let feasible_branches = self.find_feasible_branches(rvsdg, switch);

        let mut changed = false;

        for &branch_index in &feasible_branches {
            self.env.push_scope();

            if let Some((origin, constraint)) =
                branch_selector_constraint(rvsdg, switch, branch_index)
            {
                self.broadcast_branch_selector_constraint(rvsdg, region, origin, constraint);
            }

            let branch_region = rvsdg[switch].expect_switch().branches()[branch_index];

            changed |= self.visit_region(module, rvsdg, branch_region);

            self.env.pop_scope();
        }

        match feasible_branches.len() {
            0 => changed,
            1 => {
                inline_switch_branch(module, rvsdg, switch, feasible_branches[0]);

                true
            }
            count if count < branch_count => {
                retain_switch_branches(rvsdg, switch, &feasible_branches);

                true
            }
            count => {
                assert_eq!(count, branch_count);

                changed
            }
        }
    }

    fn broadcast_branch_selector_constraint(
        &mut self,
        rvsdg: &Rvsdg,
        region: Region,
        origin: ValueOrigin,
        constraint: ValueConstraint,
    ) {
        let key = canonicalize(rvsdg, region, origin);

        // Record the constraint in the fact environment. If this does not strengthen the prior
        // constraint on the selector value, then there's no new information to propagate and we
        // can exit early.
        if !self.env.constrain_value(key, constraint) {
            return;
        }

        loop {
            let constrained_switches = self
                .env
                .constrained_outputs
                .iter()
                .filter_map(|(&producer, &(region, _))| {
                    // A fact may outlive a switch node that was already simplified away
                    // earlier, so verify that the producer is still live
                    (rvsdg.is_live_node(producer) && rvsdg[producer].is_switch())
                        .then_some((region, producer))
                })
                .collect::<Vec<_>>();

            let mut changed = false;

            for (region, switch) in constrained_switches {
                let feasible = self.find_feasible_branches(rvsdg, switch);

                let branch = if let Some(&branch) = self.env.known_branches.get(&switch) {
                    branch
                } else if feasible.len() == 1 {
                    let branch = feasible[0];

                    changed |= self.env.record_known_branch(switch, branch);

                    if let Some((origin, selector_constraint)) =
                        branch_selector_constraint(rvsdg, switch, branch)
                    {
                        let key = canonicalize(rvsdg, region, origin);

                        changed |= self.env.constrain_value(key, selector_constraint);
                    }

                    branch
                } else {
                    continue;
                };

                let output_constraints = self.env.switch_output_constraints(switch);

                for (output, constraint) in output_constraints {
                    if let BranchResultSummary::Alias((alias_region, alias_origin)) =
                        BranchResultSummary::summarize(rvsdg, switch, output, branch)
                    {
                        let key = canonicalize(rvsdg, alias_region, alias_origin);

                        changed |= self.env.constrain_value(key, constraint);
                    }
                }
            }

            if !changed {
                break;
            }
        }
    }

    /// Resolves the strongest known constraint for the given value.
    fn eval(
        &self,
        rvsdg: &Rvsdg,
        region: Region,
        origin: ValueOrigin,
        depth: usize,
        visited: &mut FxHashSet<ValueKey>,
    ) -> ValueConstraint {
        // Stop following producer and switch-summary dependencies once the recursion budget is
        // exhausted; returning Unknown conservatively prevents this path from proving anything.
        if depth == 0 {
            return ValueConstraint::Unknown;
        }

        let key = canonicalize(rvsdg, region, origin);

        // If the key was already recorded for this evaluation path, then there is a cycle in the
        // analysis and we bail.
        if !visited.insert(key) {
            return ValueConstraint::Unknown;
        }

        let mut value = self
            .env
            .values
            .get(&key)
            .cloned()
            .unwrap_or(ValueConstraint::Unknown);

        if let ValueOrigin::Output { producer, output } = key.1
            && rvsdg.is_live_node(producer)
        {
            if let Some(constant) = ScalarConstant::from_node(rvsdg, producer, output) {
                value = value.meet(&ValueConstraint::Const(constant));
            } else if rvsdg[producer].is_switch() {
                let feasible = self.compute_feasible_branches(rvsdg, producer, depth - 1, visited);

                if feasible.len() == 1 {
                    let summary =
                        BranchResultSummary::summarize(rvsdg, producer, output, feasible[0]);
                    let summary_value = self.eval_summary(rvsdg, summary, depth - 1, visited);

                    value = value.meet(&summary_value);
                } else if !feasible.is_empty() {
                    let mut common = None;
                    let mut all_constant = true;

                    for branch in feasible {
                        match BranchResultSummary::summarize(rvsdg, producer, output, branch) {
                            BranchResultSummary::Const(constant)
                                if common.is_none() || common == Some(constant) =>
                            {
                                common = Some(constant);
                            }
                            _ => {
                                all_constant = false;

                                break;
                            }
                        }
                    }

                    if all_constant && let Some(constant) = common {
                        value = value.meet(&ValueConstraint::Const(constant));
                    }
                }
            }
        }

        // Remove the key so that other independent evaluation paths can inspect it later.
        visited.remove(&key);

        value
    }

    fn eval_summary(
        &self,
        rvsdg: &Rvsdg,
        summary: BranchResultSummary,
        depth: usize,
        visited: &mut FxHashSet<ValueKey>,
    ) -> ValueConstraint {
        match summary {
            BranchResultSummary::Const(constant) => ValueConstraint::Const(constant),
            BranchResultSummary::Fallback => ValueConstraint::Unknown,
            BranchResultSummary::Alias((region, origin)) => {
                self.eval(rvsdg, region, origin, depth, visited)
            }
        }
    }

    fn find_feasible_branches(&mut self, rvsdg: &Rvsdg, switch: Node) -> FeasibleBranches {
        let mut visited = std::mem::take(&mut self.visited);

        visited.clear();

        let feasible = self.compute_feasible_branches(rvsdg, switch, MAX_EVAL_DEPTH, &mut visited);

        self.visited = visited;

        feasible
    }

    fn compute_feasible_branches(
        &self,
        rvsdg: &Rvsdg,
        switch: Node,
        depth: usize,
        visited: &mut FxHashSet<ValueKey>,
    ) -> FeasibleBranches {
        if let Some(&branch) = self.env.known_branches.get(&switch) {
            return SmallVec::from_slice(&[branch]);
        }

        let branch_count = rvsdg[switch].expect_switch().branches().len();
        let outer_region = rvsdg[switch].region();

        let mut feasible = (0..branch_count).collect::<FeasibleBranches>();

        if depth > 0 {
            let selector_origin = rvsdg[switch].expect_switch().branch_selector().origin;

            if let ValueOrigin::Output {
                producer,
                output: 0,
            } = selector_origin
                && rvsdg.is_live_node(producer)
            {
                match rvsdg[producer].kind() {
                    NodeKind::Simple(SimpleNode::ConstPredicate(predicate)) => {
                        feasible.retain(|branch| *branch == predicate.value() as usize);
                    }
                    NodeKind::Simple(SimpleNode::OpBoolToBranchSelector(_)) => {
                        let source = rvsdg[producer].value_inputs()[0].origin;

                        match self.eval(rvsdg, outer_region, source, depth - 1, visited) {
                            ValueConstraint::Const(ScalarConstant::Bool(true)) => {
                                feasible.retain(|branch| *branch == 0);
                            }
                            ValueConstraint::Const(ScalarConstant::Bool(false)) => {
                                feasible.retain(|branch| *branch == 1);
                            }
                            _ => {}
                        }
                    }
                    NodeKind::Simple(SimpleNode::OpCaseToBranchSelector(selector)) => {
                        let source = rvsdg[producer].value_inputs()[0].origin;

                        match self.eval(rvsdg, outer_region, source, depth - 1, visited) {
                            ValueConstraint::Const(constant) => {
                                if let Some(value) = constant.integer_encoding() {
                                    let selected = selector
                                        .cases()
                                        .iter()
                                        .position(|case| *case == value)
                                        .unwrap_or(selector.cases().len());

                                    feasible.retain(|branch| *branch == selected);
                                }
                            }
                            ValueConstraint::NotIn(excluded) => {
                                feasible.retain(|branch| {
                                    *branch == selector.cases().len()
                                        || !excluded.contains(&selector.cases()[*branch])
                                });
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
        }

        let constraints = self.env.switch_output_constraints(switch);

        for (output, constraint) in constraints {
            feasible.retain(|branch| {
                let summary = BranchResultSummary::summarize(rvsdg, switch, output, *branch);
                let summary_value = self.eval_summary(rvsdg, summary, depth, visited);

                !summary_value.meet(&constraint).is_impossible()
            });
        }

        feasible
    }
}

pub fn transform_entry_points(module: &mut Module, rvsdg: &mut Rvsdg) -> bool {
    let mut simplifier = CorrelatedSwitchSimplifier::new();
    let mut changed = false;
    let entry_points = module
        .entry_points
        .iter()
        .map(|entry| entry.0)
        .collect::<Vec<_>>();

    for entry_point in entry_points {
        changed |= simplifier.simplify_in_fn(module, rvsdg, entry_point);
    }

    changed
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::rvsdg::{ValueInput, ValueOutput};
    use crate::ty::{Int, TY_BOOL, TY_DUMMY, TY_PREDICATE, TY_U32};
    use crate::{FnArg, FnSig, Symbol};

    #[test]
    fn outer_switch_and_inner_switch_share_bool_branch_selector() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref("test"),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
                ty: TY_DUMMY,
                args: vec![FnArg {
                    ty: TY_BOOL,
                    shader_io_binding: None,
                }],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, body) = rvsdg.register_function(&module, function, iter::empty());

        // The outer switch also passes in as an argument, the same bool value it also uses as
        // branch-selector. The inner switch then also uses this same bool value as its
        // branch-selector. This implies that if the outer switch took branch `0`, the inner switch
        // will also take branch `0`.
        let outer_selector =
            rvsdg.add_op_bool_to_branch_selector(body, ValueInput::argument(TY_BOOL, 0));
        let outer_switch = rvsdg.add_switch(
            body,
            vec![
                ValueInput::output(TY_PREDICATE, outer_selector, 0),
                ValueInput::argument(TY_BOOL, 0),
            ],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let outer_branch_0 = rvsdg.add_switch_branch(outer_switch);

        let inner_selector =
            rvsdg.add_op_bool_to_branch_selector(outer_branch_0, ValueInput::argument(TY_BOOL, 0));
        let inner_switch = rvsdg.add_switch(
            outer_branch_0,
            vec![ValueInput::output(TY_PREDICATE, inner_selector, 0)],
            vec![ValueOutput::new(TY_U32)],
            None,
        );
        let inner_branch_0 = rvsdg.add_switch_branch(inner_switch);
        let inner_branch_0_value = rvsdg.add_const_u32(inner_branch_0, 1);

        rvsdg.reconnect_region_result(
            inner_branch_0,
            0,
            ValueOrigin::Output {
                producer: inner_branch_0_value,
                output: 0,
            },
        );

        let inner_branch_1 = rvsdg.add_switch_branch(inner_switch);
        let inner_branch_1_value = rvsdg.add_const_u32(inner_branch_1, 2);

        rvsdg.reconnect_region_result(
            inner_branch_1,
            0,
            ValueOrigin::Output {
                producer: inner_branch_1_value,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(
            outer_branch_0,
            0,
            ValueOrigin::Output {
                producer: inner_switch,
                output: 0,
            },
        );

        let outer_branch_1 = rvsdg.add_switch_branch(outer_switch);
        let outer_branch_1_value = rvsdg.add_const_u32(outer_branch_1, 3);

        rvsdg.reconnect_region_result(
            outer_branch_1,
            0,
            ValueOrigin::Output {
                producer: outer_branch_1_value,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(
            body,
            0,
            ValueOrigin::Output {
                producer: outer_switch,
                output: 0,
            },
        );

        let mut simplifier = CorrelatedSwitchSimplifier::new();

        assert!(simplifier.simplify_in_fn(&mut module, &mut rvsdg, function));

        assert!(rvsdg.is_live_node(outer_switch));
        assert!(!rvsdg.is_live_node(inner_switch));

        let ValueOrigin::Output {
            producer,
            output: 0,
        } = rvsdg[outer_branch_0].value_results()[0].origin
        else {
            panic!("expected the simplified branch to return a constant");
        };
        assert_eq!(rvsdg[producer].expect_const_u32().value(), 1);
    }

    #[test]
    fn outer_switch_and_inner_switch_share_case_branch_selector() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref("test"),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
                ty: TY_DUMMY,
                args: vec![FnArg {
                    ty: TY_U32,
                    shader_io_binding: None,
                }],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, body) = rvsdg.register_function(&module, function, iter::empty());

        let outer_selector = rvsdg.add_op_case_to_branch_selector(
            body,
            ValueInput::argument(TY_U32, 0),
            Int::U32,
            [0],
        );
        let outer_switch = rvsdg.add_switch(
            body,
            vec![
                ValueInput::output(TY_PREDICATE, outer_selector, 0),
                ValueInput::argument(TY_U32, 0),
            ],
            vec![ValueOutput::new(TY_U32)],
            None,
        );
        let outer_branch_0 = rvsdg.add_switch_branch(outer_switch);
        let zero_value = rvsdg.add_const_u32(outer_branch_0, 0);

        rvsdg.reconnect_region_result(
            outer_branch_0,
            0,
            ValueOrigin::Output {
                producer: zero_value,
                output: 0,
            },
        );

        let outer_branch_1 = rvsdg.add_switch_branch(outer_switch);

        let inner_selector = rvsdg.add_op_case_to_branch_selector(
            outer_branch_1,
            ValueInput::argument(TY_U32, 0),
            Int::U32,
            [0, 1, 2],
        );
        let inner_switch = rvsdg.add_switch(
            outer_branch_1,
            vec![ValueInput::output(TY_PREDICATE, inner_selector, 0)],
            vec![ValueOutput::new(TY_U32)],
            None,
        );
        let inner_branch_0 = rvsdg.add_switch_branch(inner_switch);
        let inner_branch_0_value = rvsdg.add_const_u32(inner_branch_0, 10);

        rvsdg.reconnect_region_result(
            inner_branch_0,
            0,
            ValueOrigin::Output {
                producer: inner_branch_0_value,
                output: 0,
            },
        );

        let inner_branch_1 = rvsdg.add_switch_branch(inner_switch);
        let inner_branch_1_value = rvsdg.add_const_u32(inner_branch_1, 11);

        rvsdg.reconnect_region_result(
            inner_branch_1,
            0,
            ValueOrigin::Output {
                producer: inner_branch_1_value,
                output: 0,
            },
        );

        let inner_branch_2 = rvsdg.add_switch_branch(inner_switch);
        let inner_branch_2_value = rvsdg.add_const_u32(inner_branch_2, 12);

        rvsdg.reconnect_region_result(
            inner_branch_2,
            0,
            ValueOrigin::Output {
                producer: inner_branch_2_value,
                output: 0,
            },
        );

        let inner_default = rvsdg.add_switch_branch(inner_switch);
        let inner_default_value = rvsdg.add_const_u32(inner_default, 13);

        rvsdg.reconnect_region_result(
            inner_default,
            0,
            ValueOrigin::Output {
                producer: inner_default_value,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(
            outer_branch_1,
            0,
            ValueOrigin::Output {
                producer: inner_switch,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(
            body,
            0,
            ValueOrigin::Output {
                producer: outer_switch,
                output: 0,
            },
        );

        let mut simplifier = CorrelatedSwitchSimplifier::new();

        assert!(simplifier.simplify_in_fn(&mut module, &mut rvsdg, function));

        assert!(rvsdg.is_live_node(inner_switch));
        assert_eq!(
            rvsdg[inner_switch].expect_switch().branches(),
            &[inner_branch_1, inner_branch_2, inner_default]
        );

        let ValueOrigin::Output {
            producer: new_selector,
            output: 0,
        } = rvsdg[inner_switch].expect_switch().branch_selector().origin
        else {
            panic!("expected the pruned switch to use a case selector");
        };
        assert_eq!(
            rvsdg[new_selector]
                .expect_op_case_to_branch_selector()
                .cases(),
            &[1, 2]
        );
    }

    #[test]
    fn correlated_switch_selectors_from_different_upstream_outputs() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref("test"),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
                ty: TY_DUMMY,
                args: vec![FnArg {
                    ty: TY_BOOL,
                    shader_io_binding: None,
                }],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, body) = rvsdg.register_function(&module, function, iter::empty());

        let upstream_selector =
            rvsdg.add_op_bool_to_branch_selector(body, ValueInput::argument(TY_BOOL, 0));
        let upstream_switch = rvsdg.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, upstream_selector, 0)],
            vec![ValueOutput::new(TY_U32), ValueOutput::new(TY_U32)],
            None,
        );

        let upstream_branch_0 = rvsdg.add_switch_branch(upstream_switch);
        let upstream_branch_0_outer_selector = rvsdg.add_const_u32(upstream_branch_0, 0);
        let upstream_branch_0_inner_selector = rvsdg.add_const_u32(upstream_branch_0, 10);

        rvsdg.reconnect_region_result(
            upstream_branch_0,
            0,
            ValueOrigin::Output {
                producer: upstream_branch_0_outer_selector,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            upstream_branch_0,
            1,
            ValueOrigin::Output {
                producer: upstream_branch_0_inner_selector,
                output: 0,
            },
        );

        let upstream_branch_1 = rvsdg.add_switch_branch(upstream_switch);
        let upstream_branch_1_outer_selector = rvsdg.add_const_u32(upstream_branch_1, 1);
        let upstream_branch_1_inner_selector = rvsdg.add_const_u32(upstream_branch_1, 20);

        rvsdg.reconnect_region_result(
            upstream_branch_1,
            0,
            ValueOrigin::Output {
                producer: upstream_branch_1_outer_selector,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            upstream_branch_1,
            1,
            ValueOrigin::Output {
                producer: upstream_branch_1_inner_selector,
                output: 0,
            },
        );

        let outer_selector = rvsdg.add_op_case_to_branch_selector(
            body,
            ValueInput::output(TY_U32, upstream_switch, 0),
            Int::U32,
            [0],
        );
        let outer_switch = rvsdg.add_switch(
            body,
            vec![
                ValueInput::output(TY_PREDICATE, outer_selector, 0),
                ValueInput::output(TY_U32, upstream_switch, 1),
            ],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let outer_branch_0 = rvsdg.add_switch_branch(outer_switch);

        let inner_selector = rvsdg.add_op_case_to_branch_selector(
            outer_branch_0,
            ValueInput::argument(TY_U32, 0),
            Int::U32,
            [10],
        );
        let inner_switch = rvsdg.add_switch(
            outer_branch_0,
            vec![ValueInput::output(TY_PREDICATE, inner_selector, 0)],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let inner_branch_0 = rvsdg.add_switch_branch(inner_switch);
        let inner_branch_0_value = rvsdg.add_const_u32(inner_branch_0, 100);

        rvsdg.reconnect_region_result(
            inner_branch_0,
            0,
            ValueOrigin::Output {
                producer: inner_branch_0_value,
                output: 0,
            },
        );

        let inner_branch_1 = rvsdg.add_switch_branch(inner_switch);
        let inner_branch_1_value = rvsdg.add_const_u32(inner_branch_1, 101);

        rvsdg.reconnect_region_result(
            inner_branch_1,
            0,
            ValueOrigin::Output {
                producer: inner_branch_1_value,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(
            outer_branch_0,
            0,
            ValueOrigin::Output {
                producer: inner_switch,
                output: 0,
            },
        );

        let outer_branch_1 = rvsdg.add_switch_branch(outer_switch);
        let outer_branch_1_value = rvsdg.add_const_u32(outer_branch_1, 200);

        rvsdg.reconnect_region_result(
            outer_branch_1,
            0,
            ValueOrigin::Output {
                producer: outer_branch_1_value,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(
            body,
            0,
            ValueOrigin::Output {
                producer: outer_switch,
                output: 0,
            },
        );

        let mut simplifier = CorrelatedSwitchSimplifier::new();

        assert!(simplifier.simplify_in_fn(&mut module, &mut rvsdg, function));

        assert!(rvsdg.is_live_node(upstream_switch));
        assert!(rvsdg.is_live_node(outer_switch));
        assert!(!rvsdg.is_live_node(inner_switch));

        let ValueOrigin::Output {
            producer,
            output: 0,
        } = rvsdg[outer_branch_0].value_results()[0].origin
        else {
            panic!("expected the correlated branch to return a constant");
        };
        assert_eq!(rvsdg[producer].expect_const_u32().value(), 100);
    }

    #[test]
    fn upstream_switch_with_constant_output() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref("test"),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
                ty: TY_DUMMY,
                args: vec![FnArg {
                    ty: TY_BOOL,
                    shader_io_binding: None,
                }],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, body) = rvsdg.register_function(&module, function, iter::empty());

        let producer_selector =
            rvsdg.add_op_bool_to_branch_selector(body, ValueInput::argument(TY_BOOL, 0));
        let producer_switch = rvsdg.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, producer_selector, 0)],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let producer_branch_0 = rvsdg.add_switch_branch(producer_switch);
        let value_0 = rvsdg.add_const_u32(producer_branch_0, 1);

        rvsdg.reconnect_region_result(
            producer_branch_0,
            0,
            ValueOrigin::Output {
                producer: value_0,
                output: 0,
            },
        );

        let producer_branch_1 = rvsdg.add_switch_branch(producer_switch);
        let value_1 = rvsdg.add_const_u32(producer_branch_1, 1);

        rvsdg.reconnect_region_result(
            producer_branch_1,
            0,
            ValueOrigin::Output {
                producer: value_1,
                output: 0,
            },
        );

        let consumer_selector = rvsdg.add_op_case_to_branch_selector(
            body,
            ValueInput::output(TY_U32, producer_switch, 0),
            Int::U32,
            [1],
        );
        let consumer_switch = rvsdg.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, consumer_selector, 0)],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let consumer_branch_0 = rvsdg.add_switch_branch(consumer_switch);
        let selected_value = rvsdg.add_const_u32(consumer_branch_0, 10);

        rvsdg.reconnect_region_result(
            consumer_branch_0,
            0,
            ValueOrigin::Output {
                producer: selected_value,
                output: 0,
            },
        );

        let consumer_branch_1 = rvsdg.add_switch_branch(consumer_switch);
        let default_value = rvsdg.add_const_u32(consumer_branch_1, 20);

        rvsdg.reconnect_region_result(
            consumer_branch_1,
            0,
            ValueOrigin::Output {
                producer: default_value,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(
            body,
            0,
            ValueOrigin::Output {
                producer: consumer_switch,
                output: 0,
            },
        );

        let mut simplifier = CorrelatedSwitchSimplifier::new();

        assert!(simplifier.simplify_in_fn(&mut module, &mut rvsdg, function));

        assert!(rvsdg.is_live_node(producer_switch));
        assert!(!rvsdg.is_live_node(consumer_switch));

        let ValueOrigin::Output {
            producer,
            output: 0,
        } = rvsdg[body].value_results()[0].origin
        else {
            panic!("expected the function to return the selected constant");
        };
        assert_eq!(rvsdg[producer].expect_const_u32().value(), 10);
    }
}
