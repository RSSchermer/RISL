//! Uses [cva] to simplify away simple nodes and switch branches.
//!
//! # Simple Nodes
//!
//! Currently, attempts to simplify the following simple node kinds: [OpBinary], [OpUnary],
//! [OpConvertToU32], [OpConvertToI32], [OpConvertToF32], [OpConvertToBool],
//! [OpCaseToBranchSelector], and [OpBoolToBranchSelector]. If [cva] finds that the output of one
//! such node is a singleton constant value, then the node is replaced with that constant value.
//! Additionally, the scalar conversion operations ([OpConvertToU32], [OpConvertToI32],
//! [OpConvertToF32], and [OpConvertToBool]) are replaced with their input value if the input
//! value's type is already the conversion's target type. There are also some additional special
//! rules for [OpBinary] nodes:
//!
//! - If the operation is addition, and [cva] evaluates either the left-hand-side or the
//!   right-hand-side input to a constant `0` value, then the node is replaced with its other
//!   input.
//! - If the operation is subtraction, and [cva] evaluates the right-hand-side input to a
//!   constant `0` value, then the node is replaced with the left-hand-side input.
//! - If the operation is a bit-wise "or" or a bit-wise "xor", and [cva] evaluates either the
//!   left-hand-side or the right-hand-side input to a constant `0` value, then the node is replaced
//!   with its other input.
//! - If the operation is a left shift or a right shift, and [cva] evaluates the right-hand-side
//!   input to a constant `0` value, then the node is replaced with the left-hand-side input.
//! - If the operation is multiplication, and [cva] evaluates either the left-hand-side or the
//!   right-hand-side input to a constant `1` value, then the node is replaced with the other
//!   input value.
//! - If the operation is division, and [cva] evaluates the right-hand-side input to a constant
//!   `1` value, then the node is replaced with the left-hand-side input.
//! - If the operation is a Boolean "and", and [cva] evaluates either the left-hand-side or the
//!   right-hand-side input to a constant `true` value, then the node is replaced with the other
//!   input.
//! - If the operation is a Boolean "or", and [cva] evaluates either the left-hand-side or the
//!   right-hand-side input to a constant `false` value, then the node is replaced with the other
//!   input.
//!
//! For the time being, the rules for numerical values only affect `u32` and `i32` values; `f32` are
//! currently excluded.
//!
//! If the simple scalar rules described above do not apply, and the node's output value is a
//! boolean, we attempt "conditional-scalar reduction". This looks for switch nodes in the node's
//! upstream value-flow. If the switch node's branch-selector is produced by an
//! [OpBoolToBranchSelector] node, then we use [cva] to evaluate the node's output value under both
//! branch assumptions to establish a potential relation between the node's output value and the
//! upstream switch node's branch-selector condition value. If we find that the node's output value
//! is always equal to the upstream condition value, then we remove the node and reconnect all users
//! to the upstream condition value. If we find that the node's output value is always the
//! complement of the upstream condition value, then we add an [OpUnary] "not" node to inverse
//! the upstream condition value and reconnect all users to the output of the "not" node.
//!
//! # Switch Branch Arguments
//!
//! Switch branch arguments that [cva] evaluates to singleton scalar constants are "materialized"
//! as branch-local constants. All users of the argument are reconnected to the output of the new
//! constant. Unused arguments and arguments whose switch entry input already connects directly to a
//! scalar constant are left unchanged.
//!
//! # Switch Nodes
//!
//! We also use [cva] to evaluate a switch node's branch-selector input. If [cva] finds that the
//! possible set of values for switch node's branch-selector input does not cover all branch indices
//! for the switch node, then the "infeasible" branches are simplified away. If there are still two
//! or more "feasible" branches, then the infeasible branches are "pruned" (see
//! [switch_branch_pruning]). If there is only a single feasible branch, then this one branch is
//! "inlined" (see [inline_switch_branch]) into the switch node's outer region.
//!
//! [cva]: crate::rvsdg::analyse::cva
//! [OpBinary]: crate::rvsdg::OpBinary
//! [OpUnary]: crate::rvsdg::OpUnary
//! [OpConvertToU32]: crate::rvsdg::OpConvertToU32
//! [OpConvertToI32]: crate::rvsdg::OpConvertToI32
//! [OpConvertToF32]: crate::rvsdg::OpConvertToF32
//! [OpConvertToBool]: crate::rvsdg::OpConvertToBool
//! [OpCaseToBranchSelector]: crate::rvsdg::OpCaseToBranchSelector
//! [OpBoolToBranchSelector]: crate::rvsdg::OpBoolToBranchSelector
//! [switch_branch_pruning]: crate::rvsdg::transform::switch_branch_pruning
use std::collections::VecDeque;

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::rvsdg::analyse::cva::{
    AbstractValue, AssumptionResult, BooleanRelation, ContextId, ContextualValueAnalysis, ValueKey,
    query_boolean_relation,
};
use crate::rvsdg::analyse::region_stratification::RegionStratifier;
use crate::rvsdg::analyse::scalar_constant::ScalarConstant;
use crate::rvsdg::transform::region_replication::inline_switch_branch;
use crate::rvsdg::transform::switch_branch_pruning::retain_switch_branches;
use crate::rvsdg::{
    Connectivity, Node, NodeKind, Region, Rvsdg, SimpleNode, ValueInput, ValueOrigin,
};
use crate::ty::TY_BOOL;
use crate::{BinaryOperator, Function, Module, UnaryOperator};

#[derive(Clone, Copy, Debug, PartialEq)]
enum ReplacementValue {
    Constant(ScalarConstant),
    Variable(ValueInput),
    Complement(ValueInput),
}

impl From<ScalarConstant> for ReplacementValue {
    fn from(value: ScalarConstant) -> Self {
        Self::Constant(value)
    }
}

struct ReduceToValueJob {
    target: Node,
    replacement: ReplacementValue,
}

impl ReduceToValueJob {
    fn apply_reduction(self, rvsdg: &mut Rvsdg) {
        let region = rvsdg[self.target].region();
        let new_origin = match self.replacement {
            ReplacementValue::Variable(input) => input.origin,
            ReplacementValue::Constant(constant) => ValueOrigin::Output {
                producer: constant.add_to_region(rvsdg, region),
                output: 0,
            },
            ReplacementValue::Complement(input) => {
                let producer = rvsdg.add_op_unary(region, UnaryOperator::Not, input);

                ValueOrigin::Output {
                    producer,
                    output: 0,
                }
            }
        };
        let original_origin = ValueOrigin::Output {
            producer: self.target,
            output: 0,
        };

        rvsdg.reconnect_value_users(region, original_origin, new_origin);
        rvsdg.remove_node(self.target);
    }
}

struct ReduceSwitchJob {
    target: Node,
    retained_branches: Vec<usize>,
}

struct ReduceSwitchArgumentJob {
    region: Region,
    argument: u32,
    constant: ScalarConstant,
}

/// Uses [cva] to simplify away simple nodes and switch branches.
///
/// See the [module-level documentation](crate::rvsdg::transform::contextual_node_reduction) for
/// details.
#[derive(Default)]
pub struct ContextualNodeReducer {
    reduce_to_value_jobs: Vec<ReduceToValueJob>,
    reduce_switch_argument_jobs: Vec<ReduceSwitchArgumentJob>,
    reduce_switch_jobs: Vec<ReduceSwitchJob>,
}

impl ContextualNodeReducer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn optimize_function(
        &mut self,
        module: &mut Module,
        rvsdg: &mut Rvsdg,
        function: Function,
    ) -> bool {
        self.collect_jobs(rvsdg, function);

        let mut changed = false;

        changed |= self.apply_reduce_to_value_jobs(rvsdg);
        changed |= self.apply_reduce_switch_argument_jobs(rvsdg);
        changed |= self.apply_reduce_switch_jobs(module, rvsdg);

        changed
    }

    fn collect_jobs(&mut self, rvsdg: &Rvsdg, function: Function) {
        debug_assert!(
            self.reduce_to_value_jobs.is_empty()
                && self.reduce_switch_argument_jobs.is_empty()
                && self.reduce_switch_jobs.is_empty()
        );

        let fn_node = rvsdg
            .get_function_node(function)
            .expect("function not registered");
        let body = rvsdg[fn_node].expect_function().body_region();
        let mut analysis = ContextualValueAnalysis::new(rvsdg, function);
        let root_context = analysis.root_context();

        self.collect_jobs_in_region(rvsdg, &mut analysis, body, root_context);
    }

    fn apply_reduce_to_value_jobs(&mut self, rvsdg: &mut Rvsdg) -> bool {
        let changed = !self.reduce_to_value_jobs.is_empty();

        while let Some(plan) = self.reduce_to_value_jobs.pop() {
            assert!(
                rvsdg.is_live_node(plan.target),
                "pending scalar was removed"
            );

            plan.apply_reduction(rvsdg);
        }

        changed
    }

    fn apply_reduce_switch_argument_jobs(&mut self, rvsdg: &mut Rvsdg) -> bool {
        let mut changed = false;

        while let Some(job) = self.reduce_switch_argument_jobs.pop() {
            if !switch_argument_needs_constant(rvsdg, job.region, job.argument) {
                continue;
            }

            let producer = job.constant.add_to_region(rvsdg, job.region);

            rvsdg.reconnect_value_users(
                job.region,
                ValueOrigin::Argument(job.argument),
                ValueOrigin::Output {
                    producer,
                    output: 0,
                },
            );
            changed = true;
        }

        changed
    }

    fn apply_reduce_switch_jobs(&mut self, module: &mut Module, rvsdg: &mut Rvsdg) -> bool {
        let changed = !self.reduce_switch_jobs.is_empty();

        while let Some(plan) = self.reduce_switch_jobs.pop() {
            assert!(
                rvsdg.is_live_node(plan.target),
                "pending switch was removed"
            );

            if plan.retained_branches.len() == 1 {
                inline_switch_branch(module, rvsdg, plan.target, plan.retained_branches[0]);
            } else {
                retain_switch_branches(rvsdg, plan.target, &plan.retained_branches);
            }
        }

        changed
    }

    fn collect_jobs_in_region(
        &mut self,
        rvsdg: &Rvsdg,
        analysis: &mut ContextualValueAnalysis,
        region: Region,
        context: ContextId,
    ) {
        if rvsdg[rvsdg[region].owner()].is_switch() {
            for argument in 0..rvsdg[region].value_arguments().len() as u32 {
                if !switch_argument_needs_constant(rvsdg, region, argument) {
                    continue;
                }

                let value = analysis.evaluate_value(
                    ValueKey::new(region, ValueOrigin::Argument(argument)),
                    context,
                );

                if let Some(constant) = value.to_scalar_constant() {
                    self.reduce_switch_argument_jobs
                        .push(ReduceSwitchArgumentJob {
                            region,
                            argument,
                            constant,
                        });
                }
            }
        }

        let mut selector_proofs = FxHashMap::default();
        let mut stratifier = RegionStratifier::new();

        stratifier.stratify(rvsdg, region, |node, _| {
            if let Some(job) = try_reduce_to_value(rvsdg, analysis, node, context) {
                if let ReplacementValue::Constant(ScalarConstant::Predicate(index)) =
                    job.replacement
                {
                    selector_proofs.insert(node, index);
                }

                self.reduce_to_value_jobs.push(job);
            }

            match rvsdg[node].kind() {
                NodeKind::Switch(switch) => {
                    let selector =
                        analysis.evaluate_value(ValueKey::for_node_input(rvsdg, node, 0), context);
                    let AbstractValue::Predicate(selector) = selector else {
                        panic!("switch selector must be a predicate")
                    };
                    let proof = match switch.branch_selector().origin {
                        ValueOrigin::Output {
                            producer,
                            output: 0,
                        } => selector_proofs.get(&producer),
                        _ => None,
                    };

                    let mut retained = Vec::new();
                    let mut children = Vec::new();

                    for (branch, &branch_region) in switch.branches().iter().enumerate() {
                        if !selector.contains(branch as u32)
                            || proof.is_some_and(|&index| index as usize != branch)
                        {
                            continue;
                        }

                        match analysis.assume_branch(context, node, branch) {
                            AssumptionResult::Contradictory => {}
                            AssumptionResult::Context(branch_context) => {
                                retained.push(branch);
                                children.push((branch_region, branch_context));
                            }
                            AssumptionResult::Unknown => retained.push(branch),
                        }
                    }

                    if !retained.is_empty()
                        && (retained.len() == 1
                            || (retained.len() < switch.branches().len()
                                && has_case_selector(rvsdg, node)))
                    {
                        self.reduce_switch_jobs.push(ReduceSwitchJob {
                            target: node,
                            retained_branches: retained,
                        });
                    }

                    for (branch_region, branch_context) in children {
                        self.collect_jobs_in_region(rvsdg, analysis, branch_region, branch_context);
                    }
                }
                NodeKind::Loop(loop_node) => {
                    let loop_region = loop_node.loop_region();

                    if let AssumptionResult::Context(loop_context) =
                        analysis.context_for_region(context, loop_region)
                    {
                        self.collect_jobs_in_region(rvsdg, analysis, loop_region, loop_context);
                    }
                }
                _ => {}
            }
        });
    }
}

pub fn transform_entry_points(module: &mut Module, rvsdg: &mut Rvsdg) -> bool {
    let mut optimizer = ContextualNodeReducer::new();
    let entry_points = module.entry_points.iter().collect::<Vec<_>>();
    let mut changed = false;

    for (entry_point, _) in entry_points {
        while optimizer.optimize_function(module, rvsdg, entry_point) {
            changed = true;
        }
    }

    changed
}

fn switch_argument_needs_constant(rvsdg: &Rvsdg, region: Region, argument: u32) -> bool {
    if rvsdg[region].value_arguments()[argument as usize]
        .users
        .is_empty()
    {
        return false;
    }

    let switch = rvsdg[rvsdg[region].owner()].expect_switch();

    match switch.entry_inputs()[argument as usize].origin {
        ValueOrigin::Output { producer, output } => {
            ScalarConstant::from_node(rvsdg, producer, output).is_none()
        }
        ValueOrigin::Argument(_) => true,
    }
}

fn has_case_selector(rvsdg: &Rvsdg, switch: Node) -> bool {
    if let ValueOrigin::Output {
        producer,
        output: 0,
    } = rvsdg[switch].expect_switch().branch_selector().origin
    {
        matches!(
            rvsdg[producer].kind(),
            NodeKind::Simple(SimpleNode::OpCaseToBranchSelector(_))
        )
    } else {
        false
    }
}

fn is_candidate(rvsdg: &Rvsdg, node: Node) -> bool {
    use NodeKind::*;
    use SimpleNode::*;

    matches!(
        rvsdg[node].kind(),
        Simple(
            OpBinary(_)
                | OpUnary(_)
                | OpConvertToU32(_)
                | OpConvertToI32(_)
                | OpConvertToF32(_)
                | OpConvertToBool(_)
                | OpCaseToBranchSelector(_)
                | OpBoolToBranchSelector(_)
        )
    )
}

fn try_reduce_to_value(
    rvsdg: &Rvsdg,
    analysis: &mut ContextualValueAnalysis,
    node: Node,
    context: ContextId,
) -> Option<ReduceToValueJob> {
    let data = &rvsdg[node];

    if !is_candidate(rvsdg, node) {
        return None;
    }

    let region = data.region();
    let key = ValueKey::new(
        region,
        ValueOrigin::Output {
            producer: node,
            output: 0,
        },
    );
    let value = analysis.evaluate_value(key, context);

    if let Some(constant) = value.to_scalar_constant() {
        return Some(ReduceToValueJob {
            target: node,
            replacement: constant.into(),
        });
    }

    if value.is_bottom() {
        return None;
    }

    if let Some(replacement) = try_reduce_simple_scalar(rvsdg, analysis, node, context) {
        return Some(ReduceToValueJob {
            target: node,
            replacement,
        });
    }

    if data.value_outputs().len() == 1 && data.value_outputs()[0].ty == TY_BOOL {
        if let Some(job) = try_reduce_conditional_scalar(rvsdg, analysis, node, context) {
            return Some(job);
        }
    }

    None
}

fn try_reduce_simple_scalar(
    rvsdg: &Rvsdg,
    analysis: &mut ContextualValueAnalysis,
    node: Node,
    context: ContextId,
) -> Option<ReplacementValue> {
    use NodeKind::*;
    use SimpleNode::*;

    let data = &rvsdg[node];

    match data.kind() {
        Simple(OpBinary(op)) => {
            let lhs = resolve_value(rvsdg, analysis, node, 0, context)?;
            let rhs = resolve_value(rvsdg, analysis, node, 1, context)?;

            try_reduce_binary(op.operator(), lhs, rhs)
        }
        Simple(OpConvertToU32(_))
        | Simple(OpConvertToI32(_))
        | Simple(OpConvertToF32(_))
        | Simple(OpConvertToBool(_)) => {
            let value = resolve_value(rvsdg, analysis, node, 0, context)?;

            match value {
                ReplacementValue::Variable(input) if input.ty == data.value_outputs()[0].ty => {
                    Some(value)
                }
                _ => None,
            }
        }
        _ => None,
    }
}

fn is_conditional_scalar_dependency(rvsdg: &Rvsdg, node: Node) -> bool {
    use NodeKind::*;
    use SimpleNode::*;

    matches!(
        rvsdg[node].kind(),
        Simple(
            OpBinary(_)
                | OpUnary(_)
                | OpConvertToU32(_)
                | OpConvertToI32(_)
                | OpConvertToF32(_)
                | OpConvertToBool(_)
                | ValueProxy(_)
        )
    )
}

fn find_conditional_scalar_producers(
    rvsdg: &Rvsdg,
    node: Node,
) -> SmallVec<[(Node, ValueInput); 4]> {
    let region = rvsdg[node].region();

    let mut visited = FxHashSet::default();
    let mut producers = SmallVec::new();
    let mut queue = VecDeque::new();

    // Bound admitted scalar nodes, but finish scanning every node already queued.
    visited.insert(node);
    queue.push_back(node);

    while let Some(curr) = queue.pop_front() {
        for input in rvsdg[curr].value_inputs() {
            if let ValueOrigin::Output { producer, .. } = input.origin {
                if rvsdg[producer].region() != region {
                    continue;
                }

                if rvsdg[producer].is_switch() {
                    let switch_data = rvsdg[producer].expect_switch();

                    if switch_data.branches().len() != 2
                        || producers.iter().any(|&(p, _)| p == producer)
                    {
                        continue;
                    }

                    let ValueOrigin::Output {
                        producer: selector_producer,
                        output: 0,
                    } = switch_data.branch_selector().origin
                    else {
                        continue;
                    };

                    if !matches!(
                        rvsdg[selector_producer].kind(),
                        NodeKind::Simple(SimpleNode::OpBoolToBranchSelector(_))
                    ) {
                        continue;
                    }

                    let condition = rvsdg[selector_producer].value_inputs()[0];

                    debug_assert_eq!(rvsdg[selector_producer].region(), region);
                    debug_assert_eq!(condition.ty, TY_BOOL);

                    producers.push((producer, condition));

                    if producers.len() == 4 {
                        return producers;
                    }
                } else if is_conditional_scalar_dependency(rvsdg, producer)
                    && visited.len() < 256
                    && visited.insert(producer)
                {
                    queue.push_back(producer);
                }
            }
        }
    }

    producers
}

fn try_reduce_conditional_scalar(
    rvsdg: &Rvsdg,
    analysis: &mut ContextualValueAnalysis,
    node: Node,
    context: ContextId,
) -> Option<ReduceToValueJob> {
    let region = rvsdg[node].region();
    let v_key = ValueKey::new(
        region,
        ValueOrigin::Output {
            producer: node,
            output: 0,
        },
    );

    for (switch, condition) in find_conditional_scalar_producers(rvsdg, node) {
        if let Some(relation) = query_boolean_relation(analysis, v_key, switch, context) {
            let replacement = match relation {
                BooleanRelation::Constant(b) => ReplacementValue::Constant(ScalarConstant::Bool(b)),
                BooleanRelation::Input => ReplacementValue::Variable(condition),
                BooleanRelation::Complement => ReplacementValue::Complement(condition),
            };

            return Some(ReduceToValueJob {
                target: node,
                replacement,
            });
        }
    }

    None
}

fn resolve_value(
    rvsdg: &Rvsdg,
    analysis: &mut ContextualValueAnalysis,
    node: Node,
    input_index: usize,
    context: ContextId,
) -> Option<ReplacementValue> {
    let input = rvsdg[node].value_inputs()[input_index];
    let value =
        analysis.evaluate_value(ValueKey::for_node_input(rvsdg, node, input_index), context);

    if value.is_bottom() {
        return None;
    }

    Some(
        value
            .to_scalar_constant()
            .map(Into::into)
            .unwrap_or(ReplacementValue::Variable(input)),
    )
}

fn try_reduce_binary(
    op: BinaryOperator,
    lhs: ReplacementValue,
    rhs: ReplacementValue,
) -> Option<ReplacementValue> {
    use BinaryOperator::*;
    use ReplacementValue::*;
    use ScalarConstant::*;

    match (op, lhs, rhs) {
        (Add | BitOr | BitXor, Variable(v), Constant(U32(0) | I32(0)))
        | (Add | BitOr | BitXor, Constant(U32(0) | I32(0)), Variable(v))
        | (Sub | Shl | Shr, Variable(v), Constant(U32(0) | I32(0)))
        | (Mul, Variable(v), Constant(U32(1) | I32(1)))
        | (Mul, Constant(U32(1) | I32(1)), Variable(v))
        | (Div, Variable(v), Constant(U32(1) | I32(1)))
        | (And, Variable(v), Constant(Bool(true)))
        | (And, Constant(Bool(true)), Variable(v))
        | (Or, Variable(v), Constant(Bool(false)))
        | (Or, Constant(Bool(false)), Variable(v)) => Some(Variable(v)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::ContextualNodeReducer;
    use crate::rvsdg::{
        Connectivity, Rvsdg, StateOrigin, StateUser, ValueInput, ValueOrigin, ValueOutput,
        ValueUser,
    };
    use crate::ty::{Int, TY_BOOL, TY_DUMMY, TY_F32, TY_I32, TY_PREDICATE, TY_PTR_U32, TY_U32};
    use crate::{
        BinaryOperator, BranchCase, FnArg, FnSig, Function, Module, Symbol, UnaryOperator, thin_set,
    };

    #[test]
    fn reduce_simple_op_chain_to_constant_value() {
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
                args: vec![],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, body) = rvsdg.register_function(&module, function, iter::empty());

        let one = rvsdg.add_const_u32(body, 1);
        let two = rvsdg.add_const_u32(body, 2);

        let four = rvsdg.add_const_u32(body, 4);

        let sum = rvsdg.add_op_binary(
            body,
            BinaryOperator::Add,
            ValueInput::output(TY_U32, one, 0),
            ValueInput::output(TY_U32, two, 0),
        );

        let product = rvsdg.add_op_binary(
            body,
            BinaryOperator::Mul,
            ValueInput::output(TY_U32, sum, 0),
            ValueInput::output(TY_U32, four, 0),
        );
        rvsdg.reconnect_region_result(
            body,
            0,
            ValueOrigin::Output {
                producer: product,
                output: 0,
            },
        );

        let mut optimizer = ContextualNodeReducer::new();

        assert!(optimizer.optimize_function(&mut module, &mut rvsdg, function));

        assert!(!rvsdg.is_live_node(sum));
        assert!(!rvsdg.is_live_node(product));

        let ValueOrigin::Output {
            producer,
            output: 0,
        } = rvsdg[body].value_results()[0].origin
        else {
            panic!("expected a node output");
        };
        assert_eq!(rvsdg[producer].region(), body);
        assert_eq!(rvsdg[producer].expect_const_u32().value(), 12);

        let result = rvsdg[body].value_results()[0];
        let nodes = rvsdg[body].nodes().clone();

        assert!(!optimizer.optimize_function(&mut module, &mut rvsdg, function));
        assert_eq!(rvsdg[body].value_results()[0], result);
        assert_eq!(rvsdg[body].nodes(), &nodes);
    }

    #[test]
    fn reduce_simple_op_chain_to_preexisting_value() {
        let mut module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let function = Function {
            name: Symbol::from_ref("f"),
            module: Symbol::from_ref("m"),
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
                ret_ty: Some(TY_BOOL),
            },
        );

        let (_, region) = rvsdg.register_function(&module, function, []);

        let const_true = rvsdg.add_const_bool(region, true);
        let const_false = rvsdg.add_const_bool(region, false);

        let and = rvsdg.add_op_binary(
            region,
            BinaryOperator::And,
            ValueInput::argument(TY_BOOL, 0),
            ValueInput::output(TY_BOOL, const_true, 0),
        );

        let or = rvsdg.add_op_binary(
            region,
            BinaryOperator::Or,
            ValueInput::output(TY_BOOL, const_false, 0),
            ValueInput::output(TY_BOOL, and, 0),
        );
        rvsdg.reconnect_region_result(region, 0, ValueInput::output(TY_BOOL, or, 0).origin);

        let mut optimizer = ContextualNodeReducer::new();

        assert!(optimizer.optimize_function(&mut module, &mut rvsdg, function));

        assert!(!rvsdg.is_live_node(and));
        assert!(!rvsdg.is_live_node(or));
        assert_eq!(
            rvsdg[region].value_results()[0],
            ValueInput::argument(TY_BOOL, 0)
        );

        assert!(!optimizer.optimize_function(&mut module, &mut rvsdg, function));
    }

    #[test]
    fn reduce_addition_using_branch_assumptions() {
        let mut module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let function = Function {
            name: Symbol::from_ref("f"),
            module: Symbol::from_ref("m"),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
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
                ret_ty: None,
            },
        );

        let (_, region) = rvsdg.register_function(&module, function, []);

        let zero = rvsdg.add_const_u32(region, 0);
        let condition = rvsdg.add_op_binary(
            region,
            BinaryOperator::Eq,
            ValueInput::argument(TY_U32, 0),
            ValueInput::output(TY_U32, zero, 0),
        );

        let selector =
            rvsdg.add_op_bool_to_branch_selector(region, ValueInput::output(TY_BOOL, condition, 0));
        let switch = rvsdg.add_switch(
            region,
            vec![
                ValueInput::output(TY_PREDICATE, selector, 0),
                ValueInput::argument(TY_U32, 0),
                ValueInput::argument(TY_U32, 1),
            ],
            vec![],
            None,
        );

        let branch = rvsdg.add_switch_branch(switch);

        // Contextual value analysis should infer that inside this branch, argument `0` is a
        // constant `0` value (since we test it against `0` to create the branch selector).

        let add_dynamic = rvsdg.add_op_binary(
            branch,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 1),
            ValueInput::argument(TY_U32, 0),
        );
        let dynamic_user =
            rvsdg.add_value_proxy(branch, ValueInput::output(TY_U32, add_dynamic, 0));

        let local_zero = rvsdg.add_const_u32(branch, 0);
        let add_constant = rvsdg.add_op_binary(
            branch,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 0),
            ValueInput::output(TY_U32, local_zero, 0),
        );
        let constant_user =
            rvsdg.add_value_proxy(branch, ValueInput::output(TY_U32, add_constant, 0));

        rvsdg.add_switch_branch(switch);

        let mut optimizer = ContextualNodeReducer::new();

        assert!(optimizer.optimize_function(&mut module, &mut rvsdg, function));

        assert!(!rvsdg.is_live_node(add_dynamic));
        assert!(!rvsdg.is_live_node(add_constant));

        assert_eq!(
            rvsdg[dynamic_user].value_inputs()[0],
            ValueInput::argument(TY_U32, 1)
        );

        let ValueOrigin::Output {
            producer,
            output: 0,
        } = rvsdg[constant_user].value_inputs()[0].origin
        else {
            panic!("expected a node output");
        };
        assert_eq!(rvsdg[producer].expect_const_u32().value(), 0);
    }

    #[test]
    fn materializes_constant_switch_argument_and_reconnects_all_users() {
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
        let (_, body) = rvsdg.register_function(&module, function, []);
        let ptr = rvsdg.add_op_alloca(body, TY_U32);
        let selector = rvsdg.add_op_case_to_branch_selector(
            body,
            ValueInput::argument(TY_U32, 0),
            Int::U32,
            [BranchCase::from(7u32)],
        );
        let switch = rvsdg.add_switch(
            body,
            vec![
                ValueInput::output(TY_PREDICATE, selector, 0),
                ValueInput::argument(TY_U32, 0),
                ValueInput::output(TY_PTR_U32, ptr, 0),
            ],
            vec![ValueOutput::new(TY_U32)],
            Some(StateOrigin::Argument),
        );

        let branch_0 = rvsdg.add_switch_branch(switch);
        let first_store = rvsdg.add_op_store(
            branch_0,
            ValueInput::argument(TY_PTR_U32, 1),
            ValueInput::argument(TY_U32, 0),
            StateOrigin::Argument,
        );
        let second_store = rvsdg.add_op_store(
            branch_0,
            ValueInput::argument(TY_PTR_U32, 1),
            ValueInput::argument(TY_U32, 0),
            StateOrigin::Node(first_store),
        );
        rvsdg.reconnect_region_result(branch_0, 0, ValueOrigin::Argument(0));

        let branch_1 = rvsdg.add_switch_branch(switch);
        rvsdg.reconnect_region_result(branch_1, 0, ValueOrigin::Argument(0));

        rvsdg.reconnect_region_result(body, 0, ValueInput::output(TY_U32, switch, 0).origin);

        let mut optimizer = ContextualNodeReducer::new();

        assert!(optimizer.optimize_function(&mut module, &mut rvsdg, function));

        let new_val = rvsdg[branch_0].value_results()[0];
        let ValueOrigin::Output {
            producer: materialized_const,
            output: 0,
        } = new_val.origin
        else {
            panic!("expected a node output");
        };
        assert_eq!(rvsdg[materialized_const].expect_const_u32().value(), 7);

        assert_eq!(rvsdg[first_store].value_inputs()[1], new_val);
        assert_eq!(rvsdg[second_store].value_inputs()[1], new_val);

        assert_eq!(
            rvsdg[materialized_const].value_outputs()[0].users,
            thin_set![
                ValueUser::Result(0),
                ValueUser::Input {
                    consumer: first_store,
                    input: 1
                },
                ValueUser::Input {
                    consumer: second_store,
                    input: 1
                }
            ]
        );
        assert!(rvsdg[branch_0].value_arguments()[0].users.is_empty());

        assert!(!optimizer.optimize_function(&mut module, &mut rvsdg, function));
    }

    #[test]
    fn skips_switch_arg_reduction_when_argument_is_unused() {
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
                ret_ty: None,
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, body) = rvsdg.register_function(&module, function, []);
        let selector = rvsdg.add_op_bool_to_branch_selector(body, ValueInput::argument(TY_BOOL, 0));
        let switch = rvsdg.add_switch(
            body,
            vec![
                ValueInput::output(TY_PREDICATE, selector, 0),
                ValueInput::output(TY_PREDICATE, selector, 0),
            ],
            vec![],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch);
        let branch_1 = rvsdg.add_switch_branch(switch);

        let mut optimizer = ContextualNodeReducer::new();

        assert!(!optimizer.optimize_function(&mut module, &mut rvsdg, function));
    }

    #[test]
    fn skips_switch_argument_reduction_when_input_folds_to_constant() {
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
        let (_, body) = rvsdg.register_function(&module, function, []);
        let one = rvsdg.add_const_u32(body, 1);
        let sum = rvsdg.add_op_binary(
            body,
            BinaryOperator::Add,
            ValueInput::output(TY_U32, one, 0),
            ValueInput::output(TY_U32, one, 0),
        );
        let selector = rvsdg.add_op_bool_to_branch_selector(body, ValueInput::argument(TY_BOOL, 0));
        let switch = rvsdg.add_switch(
            body,
            vec![
                ValueInput::output(TY_PREDICATE, selector, 0),
                ValueInput::output(TY_U32, sum, 0),
            ],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch);
        rvsdg.reconnect_region_result(branch_0, 0, ValueOrigin::Argument(0));

        let branch_1 = rvsdg.add_switch_branch(switch);
        rvsdg.reconnect_region_result(branch_1, 0, ValueOrigin::Argument(0));

        rvsdg.reconnect_region_result(body, 0, ValueInput::output(TY_U32, switch, 0).origin);

        let mut optimizer = ContextualNodeReducer::new();

        assert!(optimizer.optimize_function(&mut module, &mut rvsdg, function));

        assert!(!rvsdg.is_live_node(sum));

        assert!(rvsdg[branch_0].nodes().is_empty());
        assert!(rvsdg[branch_1].nodes().is_empty());

        assert_eq!(
            rvsdg[branch_0].value_results()[0],
            ValueInput::argument(TY_U32, 0)
        );
        assert_eq!(
            rvsdg[branch_1].value_results()[0],
            ValueInput::argument(TY_U32, 0)
        );

        let ValueOrigin::Output {
            producer,
            output: 0,
        } = rvsdg[switch].value_inputs()[1].origin
        else {
            panic!("expected a node output");
        };
        assert_eq!(rvsdg[producer].expect_const_u32().value(), 2);

        assert!(!optimizer.optimize_function(&mut module, &mut rvsdg, function));
    }

    #[test]
    fn inline_nested_switch_using_branch_assumptions() {
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

        let twenty = rvsdg.add_const_u32(body, 20);
        let outer_comparison = rvsdg.add_op_binary(
            body,
            BinaryOperator::LtEq,
            ValueInput::argument(TY_U32, 0),
            ValueInput::output(TY_U32, twenty, 0),
        );

        let outer_selector = rvsdg
            .add_op_bool_to_branch_selector(body, ValueInput::output(TY_BOOL, outer_comparison, 0));
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
        let hundred = rvsdg.add_const_u32(outer_branch_0, 100);
        let inner_comparison = rvsdg.add_op_binary(
            outer_branch_0,
            BinaryOperator::Lt,
            ValueInput::argument(TY_U32, 0),
            ValueInput::output(TY_U32, hundred, 0),
        );

        let inner_selector = rvsdg.add_op_bool_to_branch_selector(
            outer_branch_0,
            ValueInput::output(TY_BOOL, inner_comparison, 0),
        );
        let inner_switch = rvsdg.add_switch(
            outer_branch_0,
            vec![
                ValueInput::output(TY_PREDICATE, inner_selector, 0),
                ValueInput::argument(TY_U32, 0),
            ],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let inner_0_branch_0 = rvsdg.add_switch_branch(inner_switch);
        rvsdg.reconnect_region_result(inner_0_branch_0, 0, ValueOrigin::Argument(0));

        let inner_0_branch_1 = rvsdg.add_switch_branch(inner_switch);
        let impossible_value = rvsdg.add_const_u32(inner_0_branch_1, 999);
        rvsdg.reconnect_region_result(
            inner_0_branch_1,
            0,
            ValueOrigin::Output {
                producer: impossible_value,
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
        rvsdg.reconnect_region_result(outer_branch_1, 0, ValueOrigin::Argument(0));

        rvsdg.reconnect_region_result(
            body,
            0,
            ValueOrigin::Output {
                producer: outer_switch,
                output: 0,
            },
        );

        let mut optimizer = ContextualNodeReducer::new();

        assert!(optimizer.optimize_function(&mut module, &mut rvsdg, function));

        assert!(rvsdg.is_live_node(outer_switch));
        assert!(!rvsdg.is_live_node(inner_switch));
        assert_eq!(
            rvsdg[outer_branch_0].value_results()[0].origin,
            ValueOrigin::Argument(0)
        );
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
            [BranchCase::from(0u32)],
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
            [0u32, 1].map(BranchCase::from),
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

        let mut simplifier = ContextualNodeReducer::new();

        assert!(simplifier.optimize_function(&mut module, &mut rvsdg, function));

        assert!(rvsdg.is_live_node(inner_switch));
        assert_eq!(
            rvsdg[inner_switch].expect_switch().branches(),
            &[inner_branch_1, inner_default]
        );

        let ValueOrigin::Output {
            producer: new_selector,
            output: 0,
        } = rvsdg[inner_switch].expect_switch().branch_selector().origin
        else {
            panic!("expected a node output");
        };
        assert_eq!(
            rvsdg[new_selector]
                .expect_op_case_to_branch_selector()
                .cases(),
            &[BranchCase::from(1u32)]
        );
    }

    #[test]
    fn inline_single_branch_switch() {
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
        let selector = rvsdg.add_const_predicate(body, 0);
        let switch = rvsdg.add_switch(
            body,
            vec![
                ValueInput::output(TY_PREDICATE, selector, 0),
                ValueInput::argument(TY_U32, 0),
            ],
            vec![ValueOutput::new(TY_U32)],
            None,
        );
        let branch = rvsdg.add_switch_branch(switch);
        rvsdg.reconnect_region_result(branch, 0, ValueOrigin::Argument(0));
        rvsdg.reconnect_region_result(body, 0, ValueInput::output(TY_U32, switch, 0).origin);

        let mut optimizer = ContextualNodeReducer::new();

        assert!(optimizer.optimize_function(&mut module, &mut rvsdg, function));
        assert!(!rvsdg.is_live_node(switch));
        assert_eq!(
            rvsdg[body].value_results()[0],
            ValueInput::argument(TY_U32, 0)
        );
        assert!(!optimizer.optimize_function(&mut module, &mut rvsdg, function));
    }

    #[test]
    fn switch_reduction_composes_with_nested_simple_reductions() {
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
                args: vec![],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, body) = rvsdg.register_function(&module, function, iter::empty());

        let condition = rvsdg.add_const_bool(body, true);
        let outer_selector =
            rvsdg.add_op_bool_to_branch_selector(body, ValueInput::output(TY_BOOL, condition, 0));
        let outer_switch = rvsdg.add_switch(
            body,
            vec![
                ValueInput::output(TY_PREDICATE, outer_selector, 0),
                ValueInput::output(TY_BOOL, condition, 0),
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
        let one = rvsdg.add_const_u32(inner_branch_0, 1);
        let two = rvsdg.add_const_u32(inner_branch_0, 2);

        let four = rvsdg.add_const_u32(inner_branch_0, 4);

        let sum = rvsdg.add_op_binary(
            inner_branch_0,
            BinaryOperator::Add,
            ValueInput::output(TY_U32, one, 0),
            ValueInput::output(TY_U32, two, 0),
        );

        let product = rvsdg.add_op_binary(
            inner_branch_0,
            BinaryOperator::Mul,
            ValueInput::output(TY_U32, sum, 0),
            ValueInput::output(TY_U32, four, 0),
        );
        rvsdg.reconnect_region_result(
            inner_branch_0,
            0,
            ValueOrigin::Output {
                producer: product,
                output: 0,
            },
        );

        let inner_branch_1 = rvsdg.add_switch_branch(inner_switch);
        let inner_fallback = rvsdg.add_const_u32(inner_branch_1, 99);
        rvsdg.reconnect_region_result(
            inner_branch_1,
            0,
            ValueOrigin::Output {
                producer: inner_fallback,
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
        let outer_fallback = rvsdg.add_const_u32(outer_branch_1, 100);
        rvsdg.reconnect_region_result(
            outer_branch_1,
            0,
            ValueOrigin::Output {
                producer: outer_fallback,
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

        let mut optimizer = ContextualNodeReducer::new();

        assert!(optimizer.optimize_function(&mut module, &mut rvsdg, function));

        assert!(!rvsdg.is_live_node(sum));
        assert!(!rvsdg.is_live_node(product));
        assert!(!rvsdg.is_live_node(inner_selector));
        assert!(!rvsdg.is_live_node(inner_switch));
        assert!(!rvsdg.is_live_node(outer_selector));
        assert!(!rvsdg.is_live_node(outer_switch));

        let ValueOrigin::Output {
            producer,
            output: 0,
        } = rvsdg[body].value_results()[0].origin
        else {
            panic!("expected a node output");
        };
        assert_eq!(rvsdg[producer].expect_const_u32().value(), 12);

        assert!(!optimizer.optimize_function(&mut module, &mut rvsdg, function));
    }

    #[test]
    fn inline_switches_with_shared_selector() {
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
                args: vec![],
                ret_ty: None,
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, body) = rvsdg.register_function(&module, function, iter::empty());

        let two = rvsdg.add_const_u32(body, 2);

        let selector = rvsdg.add_op_case_to_branch_selector(
            body,
            ValueInput::output(TY_U32, two, 0),
            Int::U32,
            [0u32, 2].map(BranchCase::from),
        );
        let switch_0 = rvsdg.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, selector, 0)],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let switch_0_branch_0 = rvsdg.add_switch_branch(switch_0);
        let ten = rvsdg.add_const_u32(switch_0_branch_0, 10);
        rvsdg.reconnect_region_result(
            switch_0_branch_0,
            0,
            ValueInput::output(TY_U32, ten, 0).origin,
        );

        let switch_0_branch_1 = rvsdg.add_switch_branch(switch_0);
        let eleven = rvsdg.add_const_u32(switch_0_branch_1, 11);
        rvsdg.reconnect_region_result(
            switch_0_branch_1,
            0,
            ValueInput::output(TY_U32, eleven, 0).origin,
        );

        let switch_0_default_branch = rvsdg.add_switch_branch(switch_0);
        let twelve = rvsdg.add_const_u32(switch_0_default_branch, 12);
        rvsdg.reconnect_region_result(
            switch_0_default_branch,
            0,
            ValueInput::output(TY_U32, twelve, 0).origin,
        );

        let switch_1 = rvsdg.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, selector, 0)],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let switch_1_branch_0 = rvsdg.add_switch_branch(switch_1);
        let twenty = rvsdg.add_const_u32(switch_1_branch_0, 20);
        rvsdg.reconnect_region_result(
            switch_1_branch_0,
            0,
            ValueInput::output(TY_U32, twenty, 0).origin,
        );

        let switch_1_branch_1 = rvsdg.add_switch_branch(switch_1);
        let twenty_one = rvsdg.add_const_u32(switch_1_branch_1, 21);
        rvsdg.reconnect_region_result(
            switch_1_branch_1,
            0,
            ValueInput::output(TY_U32, twenty_one, 0).origin,
        );

        let switch_1_default_branch = rvsdg.add_switch_branch(switch_1);
        let twenty_two = rvsdg.add_const_u32(switch_1_default_branch, 22);
        rvsdg.reconnect_region_result(
            switch_1_default_branch,
            0,
            ValueInput::output(TY_U32, twenty_two, 0).origin,
        );

        let ptr = rvsdg.add_op_alloca(body, TY_U32);
        let store_0 = rvsdg.add_op_store(
            body,
            ValueInput::output(TY_PTR_U32, ptr, 0),
            ValueInput::output(TY_U32, switch_0, 0),
            StateOrigin::Argument,
        );
        let store_1 = rvsdg.add_op_store(
            body,
            ValueInput::output(TY_PTR_U32, ptr, 0),
            ValueInput::output(TY_U32, switch_1, 0),
            StateOrigin::Node(store_0),
        );

        let mut optimizer = ContextualNodeReducer::new();

        assert!(optimizer.optimize_function(&mut module, &mut rvsdg, function));

        assert!(!rvsdg.is_live_node(selector));
        assert!(!rvsdg.is_live_node(switch_0));
        assert!(!rvsdg.is_live_node(switch_1));

        let ValueOrigin::Output {
            producer: switch_0_result,
            output: 0,
        } = rvsdg[store_0].value_inputs()[1].origin
        else {
            panic!("expected a node output");
        };
        let ValueOrigin::Output {
            producer: switch_1_result,
            output: 0,
        } = rvsdg[store_1].value_inputs()[1].origin
        else {
            panic!("expected a node output");
        };

        assert_eq!(rvsdg[switch_0_result].expect_const_u32().value(), 11);
        assert_eq!(rvsdg[switch_1_result].expect_const_u32().value(), 21);

        assert!(!optimizer.optimize_function(&mut module, &mut rvsdg, function));
    }

    #[test]
    fn inline_data_dependent_switch_chain() {
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

        let tag = rvsdg.add_const_u32(body, 2);
        let producer_selector = rvsdg.add_op_case_to_branch_selector(
            body,
            ValueInput::output(TY_U32, tag, 0),
            Int::U32,
            [BranchCase::from(2u32)],
        );
        let producer = rvsdg.add_switch(
            body,
            vec![
                ValueInput::output(TY_PREDICATE, producer_selector, 0),
                ValueInput::argument(TY_U32, 0),
            ],
            vec![ValueOutput::new(TY_U32), ValueOutput::new(TY_U32)],
            None,
        );

        let producer_branch_0 = rvsdg.add_switch_branch(producer);
        let zero = rvsdg.add_const_u32(producer_branch_0, 0);
        let add_zero = rvsdg.add_op_binary(
            producer_branch_0,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 0),
            ValueInput::output(TY_U32, zero, 0),
        );
        let producer_tag = rvsdg.add_const_u32(producer_branch_0, 2);
        rvsdg.reconnect_region_result(
            producer_branch_0,
            0,
            ValueInput::output(TY_U32, add_zero, 0).origin,
        );
        rvsdg.reconnect_region_result(
            producer_branch_0,
            1,
            ValueInput::output(TY_U32, producer_tag, 0).origin,
        );

        let producer_default_branch = rvsdg.add_switch_branch(producer);
        let producer_default_value = rvsdg.add_const_u32(producer_default_branch, 99);
        let producer_default_tag = rvsdg.add_const_u32(producer_default_branch, 0);
        rvsdg.reconnect_region_result(
            producer_default_branch,
            0,
            ValueInput::output(TY_U32, producer_default_value, 0).origin,
        );
        rvsdg.reconnect_region_result(
            producer_default_branch,
            1,
            ValueInput::output(TY_U32, producer_default_tag, 0).origin,
        );

        let consumer_selector = rvsdg.add_op_case_to_branch_selector(
            body,
            ValueInput::output(TY_U32, producer, 1),
            Int::U32,
            [BranchCase::from(2u32)],
        );
        let consumer = rvsdg.add_switch(
            body,
            vec![
                ValueInput::output(TY_PREDICATE, consumer_selector, 0),
                ValueInput::output(TY_U32, producer, 0),
            ],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let consumer_branch_0 = rvsdg.add_switch_branch(consumer);
        let one = rvsdg.add_const_u32(consumer_branch_0, 1);
        let mul_one = rvsdg.add_op_binary(
            consumer_branch_0,
            BinaryOperator::Mul,
            ValueInput::argument(TY_U32, 0),
            ValueInput::output(TY_U32, one, 0),
        );
        rvsdg.reconnect_region_result(
            consumer_branch_0,
            0,
            ValueInput::output(TY_U32, mul_one, 0).origin,
        );

        let consumer_default_branch = rvsdg.add_switch_branch(consumer);
        let consumer_default_value = rvsdg.add_const_u32(consumer_default_branch, 999);
        rvsdg.reconnect_region_result(
            consumer_default_branch,
            0,
            ValueInput::output(TY_U32, consumer_default_value, 0).origin,
        );

        rvsdg.reconnect_region_result(body, 0, ValueInput::output(TY_U32, consumer, 0).origin);

        let mut optimizer = ContextualNodeReducer::new();

        assert!(optimizer.optimize_function(&mut module, &mut rvsdg, function));

        assert!(!rvsdg.is_live_node(producer_selector));
        assert!(!rvsdg.is_live_node(consumer_selector));
        assert!(!rvsdg.is_live_node(producer));
        assert!(!rvsdg.is_live_node(consumer));

        assert_eq!(
            rvsdg[body].value_results()[0],
            ValueInput::argument(TY_U32, 0)
        );

        assert!(!optimizer.optimize_function(&mut module, &mut rvsdg, function));
    }

    #[test]
    fn inline_branch_preserves_state_chain() {
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
                args: vec![],
                ret_ty: None,
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, body) = rvsdg.register_function(&module, function, iter::empty());

        let ptr = rvsdg.add_op_alloca(body, TY_U32);
        let initial = rvsdg.add_const_u32(body, 99);
        let before = rvsdg.add_op_store(
            body,
            ValueInput::output(TY_PTR_U32, ptr, 0),
            ValueInput::output(TY_U32, initial, 0),
            StateOrigin::Argument,
        );

        let selector = rvsdg.add_const_predicate(body, 0);
        let switch = rvsdg.add_switch(
            body,
            vec![
                ValueInput::output(TY_PREDICATE, selector, 0),
                ValueInput::output(TY_PTR_U32, ptr, 0),
            ],
            vec![],
            Some(StateOrigin::Node(before)),
        );

        let retained_branch = rvsdg.add_switch_branch(switch);
        let one = rvsdg.add_const_u32(retained_branch, 1);
        let two = rvsdg.add_const_u32(retained_branch, 2);

        let first_store = rvsdg.add_op_store(
            retained_branch,
            ValueInput::argument(TY_PTR_U32, 0),
            ValueInput::output(TY_U32, one, 0),
            StateOrigin::Argument,
        );

        let second_store = rvsdg.add_op_store(
            retained_branch,
            ValueInput::argument(TY_PTR_U32, 0),
            ValueInput::output(TY_U32, two, 0),
            StateOrigin::Node(first_store),
        );

        let rejected_branch = rvsdg.add_switch_branch(switch);
        let rejected_value = rvsdg.add_const_u32(rejected_branch, 999);
        let rejected_store = rvsdg.add_op_store(
            rejected_branch,
            ValueInput::argument(TY_PTR_U32, 0),
            ValueInput::output(TY_U32, rejected_value, 0),
            StateOrigin::Argument,
        );

        let after = rvsdg.add_op_store(
            body,
            ValueInput::output(TY_PTR_U32, ptr, 0),
            ValueInput::output(TY_U32, initial, 0),
            StateOrigin::Node(switch),
        );

        let mut optimizer = ContextualNodeReducer::new();

        assert!(optimizer.optimize_function(&mut module, &mut rvsdg, function));

        assert!(!rvsdg.is_live_node(first_store));
        assert!(!rvsdg.is_live_node(second_store));
        assert!(!rvsdg.is_live_node(rejected_store));
        assert!(!rvsdg.is_live_node(switch));

        let StateUser::Node(inlined_first) = rvsdg[before].state().unwrap().user else {
            panic!("expected the first retained store after the `before` store");
        };
        let StateUser::Node(inlined_second) = rvsdg[inlined_first].state().unwrap().user else {
            panic!("expected the second retained store after the first retained store");
        };

        assert!(rvsdg[inlined_first].is_op_store());
        assert!(rvsdg[inlined_second].is_op_store());

        assert_eq!(
            rvsdg[inlined_first].state().unwrap().origin,
            StateOrigin::Node(before)
        );
        assert_eq!(
            rvsdg[inlined_first].state().unwrap().user,
            StateUser::Node(inlined_second)
        );
        assert_eq!(
            rvsdg[inlined_second].state().unwrap().origin,
            StateOrigin::Node(inlined_first)
        );
        assert_eq!(
            rvsdg[inlined_second].state().unwrap().user,
            StateUser::Node(after)
        );
        assert_eq!(
            rvsdg[after].state().unwrap().origin,
            StateOrigin::Node(inlined_second)
        );

        assert_eq!(
            rvsdg[inlined_first].value_inputs()[0],
            ValueInput::output(TY_PTR_U32, ptr, 0)
        );
        let ValueOrigin::Output {
            producer,
            output: 0,
        } = rvsdg[inlined_first].value_inputs()[1].origin
        else {
            panic!("expected a node output");
        };
        assert_eq!(rvsdg[producer].expect_const_u32().value(), 1);

        assert_eq!(
            rvsdg[inlined_second].value_inputs()[0],
            ValueInput::output(TY_PTR_U32, ptr, 0)
        );
        let ValueOrigin::Output {
            producer,
            output: 0,
        } = rvsdg[inlined_second].value_inputs()[1].origin
        else {
            panic!("expected a node output");
        };
        assert_eq!(rvsdg[producer].expect_const_u32().value(), 2);
    }

    #[test]
    fn reduces_scalar_constant_conversions() {
        let mut module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let function = Function {
            name: Symbol::from_ref("f"),
            module: Symbol::from_ref("m"),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
                ty: TY_DUMMY,
                args: vec![],
                ret_ty: Some(TY_I32),
            },
        );

        let (_, region) = rvsdg.register_function(&module, function, []);

        let source = rvsdg.add_const_i32(region, -1);
        let unsigned = rvsdg.add_op_convert_to_u32(region, ValueInput::output(TY_I32, source, 0));
        let unsigned_user = rvsdg.add_value_proxy(region, ValueInput::output(TY_U32, unsigned, 0));

        let float = rvsdg.add_op_convert_to_f32(region, ValueInput::output(TY_U32, unsigned, 0));
        let float_user = rvsdg.add_value_proxy(region, ValueInput::output(TY_F32, float, 0));

        let boolean = rvsdg.add_op_convert_to_bool(region, ValueInput::output(TY_F32, float, 0));
        let boolean_user = rvsdg.add_value_proxy(region, ValueInput::output(TY_BOOL, boolean, 0));

        let signed = rvsdg.add_op_convert_to_i32(region, ValueInput::output(TY_BOOL, boolean, 0));

        rvsdg.reconnect_region_result(region, 0, ValueInput::output(TY_I32, signed, 0).origin);

        let mut optimizer = ContextualNodeReducer::new();

        assert!(optimizer.optimize_function(&mut module, &mut rvsdg, function));

        assert!(!rvsdg.is_live_node(unsigned));
        assert!(!rvsdg.is_live_node(float));
        assert!(!rvsdg.is_live_node(boolean));
        assert!(!rvsdg.is_live_node(signed));

        let ValueOrigin::Output {
            producer,
            output: 0,
        } = rvsdg[unsigned_user].value_inputs()[0].origin
        else {
            panic!("expected an node output");
        };
        assert_eq!(rvsdg[producer].expect_const_u32().value(), u32::MAX);

        let ValueOrigin::Output {
            producer,
            output: 0,
        } = rvsdg[float_user].value_inputs()[0].origin
        else {
            panic!("expected a float constant");
        };
        assert_eq!(rvsdg[producer].expect_const_f32().value(), 4_294_967_296.0);

        let ValueOrigin::Output {
            producer,
            output: 0,
        } = rvsdg[boolean_user].value_inputs()[0].origin
        else {
            panic!("expected a node output");
        };
        assert!(rvsdg[producer].expect_const_bool().value());

        let ValueOrigin::Output {
            producer,
            output: 0,
        } = rvsdg[region].value_results()[0].origin
        else {
            panic!("expected a node output");
        };
        assert_eq!(rvsdg[producer].expect_const_i32().value(), 1);

        assert!(!optimizer.optimize_function(&mut module, &mut rvsdg, function));
    }

    #[test]
    fn division_by_zero_is_not_reduced() {
        let mut module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let function = Function {
            name: Symbol::from_ref("f"),
            module: Symbol::from_ref("m"),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
                ty: TY_DUMMY,
                args: vec![],
                ret_ty: None,
            },
        );

        let (_, region) = rvsdg.register_function(&module, function, []);

        let lhs = rvsdg.add_const_u32(region, 1);
        let rhs = rvsdg.add_const_u32(region, 0);
        let node = rvsdg.add_op_binary(
            region,
            BinaryOperator::Div,
            ValueInput::output(TY_U32, lhs, 0),
            ValueInput::output(TY_U32, rhs, 0),
        );

        let user = rvsdg.add_value_proxy(region, ValueInput::output(TY_U32, node, 0));

        let mut optimizer = ContextualNodeReducer::new();

        assert!(!optimizer.optimize_function(&mut module, &mut rvsdg, function));

        assert!(rvsdg.is_live_node(node));
        assert_eq!(
            rvsdg[user].value_inputs()[0].origin,
            ValueOrigin::Output {
                producer: node,
                output: 0,
            }
        );
    }

    #[test]
    fn can_fold_invariant_loop_value() {
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
                args: vec![],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, body) = rvsdg.register_function(&module, function, iter::empty());

        let initial = rvsdg.add_const_u32(body, 7);
        let (loop_node, loop_body) = rvsdg.add_loop(
            body,
            vec![
                ValueInput::output(TY_U32, initial, 0),
                ValueInput::output(TY_U32, initial, 0),
            ],
            None,
        );

        let three = rvsdg.add_const_u32(loop_body, 3);
        let sum = rvsdg.add_op_binary(
            loop_body,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 0),
            ValueInput::output(TY_U32, three, 0),
        );

        let reentry = rvsdg.add_const_bool(loop_body, true);

        rvsdg.reconnect_region_result(loop_body, 0, ValueInput::output(TY_BOOL, reentry, 0).origin);
        rvsdg.reconnect_region_result(loop_body, 1, ValueOrigin::Argument(0));
        rvsdg.reconnect_region_result(loop_body, 2, ValueInput::output(TY_U32, sum, 0).origin);

        rvsdg.reconnect_region_result(body, 0, ValueInput::output(TY_U32, loop_node, 1).origin);

        let mut optimizer = ContextualNodeReducer::new();

        assert!(optimizer.optimize_function(&mut module, &mut rvsdg, function));

        assert!(!rvsdg.is_live_node(sum));
        assert_eq!(
            rvsdg[loop_body].value_results()[1],
            ValueInput::argument(TY_U32, 0)
        );

        let ValueOrigin::Output {
            producer: folded_constant,
            output: 0,
        } = rvsdg[loop_body].value_results()[2].origin
        else {
            panic!("expected a node output");
        };

        assert_eq!(rvsdg[folded_constant].expect_const_u32().value(), 10);
    }

    #[test]
    fn cannot_fold_loop_variant_value() {
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
                args: vec![],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, body) = rvsdg.register_function(&module, function, iter::empty());

        let initial = rvsdg.add_const_u32(body, 7);
        let (loop_node, loop_body) =
            rvsdg.add_loop(body, vec![ValueInput::output(TY_U32, initial, 0)], None);

        let three = rvsdg.add_const_u32(loop_body, 3);
        let sum = rvsdg.add_op_binary(
            loop_body,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 0),
            ValueInput::output(TY_U32, three, 0),
        );

        let reentry = rvsdg.add_const_bool(loop_body, true);

        rvsdg.reconnect_region_result(loop_body, 0, ValueInput::output(TY_BOOL, reentry, 0).origin);
        rvsdg.reconnect_region_result(loop_body, 1, ValueInput::output(TY_U32, sum, 0).origin);

        rvsdg.reconnect_region_result(body, 0, ValueInput::output(TY_U32, loop_node, 0).origin);

        let mut optimizer = ContextualNodeReducer::new();

        assert!(!optimizer.optimize_function(&mut module, &mut rvsdg, function));
    }

    #[test]
    fn applies_simple_conditional_scalar_reduction() {
        let mut module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());
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
                ret_ty: Some(TY_BOOL),
            },
        );

        let (_, body) = rvsdg.register_function(&module, function, []);

        let condition = ValueInput::argument(TY_BOOL, 0);
        let selector = rvsdg.add_op_bool_to_branch_selector(body, condition);
        let switch = rvsdg.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, selector, 0)],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch);
        let value_0 = rvsdg.add_const_u32(branch_0, 1);
        rvsdg.reconnect_region_result(branch_0, 0, ValueInput::output(TY_U32, value_0, 0).origin);

        let branch_1 = rvsdg.add_switch_branch(switch);
        let value_1 = rvsdg.add_const_u32(branch_1, 0);
        rvsdg.reconnect_region_result(branch_1, 0, ValueInput::output(TY_U32, value_1, 0).origin);

        let one = rvsdg.add_const_u32(body, 1);

        let target = rvsdg.add_op_binary(
            body,
            BinaryOperator::Eq,
            ValueInput::output(TY_U32, switch, 0),
            ValueInput::output(TY_U32, one, 0),
        );

        rvsdg.reconnect_region_result(body, 0, ValueInput::output(TY_BOOL, target, 0).origin);

        let mut optimizer = ContextualNodeReducer::new();

        assert!(optimizer.optimize_function(&mut module, &mut rvsdg, function));

        assert!(!rvsdg.is_live_node(target));

        assert_eq!(rvsdg[body].value_results()[0], condition);
        assert!(rvsdg[switch].value_outputs()[0].users.is_empty());
    }

    #[test]
    fn applies_simple_conditional_scalar_reduction_to_complement() {
        let mut module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());
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
                ret_ty: Some(TY_BOOL),
            },
        );

        let (_, body) = rvsdg.register_function(&module, function, []);

        let condition = ValueInput::argument(TY_BOOL, 0);
        let selector = rvsdg.add_op_bool_to_branch_selector(body, condition);
        let switch = rvsdg.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, selector, 0)],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch);
        let value_0 = rvsdg.add_const_u32(branch_0, 0);
        rvsdg.reconnect_region_result(branch_0, 0, ValueInput::output(TY_U32, value_0, 0).origin);

        let branch_1 = rvsdg.add_switch_branch(switch);
        let value_1 = rvsdg.add_const_u32(branch_1, 1);
        rvsdg.reconnect_region_result(branch_1, 0, ValueInput::output(TY_U32, value_1, 0).origin);

        let one = rvsdg.add_const_u32(body, 1);

        let target = rvsdg.add_op_binary(
            body,
            BinaryOperator::Eq,
            ValueInput::output(TY_U32, switch, 0),
            ValueInput::output(TY_U32, one, 0),
        );

        rvsdg.reconnect_region_result(body, 0, ValueInput::output(TY_BOOL, target, 0).origin);

        let mut optimizer = ContextualNodeReducer::new();

        assert!(optimizer.optimize_function(&mut module, &mut rvsdg, function));

        assert!(!rvsdg.is_live_node(target));

        let ValueOrigin::Output {
            producer: complement,
            output: 0,
        } = rvsdg[body].value_results()[0].origin
        else {
            panic!("expected a complement");
        };

        assert_eq!(rvsdg[complement].region(), body);
        assert_eq!(
            rvsdg[complement].expect_op_unary().operator(),
            UnaryOperator::Not
        );
        assert_eq!(rvsdg[complement].value_inputs(), &[condition]);
        assert_eq!(
            rvsdg[complement].value_outputs()[0].users,
            thin_set![ValueUser::Result(0)]
        );
        assert!(rvsdg[switch].value_outputs()[0].users.is_empty());
    }

    #[test]
    fn simple_scalar_reduction_takes_precedence_over_conditional_scalar_reduction() {
        let mut module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());
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
                ret_ty: Some(TY_BOOL),
            },
        );

        let (_, body) = rvsdg.register_function(&module, function, []);

        let selector = rvsdg.add_op_bool_to_branch_selector(body, ValueInput::argument(TY_BOOL, 0));
        let switch = rvsdg.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, selector, 0)],
            vec![ValueOutput::new(TY_BOOL)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch);
        let value_0 = rvsdg.add_const_bool(branch_0, false);
        rvsdg.reconnect_region_result(branch_0, 0, ValueInput::output(TY_BOOL, value_0, 0).origin);

        let branch_1 = rvsdg.add_switch_branch(switch);
        let value_1 = rvsdg.add_const_bool(branch_1, true);
        rvsdg.reconnect_region_result(branch_1, 0, ValueInput::output(TY_BOOL, value_1, 0).origin);

        let const_false = rvsdg.add_const_bool(body, false);
        let target = rvsdg.add_op_binary(
            body,
            BinaryOperator::Or,
            ValueInput::output(TY_BOOL, switch, 0),
            ValueInput::output(TY_BOOL, const_false, 0),
        );

        rvsdg.reconnect_region_result(body, 0, ValueInput::output(TY_BOOL, target, 0).origin);

        let mut optimizer = ContextualNodeReducer::new();

        assert!(optimizer.optimize_function(&mut module, &mut rvsdg, function));

        assert!(!rvsdg.is_live_node(target));

        assert_eq!(
            rvsdg[body].value_results()[0],
            ValueInput::output(TY_BOOL, switch, 0)
        );

        assert!(!optimizer.optimize_function(&mut module, &mut rvsdg, function));
    }

    #[test]
    fn conditional_scalar_reduction_tries_multiple_producer_switch_nodes() {
        let mut module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let function = Function {
            name: Symbol::from_ref("test"),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
                ty: TY_DUMMY,
                args: vec![
                    FnArg {
                        ty: TY_BOOL,
                        shader_io_binding: None,
                    },
                    FnArg {
                        ty: TY_BOOL,
                        shader_io_binding: None,
                    },
                ],
                ret_ty: Some(TY_BOOL),
            },
        );

        let (_, body) = rvsdg.register_function(&module, function, []);

        let switch_0_selector =
            rvsdg.add_op_bool_to_branch_selector(body, ValueInput::argument(TY_BOOL, 0));
        let switch_0 = rvsdg.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, switch_0_selector, 0)],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let switch_0_branch_0 = rvsdg.add_switch_branch(switch_0);
        let switch_0_tag_0 = rvsdg.add_const_u32(switch_0_branch_0, 1);
        rvsdg.reconnect_region_result(
            switch_0_branch_0,
            0,
            ValueInput::output(TY_U32, switch_0_tag_0, 0).origin,
        );

        let switch_0_branch_1 = rvsdg.add_switch_branch(switch_0);
        let switch_0_tag_1 = rvsdg.add_const_u32(switch_0_branch_1, 2);
        rvsdg.reconnect_region_result(
            switch_0_branch_1,
            0,
            ValueInput::output(TY_U32, switch_0_tag_1, 0).origin,
        );

        let switch_1_condition = ValueInput::argument(TY_BOOL, 1);
        let switch_1_selector = rvsdg.add_op_bool_to_branch_selector(body, switch_1_condition);
        let switch_1 = rvsdg.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, switch_1_selector, 0)],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let switch_1_branch_0 = rvsdg.add_switch_branch(switch_1);
        let switch_1_tag_0 = rvsdg.add_const_u32(switch_1_branch_0, 3);
        rvsdg.reconnect_region_result(
            switch_1_branch_0,
            0,
            ValueInput::output(TY_U32, switch_1_tag_0, 0).origin,
        );

        let switch_1_branch_1 = rvsdg.add_switch_branch(switch_1);
        let switch_1_tag_1 = rvsdg.add_const_u32(switch_1_branch_1, 0);
        rvsdg.reconnect_region_result(
            switch_1_branch_1,
            0,
            ValueInput::output(TY_U32, switch_1_tag_1, 0).origin,
        );

        let target = rvsdg.add_op_binary(
            body,
            BinaryOperator::Lt,
            ValueInput::output(TY_U32, switch_0, 0),
            ValueInput::output(TY_U32, switch_1, 0),
        );

        let user = rvsdg.add_value_proxy(body, ValueInput::output(TY_BOOL, target, 0));

        rvsdg.reconnect_region_result(body, 0, ValueInput::output(TY_BOOL, target, 0).origin);

        let mut optimizer = ContextualNodeReducer::new();

        assert!(optimizer.optimize_function(&mut module, &mut rvsdg, function));

        assert!(!rvsdg.is_live_node(target));

        assert_eq!(rvsdg[body].value_results()[0], switch_1_condition);
        assert_eq!(rvsdg[user].value_inputs()[0], switch_1_condition);

        assert_eq!(
            rvsdg[body].value_arguments()[1].users,
            thin_set![
                ValueUser::Input {
                    consumer: switch_1_selector,
                    input: 0
                },
                ValueUser::Input {
                    consumer: user,
                    input: 0
                },
                ValueUser::Result(0)
            ]
        );

        assert!(!optimizer.optimize_function(&mut module, &mut rvsdg, function));
    }

    #[test]
    fn applies_multiple_dependent_conditional_scalar_reductions_in_single_pass() {
        let mut module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());
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
                ret_ty: Some(TY_BOOL),
            },
        );

        let (_, body) = rvsdg.register_function(&module, function, []);

        let condition = ValueInput::argument(TY_BOOL, 0);
        let selector = rvsdg.add_op_bool_to_branch_selector(body, condition);
        let producer = rvsdg.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, selector, 0)],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(producer);
        let value_0 = rvsdg.add_const_u32(branch_0, 1);
        rvsdg.reconnect_region_result(branch_0, 0, ValueInput::output(TY_U32, value_0, 0).origin);

        let branch_1 = rvsdg.add_switch_branch(producer);
        let value_1 = rvsdg.add_const_u32(branch_1, 0);
        rvsdg.reconnect_region_result(branch_1, 0, ValueInput::output(TY_U32, value_1, 0).origin);

        let one = rvsdg.add_const_u32(body, 1);

        let eq_one = rvsdg.add_op_binary(
            body,
            BinaryOperator::Eq,
            ValueInput::output(TY_U32, producer, 0),
            ValueInput::output(TY_U32, one, 0),
        );
        let invert = rvsdg.add_op_unary(
            body,
            UnaryOperator::Not,
            ValueInput::output(TY_BOOL, eq_one, 0),
        );

        let eq_one_user = rvsdg.add_value_proxy(body, ValueInput::output(TY_BOOL, eq_one, 0));
        let invert_user = rvsdg.add_value_proxy(body, ValueInput::output(TY_BOOL, invert, 0));

        rvsdg.reconnect_region_result(body, 0, ValueInput::output(TY_BOOL, invert, 0).origin);

        let mut optimizer = ContextualNodeReducer::new();

        assert!(optimizer.optimize_function(&mut module, &mut rvsdg, function));

        assert!(!rvsdg.is_live_node(eq_one));
        assert!(!rvsdg.is_live_node(invert));

        assert_eq!(rvsdg[eq_one_user].value_inputs()[0], condition);

        let replacement = rvsdg[body].value_results()[0];
        let ValueOrigin::Output {
            producer: fresh_invert,
            output: 0,
        } = replacement.origin
        else {
            panic!("expected a fresh complement");
        };

        assert_eq!(rvsdg[fresh_invert].region(), body);
        assert_eq!(
            rvsdg[fresh_invert].expect_op_unary().operator(),
            UnaryOperator::Not
        );
        assert_eq!(rvsdg[fresh_invert].value_inputs(), &[condition]);
        assert_eq!(rvsdg[invert_user].value_inputs()[0], replacement);
        assert_eq!(
            rvsdg[fresh_invert].value_outputs()[0].users,
            thin_set![
                ValueUser::Input {
                    consumer: invert_user,
                    input: 0
                },
                ValueUser::Result(0)
            ]
        );

        assert_eq!(
            rvsdg[body].value_arguments()[0].users,
            thin_set![
                ValueUser::Input {
                    consumer: selector,
                    input: 0
                },
                ValueUser::Input {
                    consumer: eq_one_user,
                    input: 0
                },
                ValueUser::Input {
                    consumer: fresh_invert,
                    input: 0
                }
            ]
        );

        assert!(!optimizer.optimize_function(&mut module, &mut rvsdg, function));
    }
}
