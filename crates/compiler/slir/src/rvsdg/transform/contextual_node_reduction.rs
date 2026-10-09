//! Uses [cva] to simplify away simple nodes and switch branches.
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

use rustc_hash::FxHashMap;

use crate::rvsdg::analyse::cva::{
    AbstractValue, AssumptionResult, ContextId, ContextualValueAnalysis, ValueKey,
};
use crate::rvsdg::analyse::region_stratification::RegionStratifier;
use crate::rvsdg::analyse::scalar_constant::ScalarConstant;
use crate::rvsdg::transform::region_replication::inline_switch_branch;
use crate::rvsdg::transform::switch_branch_pruning::retain_switch_branches;
use crate::rvsdg::{
    Connectivity, Node, NodeKind, Region, Rvsdg, SimpleNode, ValueInput, ValueOrigin,
};
use crate::{BinaryOperator, Function, Module};

#[derive(Clone, Copy, Debug, PartialEq)]
enum ReplacementValue {
    Constant(ScalarConstant),
    Variable(ValueInput),
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

/// Uses [cva] to simplify away simple nodes and switch branches.
///
/// See the [module-level documentation](crate::rvsdg::transform::contextual_node_reduction) for
/// details.
#[derive(Default)]
pub struct ContextualNodeReducer {
    reduce_to_value_jobs: Vec<ReduceToValueJob>,
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
        changed |= self.apply_reduce_switch_jobs(module, rvsdg);

        changed
    }

    fn collect_jobs(&mut self, rvsdg: &Rvsdg, function: Function) {
        debug_assert!(self.reduce_to_value_jobs.is_empty() && self.reduce_switch_jobs.is_empty());

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
                        && retained.len() < switch.branches().len()
                        && (retained.len() == 1 || has_case_selector(rvsdg, node))
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
    use NodeKind::*;
    use SimpleNode::*;

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

    let replacement = if let Some(constant) = value.to_scalar_constant() {
        constant.into()
    } else {
        if value.is_bottom() {
            return None;
        }

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
        }?
    };

    Some(ReduceToValueJob {
        target: node,
        replacement,
    })
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
    };
    use crate::ty::{Int, TY_BOOL, TY_DUMMY, TY_F32, TY_I32, TY_PREDICATE, TY_PTR_U32, TY_U32};
    use crate::{BinaryOperator, BranchCase, FnArg, FnSig, Function, Module, Symbol};

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
}
