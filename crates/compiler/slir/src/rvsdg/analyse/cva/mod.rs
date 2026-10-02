//! Contextual value analysis.
//!
//! "Context" in this instance refers to control-flow context. When execution enters a switch branch
//! or a loop-region, then this implies constraints on the state of the program for this to have
//! happened. For example, if execution has entered the second branch of a switch node, then this
//! implies constraints on the switch node's branch-selector value, namely that it was exactly the
//! value `1`. This implication can be propagated further "up the value-flow", for example, to the
//! [OpBoolToBranchSelector] node that produced the branch-selector value, implying that the
//! [OpBoolToBranchSelector]'s input value was exactly the value `false`.
//!
//! Constraints do not only follow from control-flow decisions. For example, if a value flows
//! from a constant value, then the value-flow is clearly constrained to exactly that constant
//! value. Constraints also do not necessarily imply a single concrete value. For example, an
//! [OpBinary] node may be used to express `LHS < RHS`. If result of the comparison is constrained
//! to a `true` constant and the `RHS` input value is constrained to the constant `u32` value `10`,
//! then the implication is that the `LHS` input value is constrained to the interval `0..10`. We
//! use the [AbstractValue] type to represent the set of values that may be still be taken under
//! the constraints that we've derived for a value. If we've not been able to derive or uphold any
//! constraints on an abstract-value, we say that the value is a "top" value (see
//! [AbstractValue::is_top]). If the constraints we derive for a value become contradictory (for
//! example, we've determined that a single value must simultaneously be `0` and `1`), we say the
//! value is a "bottom" value (see [AbstractValue::is_bottom]). We loosely borrow the terms "top"
//! and "bottom" from lattice theory.
//!
//! Note that "bottom" values do not necessarily imply an invalid RVSDG; such values may naturally
//! occur in contexts that assume certain control-flow decisions. Assuming that a downstream switch
//! node enters branch `0` may result in constraints on the output of an upstream switch node. When
//! intersecting those constraints with the constraints on each of the output's corresponding
//! branch-region results, we may find such "bottom" values. This has useful implications: for any
//! branch where the branch-result is a "bottom" value, the upstream switch node must not have been
//! able to select that branch; under the assumptions we've made for the control-flow context, the
//! upstream switch node could only "feasibly" have selected branches that do not lead to
//! contradictions. The [correlated_switch_simplification] transform leverages implications like
//! these to simplify the control-flow.
//!
//! Contextual value analysis is session-based. Such sessions are represented by instances of the
//! [ContextualValueAnalysis] type. Over the course of a session, it accumulates cached information
//! about the queries that were made. This information only remains valid as long as the RVSDG
//! remains unchanged. To enforce this, a [ContextualValueAnalysis] holds onto a borrow of the RVSDG
//! it analyzes. If you wish to use the results of the analysis to modify the RVSDG, then the
//! recommended approach is to split that into a "planning" and "modification" phase: you use a
//! single [ContextualValueAnalysis] session to plan as many modifications as possible (contextual
//! value analysis is relatively expensive, the cached information should be reused as much as
//! possible); you then drop the session and apply the modifications to the RVSDG now that the
//! borrow has been released.
//!
//! [ContextualValueAnalysis] sessions are tied to one specific function in the RVSDG. The analysis
//! currently does not support cross-function analysis; all function calls are expected to have
//! been inlined exhaustively. The [ContextualValueAnalysis::root_context] represents a control-flow
//! context where no control-flow decisions have been assumed. Note that this does not mean that no
//! value-constraints can be derived in the root context: constraints can be derived from constant
//! values and their downstream value-flow. Switch branch selection decisions can be assumed by
//! calling [ContextualValueAnalysis::assume_branch]. This creates a "child" context in which
//! additional constraints on values can be derived. The assumed branch implies a constant branch
//! selector value. This branch-selector constraint is, in turn, propagated to other values in the
//! graph through "saturation".
//!
//! To derive the constraints on a specific value, call [ContextualValueAnalysis::evaluate_value].
//! This method expects a specific context as one of its arguments. As outlined above, more specific
//! contexts may produce tighter constraints. The constraints are represented by an [AbstractValue].
//! Abstract-values are typed and represent the set of values that are compatible with the value's
//! constraints. For details, refer to the documentation for [AbstractValue].
//!
//! [OpBoolToBranchSelector]: crate::rvsdg::OpBoolToBranchSelector
//! [OpBinary]: crate::rvsdg::OpBinary
//! [correlated_switch_simplification]: crate::rvsdg::transform::correlated_switch_simplification

mod context;
pub mod domain;
mod engine;
pub(crate) mod routing;

use std::sync::atomic::{AtomicU64, Ordering};

use rustc_hash::FxHashMap;

use self::context::{ConstraintFrame, Context, Environment, constraint_frame_for_assumed_branch};
pub use self::domain::AbstractValue;
use self::engine::Engine;
pub use self::routing::ValueKey;
use self::routing::contains_region;
use crate::Function;
use crate::rvsdg::{Connectivity, Node, NodeKind, Region, Rvsdg};

static NEXT_SESSION: AtomicU64 = AtomicU64::new(1);

/// Identifies a control-flow context for a [ContextualValueAnalysis] session.
///
/// Context IDs cannot be used between different sessions.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ContextId {
    session: u64,
    index: usize,
}

/// The outcome of constructing a context under a control-flow assumption.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AssumptionResult {
    /// A context in which the implications of the assumption affect the constraints we derive on
    /// values.
    Context(ContextId),

    /// The assumption contradicts the pre-existing constraints on the parent context.
    Contradictory,

    /// The analysis could not establish a context or prove the assumption contradictory.
    ///
    /// This occurs when the requested region is outside the parent context's scope.
    Unknown,
}

impl AssumptionResult {
    /// Returns the context established by the assumption.
    ///
    /// # Panics
    ///
    /// Panics if the result is [Self::Contradictory] or [Self::Unknown].
    #[track_caller]
    pub fn expect_context(self) -> ContextId {
        if let Self::Context(context) = self {
            context
        } else {
            panic!("expected context, got {self:?}")
        }
    }
}

fn branches_matching_selector(selector: &AbstractValue, count: usize) -> Vec<usize> {
    let AbstractValue::Predicate(selector) = selector else {
        unreachable!("switch selector is a predicate")
    };

    (0..count)
        .filter(|&branch| selector.contains(branch as u32))
        .collect()
}

/// The completed result of constructing an environment under one branch assumption.
enum AssumedBranchData {
    Contradictory,
    Context {
        constraint_frame: ConstraintFrame,
        cache: QueryCache,
    },
}

#[derive(Default)]
struct QueryCache {
    value_constraints: FxHashMap<ValueKey, AbstractValue>,
    assumed_branch_data: FxHashMap<(Node, usize), AssumedBranchData>,
}

impl QueryCache {
    fn clear(&mut self) {
        self.value_constraints.clear();
        self.assumed_branch_data.clear();
    }
}

/// Helper type for looking up and reusing "child" contexts that have already been recorded earlier
/// for a [ContextualValueAnalysis].
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum ChildKey {
    Branch(ContextId, Node, usize),
    Loop(ContextId, Node),
}

/// A contextual-value-analysis session.
///
/// See the module-level documentation for more information.
pub struct ContextualValueAnalysis<'a> {
    graph: &'a Rvsdg,
    work_budget: usize,
    session: u64,
    contexts: Vec<Context>,
    caches: Vec<QueryCache>,
    child_contexts: FxHashMap<ChildKey, ContextId>,
}

impl<'a> ContextualValueAnalysis<'a> {
    /// Creates a new contextual-value-analysis for the given `function`.
    ///
    /// Assumes all function calls made by the function have been exhaustively inlined.
    ///
    /// # Panics
    ///
    /// Panics if the `function` is not registered with the `rvsdg`.
    pub fn new(rvsdg: &'a Rvsdg, function: Function) -> Self {
        Self::with_work_budget(rvsdg, function, 4096)
    }

    /// Creates a new contextual-value-analysis with the given amount of work available to each
    /// query.
    ///
    /// Assumes all function calls made by the function have been exhaustively inlined.
    ///
    /// # Panics
    ///
    /// Panics if the `function` is not registered with the `rvsdg`.
    pub fn with_work_budget(rvsdg: &'a Rvsdg, function: Function, work_budget: usize) -> Self {
        let function_node = rvsdg
            .get_function_node(function)
            .expect("function not registered");
        let body = rvsdg[function_node].expect_function().body_region();

        Self {
            graph: rvsdg,
            work_budget,
            session: NEXT_SESSION.fetch_add(1, Ordering::Relaxed),
            contexts: vec![Context {
                parent: None,
                scope: body,
                branch_assumption: None,
                constraint_frame: ConstraintFrame::default(),
            }],
            caches: vec![QueryCache::default()],
            child_contexts: FxHashMap::default(),
        }
    }

    /// Returns the context ID for the function's root context, without any assumptions on
    /// control-flow decisions.
    pub fn root_context(&self) -> ContextId {
        ContextId {
            session: self.session,
            index: 0,
        }
    }

    /// Evaluates the constraints that can be derived for the given `value` under the assumptions
    /// made for the `context`.
    ///
    /// This does not necessarily produce the tightest possible constraints on the value, it may
    /// produce a conservative over-approximation. However, if a concrete instance of the value is
    /// not contained within the abstract value this returns, then under the assumptions made for
    /// the context, that concrete instance can be considered impossible.
    ///
    /// This may return a "bottom" value if the assumptions lead to a contradiction.
    pub fn evaluate_value(&mut self, value: ValueKey, context: ContextId) -> AbstractValue {
        self.validate_context_id(context);

        let env = Environment::init(&self.contexts, context);

        Engine::new(self.graph, self.work_budget).evaluate_value(
            value,
            &env,
            &mut self.caches[context.index],
        )
    }

    /// Assumes that control-flow has selected the given `branch` for the `switch` node and
    /// evaluates values for the branch result associated with the switch `output`.
    ///
    /// This is a convenience method for first assuming the switch branch with [assume_branch],
    /// then evaluating the origin value for the branch result with [evaluate_value].
    pub fn evaluate_branch_result(
        &mut self,
        switch: Node,
        output: u32,
        branch: usize,
        context: ContextId,
    ) -> AbstractValue {
        let ty = self.graph[switch].value_outputs()[output as usize].ty;

        match self.assume_branch(context, switch, branch) {
            AssumptionResult::Context(child) => {
                let region = self.graph[switch].expect_switch().branches()[branch];
                let key = ValueKey::new(
                    region,
                    self.graph[region].value_results()[output as usize].origin,
                );

                self.evaluate_value(key, child)
            }
            AssumptionResult::Contradictory => AbstractValue::bottom(ty),
            AssumptionResult::Unknown => AbstractValue::top(ty),
        }
    }

    /// Returns a context that assumes `branch` is selected for `switch`.
    ///
    /// This may fail to produce a context, see [AssumptionResult]. If a context is created, then
    /// queries made against the context will observe the constraint implied by the branch
    /// assumption on the switch's branch-selector input value and its related value-flow.
    ///
    /// When `switch` is nested inside other switches, the returned context also implicitly assumes
    /// that the enclosing branches must have been selected for the control-flow to reach the
    /// `switch` node.
    ///
    /// # Panics
    ///
    /// Panics if `context` belongs to a different analysis session, `switch` is not a switch node,
    /// or `branch` is not a branch of `switch`.
    pub fn assume_branch(
        &mut self,
        context: ContextId,
        switch: Node,
        branch: usize,
    ) -> AssumptionResult {
        self.validate_context_id(context);

        let data = self.graph[switch].expect_switch();
        assert!(branch < data.branches().len(), "branch index out of bounds");

        let context = match self.refine_context(context, self.graph[switch].region()) {
            AssumptionResult::Context(context) => context,
            result => return result,
        };

        self.apply_branch_assumption(context, switch, branch)
    }

    fn refine_context(&mut self, mut context: ContextId, mut region: Region) -> AssumptionResult {
        self.validate_context_id(context);

        let scope = self.contexts[context.index].scope;
        let mut region_trace = Vec::new();

        while region != scope {
            if region == self.graph.global_region() {
                return AssumptionResult::Unknown;
            }

            region_trace.push(region);
            region = self.graph[self.graph[region].owner()].region();
        }

        for region in region_trace.into_iter().rev() {
            let owner = self.graph[region].owner();

            let result = match self.graph[owner].kind() {
                NodeKind::Switch(switch) => {
                    let branch = switch.branches().iter().position(|&r| r == region).unwrap();
                    let env = Environment::init(&self.contexts, context);

                    match env
                        .branch_assumptions
                        .iter()
                        .find(|&&(node, _)| node == owner)
                    {
                        Some(&(_, selected)) if selected == branch => {
                            AssumptionResult::Context(context)
                        }
                        Some(_) => AssumptionResult::Contradictory,
                        None => self.apply_branch_assumption(context, owner, branch),
                    }
                }
                NodeKind::Loop(_) => self.apply_loop_entry(context, owner),
                _ => return AssumptionResult::Unknown,
            };

            match result {
                AssumptionResult::Context(child) => context = child,
                result => return result,
            }
        }

        AssumptionResult::Context(context)
    }

    fn apply_branch_assumption(
        &mut self,
        context: ContextId,
        switch: Node,
        branch: usize,
    ) -> AssumptionResult {
        let key = ChildKey::Branch(context, switch, branch);

        if let Some(&id) = self.child_contexts.get(&key) {
            return AssumptionResult::Context(id);
        }

        let env = Environment::init(&self.contexts, context);

        let mut engine = Engine::new(self.graph, self.work_budget);

        if !env.is_in_context(self.graph, self.graph[switch].region()) {
            return AssumptionResult::Unknown;
        }

        let Some(constraint_frame) =
            constraint_frame_for_assumed_branch(&mut engine, &env, switch, branch)
        else {
            return AssumptionResult::Contradictory;
        };

        let scope = env.scope;
        let id = self.create_child_context(key, context, scope, constraint_frame);

        AssumptionResult::Context(id)
    }

    fn apply_loop_entry(&mut self, context: ContextId, node: Node) -> AssumptionResult {
        let data = self.graph[node].expect_loop();

        let scope = data.loop_region();
        let env = Environment::init(&self.contexts, context);

        if contains_region(self.graph, scope, env.scope)
            || !env.is_in_context(self.graph, self.graph[node].region())
        {
            return AssumptionResult::Unknown;
        }

        let key = ChildKey::Loop(context, node);

        let id = if let Some(&id) = self.child_contexts.get(&key) {
            id
        } else {
            self.create_child_context(key, context, scope, ConstraintFrame::default())
        };

        AssumptionResult::Context(id)
    }

    fn validate_context_id(&self, context: ContextId) {
        assert_eq!(
            context.session, self.session,
            "context belongs to a different analysis session"
        );
        assert!(context.index < self.contexts.len());
    }

    fn create_child_context(
        &mut self,
        child_key: ChildKey,
        parent: ContextId,
        scope: Region,
        constraint_frame: ConstraintFrame,
    ) -> ContextId {
        let id = ContextId {
            session: self.session,
            index: self.contexts.len(),
        };

        self.contexts.push(Context {
            parent: Some(parent),
            scope,
            branch_assumption: match child_key {
                ChildKey::Branch(_, switch, branch) => Some((switch, branch)),
                ChildKey::Loop(_, _) => None,
            },
            constraint_frame,
        });
        self.caches.push(QueryCache::default());
        self.child_contexts.insert(child_key, id);

        id
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::domain::{AbstractPredicate, AbstractU32};
    use super::{AbstractValue, AssumptionResult, ContextualValueAnalysis, ValueKey};
    use crate::rvsdg::analyse::scalar_constant::ScalarConstant;
    use crate::rvsdg::{Rvsdg, ValueInput, ValueOrigin, ValueOutput};
    use crate::ty::{Int, TY_BOOL, TY_DUMMY, TY_I32, TY_PREDICATE, TY_U32};
    use crate::{
        BinaryOperator, BranchCase, FnArg, FnSig, Function, Module, Symbol, UnaryOperator,
    };

    #[test]
    fn recursive_exact_unsigned_evaluation() {
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

        let mut graph = Rvsdg::new(module.ty.clone());
        let (_, body) = graph.register_function(&module, function, iter::empty());

        let a = graph.add_const_u32(body, 1);
        let b = graph.add_const_u32(body, 2);

        let c = graph.add_const_u32(body, 4);

        let sum = graph.add_op_binary(
            body,
            BinaryOperator::Add,
            ValueInput::output(TY_U32, a, 0),
            ValueInput::output(TY_U32, b, 0),
        );

        let product = graph.add_op_binary(
            body,
            BinaryOperator::Mul,
            ValueInput::output(TY_U32, sum, 0),
            ValueInput::output(TY_U32, c, 0),
        );

        let mut analysis = ContextualValueAnalysis::new(&graph, function);
        let root_context = analysis.root_context();

        assert_eq!(
            analysis
                .evaluate_value(
                    ValueKey::new(
                        body,
                        ValueOrigin::Output {
                            producer: product,
                            output: 0,
                        }
                    ),
                    root_context
                )
                .to_scalar_constant(),
            Some(ScalarConstant::U32(12))
        );
    }

    #[test]
    fn recursive_exact_signed_evaluation() {
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

        let mut graph = Rvsdg::new(module.ty.clone());
        let (_, body) = graph.register_function(&module, function, iter::empty());

        let a = graph.add_const_i32(body, -7);
        let b = graph.add_const_i32(body, 3);

        let sum = graph.add_op_binary(
            body,
            BinaryOperator::Add,
            ValueInput::output(TY_I32, a, 0),
            ValueInput::output(TY_I32, b, 0),
        );

        let negated =
            graph.add_op_unary(body, UnaryOperator::Neg, ValueInput::output(TY_I32, sum, 0));

        let greater = graph.add_op_binary(
            body,
            BinaryOperator::Gt,
            ValueInput::output(TY_I32, negated, 0),
            ValueInput::output(TY_I32, b, 0),
        );

        let mut analysis = ContextualValueAnalysis::new(&graph, function);
        let root_context = analysis.root_context();

        assert_eq!(
            analysis
                .evaluate_value(
                    ValueKey::new(
                        body,
                        ValueOrigin::Output {
                            producer: negated,
                            output: 0,
                        }
                    ),
                    root_context,
                )
                .to_scalar_constant(),
            Some(ScalarConstant::I32(4))
        );
        assert_eq!(
            analysis
                .evaluate_value(
                    ValueKey::new(
                        body,
                        ValueOrigin::Output {
                            producer: greater,
                            output: 0,
                        }
                    ),
                    root_context,
                )
                .to_scalar_constant(),
            Some(ScalarConstant::Bool(true))
        );
    }

    #[test]
    fn unknown_argument_propagates_through_arithmetic() {
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
                ret_ty: None,
            },
        );

        let mut graph = Rvsdg::new(module.ty.clone());
        let (_, body) = graph.register_function(&module, function, iter::empty());

        let two = graph.add_const_u32(body, 2);

        let sum = graph.add_op_binary(
            body,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 0),
            ValueInput::output(TY_U32, two, 0),
        );

        let mut analysis = ContextualValueAnalysis::new(&graph, function);
        let root_context = analysis.root_context();

        assert!(
            analysis
                .evaluate_value(ValueKey::new(body, ValueOrigin::Argument(0)), root_context)
                .is_top()
        );
        assert!(
            analysis
                .evaluate_value(
                    ValueKey::new(
                        body,
                        ValueOrigin::Output {
                            producer: sum,
                            output: 0,
                        }
                    ),
                    root_context,
                )
                .is_top()
        );
    }

    #[test]
    fn boolean_branch_assumptions_refine_constraints() {
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

        let mut graph = Rvsdg::new(module.ty.clone());
        let (_, body) = graph.register_function(&module, function, iter::empty());

        let negated =
            graph.add_op_unary(body, UnaryOperator::Not, ValueInput::argument(TY_BOOL, 0));

        let selector =
            graph.add_op_bool_to_branch_selector(body, ValueInput::output(TY_BOOL, negated, 0));
        let switch = graph.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, selector, 0)],
            Vec::new(),
            None,
        );
        graph.add_switch_branch(switch);
        graph.add_switch_branch(switch);

        let mut analysis = ContextualValueAnalysis::new(&graph, function);
        let root_context = analysis.root_context();

        // Without assumptions (in the root context), both the function argument and the
        // `negated` should evaluate to "top".
        assert!(
            analysis
                .evaluate_value(ValueKey::new(body, ValueOrigin::Argument(0)), root_context)
                .is_top()
        );
        assert!(
            analysis
                .evaluate_value(
                    ValueKey::new(
                        body,
                        ValueOrigin::Output {
                            producer: negated,
                            output: 0,
                        }
                    ),
                    root_context,
                )
                .is_top()
        );

        // Assume the first branch
        let branch_0_context = analysis
            .assume_branch(root_context, switch, 0)
            .expect_context();
        assert_eq!(
            analysis
                .evaluate_value(
                    ValueKey::new(body, ValueOrigin::Argument(0)),
                    branch_0_context
                )
                .to_scalar_constant(),
            Some(ScalarConstant::Bool(false))
        );
        assert_eq!(
            analysis
                .evaluate_value(
                    ValueKey::new(
                        body,
                        ValueOrigin::Output {
                            producer: negated,
                            output: 0,
                        }
                    ),
                    branch_0_context,
                )
                .to_scalar_constant(),
            Some(ScalarConstant::Bool(true))
        );
        assert_eq!(
            analysis.assume_branch(branch_0_context, switch, 1),
            AssumptionResult::Contradictory
        );

        // Assume the second branch
        let branch_1_context = analysis
            .assume_branch(root_context, switch, 1)
            .expect_context();
        assert_eq!(
            analysis
                .evaluate_value(
                    ValueKey::new(body, ValueOrigin::Argument(0)),
                    branch_1_context
                )
                .to_scalar_constant(),
            Some(ScalarConstant::Bool(true))
        );
        assert_eq!(
            analysis
                .evaluate_value(
                    ValueKey::new(
                        body,
                        ValueOrigin::Output {
                            producer: negated,
                            output: 0,
                        }
                    ),
                    branch_1_context,
                )
                .to_scalar_constant(),
            Some(ScalarConstant::Bool(false))
        );
        assert_eq!(
            analysis.assume_branch(branch_1_context, switch, 0),
            AssumptionResult::Contradictory
        );
    }

    #[test]
    fn branch_assumptions_refine_values_outside_direct_reverse_value_flow() {
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
                ret_ty: None,
            },
        );

        let mut graph = Rvsdg::new(module.ty.clone());
        let (_, body) = graph.register_function(&module, function, iter::empty());

        let eight = graph.add_const_u32(body, 8);
        let sixteen = graph.add_const_u32(body, 16);

        let lt8 = graph.add_op_binary(
            body,
            BinaryOperator::Lt,
            ValueInput::argument(TY_U32, 0),
            ValueInput::output(TY_U32, eight, 0),
        );

        let lt16 = graph.add_op_binary(
            body,
            BinaryOperator::Lt,
            ValueInput::argument(TY_U32, 0),
            ValueInput::output(TY_U32, sixteen, 0),
        );

        let switch_selector =
            graph.add_op_bool_to_branch_selector(body, ValueInput::output(TY_BOOL, lt8, 0));
        let switch = graph.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, switch_selector, 0)],
            Vec::new(),
            None,
        );
        graph.add_switch_branch(switch);
        graph.add_switch_branch(switch);

        let x = ValueKey::new(body, ValueOrigin::Argument(0));

        let mut analysis = ContextualValueAnalysis::new(&graph, function);
        let root_context = analysis.root_context();

        assert!(
            analysis
                .evaluate_value(
                    ValueKey::new(
                        graph[lt16].region(),
                        ValueOrigin::Output {
                            producer: lt16,
                            output: 0,
                        }
                    ),
                    root_context
                )
                .is_top()
        );
        assert!(analysis.evaluate_value(x, root_context).is_top());

        // An assumption that affects `x` via `lt8` should also affect `lt16`, which is not part of
        // the direct reverse value-flow off the branch-selector value, but also uses `x` as an
        // operand.

        let branch_0_context = analysis
            .assume_branch(root_context, switch, 0)
            .expect_context();

        assert_eq!(
            analysis.evaluate_value(x, branch_0_context),
            AbstractValue::from(AbstractU32::from_intervals([0..=7]))
        );
        assert_eq!(
            analysis
                .evaluate_value(
                    ValueKey::new(
                        graph[lt16].region(),
                        ValueOrigin::Output {
                            producer: lt16,
                            output: 0,
                        }
                    ),
                    branch_0_context
                )
                .to_scalar_constant(),
            Some(ScalarConstant::Bool(true))
        );
    }

    #[test]
    fn chained_branch_assumptions_refine_bounds() {
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
                ret_ty: None,
            },
        );

        let mut graph = Rvsdg::new(module.ty.clone());
        let (_, body) = graph.register_function(&module, function, iter::empty());

        let constant_0 = graph.add_const_u32(body, 20);
        let comparison_0 = graph.add_op_binary(
            body,
            BinaryOperator::LtEq,
            ValueInput::argument(TY_U32, 0),
            ValueInput::output(TY_U32, constant_0, 0),
        );
        let switch_selector_0 = graph
            .add_op_bool_to_branch_selector(body, ValueInput::output(TY_BOOL, comparison_0, 0));
        let switch_0 = graph.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, switch_selector_0, 0)],
            Vec::new(),
            None,
        );
        graph.add_switch_branch(switch_0);
        graph.add_switch_branch(switch_0);

        let constant_1 = graph.add_const_u32(body, 5);
        let comparison_1 = graph.add_op_binary(
            body,
            BinaryOperator::GtEq,
            ValueInput::argument(TY_U32, 0),
            ValueInput::output(TY_U32, constant_1, 0),
        );
        let switch_selector_1 = graph
            .add_op_bool_to_branch_selector(body, ValueInput::output(TY_BOOL, comparison_1, 0));
        let switch_1 = graph.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, switch_selector_1, 0)],
            Vec::new(),
            None,
        );
        graph.add_switch_branch(switch_1);
        graph.add_switch_branch(switch_1);

        let constant_2 = graph.add_const_u32(body, 100);
        let comparison_2 = graph.add_op_binary(
            body,
            BinaryOperator::Lt,
            ValueInput::argument(TY_U32, 0),
            ValueInput::output(TY_U32, constant_2, 0),
        );
        let switch_selector_2 = graph
            .add_op_bool_to_branch_selector(body, ValueInput::output(TY_BOOL, comparison_2, 0));
        let switch_2 = graph.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, switch_selector_2, 0)],
            Vec::new(),
            None,
        );
        graph.add_switch_branch(switch_2);
        graph.add_switch_branch(switch_2);

        let x = ValueKey::new(body, ValueOrigin::Argument(0));

        let mut analysis = ContextualValueAnalysis::new(&graph, function);
        let root_context = analysis.root_context();

        let switch_0_branch_0_context = analysis
            .assume_branch(root_context, switch_0, 0)
            .expect_context();

        assert_eq!(
            analysis.evaluate_value(x, switch_0_branch_0_context),
            AbstractValue::from(AbstractU32::from_intervals([0..=20]))
        );
        assert_eq!(
            analysis.assume_branch(switch_0_branch_0_context, switch_2, 1),
            AssumptionResult::Contradictory
        );

        let switch_0_branch_0_switch_1_branch_0_context = analysis
            .assume_branch(switch_0_branch_0_context, switch_1, 0)
            .expect_context();

        assert_eq!(
            analysis.evaluate_value(x, switch_0_branch_0_switch_1_branch_0_context),
            AbstractValue::from(AbstractU32::from_intervals([5..=20]))
        );
    }

    #[test]
    fn case_branch_assumptions_refine_arithmetic_input() {
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
                    ty: TY_I32,
                    shader_io_binding: None,
                }],
                ret_ty: None,
            },
        );

        let mut graph = Rvsdg::new(module.ty.clone());
        let (_, body) = graph.register_function(&module, function, iter::empty());

        let negated = graph.add_op_unary(body, UnaryOperator::Neg, ValueInput::argument(TY_I32, 0));

        let switch_selector = graph.add_op_case_to_branch_selector(
            body,
            ValueInput::output(TY_I32, negated, 0),
            Int::I32,
            [3u32].map(BranchCase::from),
        );
        let switch = graph.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, switch_selector, 0)],
            Vec::new(),
            None,
        );
        graph.add_switch_branch(switch);
        graph.add_switch_branch(switch);

        let mut analysis = ContextualValueAnalysis::new(&graph, function);
        let root_context = analysis.root_context();

        assert!(
            analysis
                .evaluate_value(ValueKey::new(body, ValueOrigin::Argument(0)), root_context)
                .is_top()
        );

        let branch_0_context = analysis
            .assume_branch(root_context, switch, 0)
            .expect_context();

        assert_eq!(
            analysis
                .evaluate_value(
                    ValueKey::new(body, ValueOrigin::Argument(0)),
                    branch_0_context,
                )
                .to_scalar_constant(),
            Some(ScalarConstant::I32(-3))
        );
    }

    #[test]
    fn switch_output_joins_branch_results() {
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

        let mut graph = Rvsdg::new(module.ty.clone());
        let (_, body) = graph.register_function(&module, function, iter::empty());

        let switch_selector =
            graph.add_op_bool_to_branch_selector(body, ValueInput::argument(TY_BOOL, 0));
        let switch = graph.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, switch_selector, 0)],
            vec![ValueOutput::new(TY_U32), ValueOutput::new(TY_U32)],
            None,
        );

        let switch_branch_0 = graph.add_switch_branch(switch);
        let switch_branch_0_output_0 = graph.add_const_u32(switch_branch_0, 1);
        let switch_branch_0_output_1 = graph.add_const_u32(switch_branch_0, 8);
        graph.reconnect_region_result(
            switch_branch_0,
            0,
            ValueOrigin::Output {
                producer: switch_branch_0_output_0,
                output: 0,
            },
        );
        graph.reconnect_region_result(
            switch_branch_0,
            1,
            ValueOrigin::Output {
                producer: switch_branch_0_output_1,
                output: 0,
            },
        );

        let switch_branch_1 = graph.add_switch_branch(switch);
        let switch_branch_1_output_0 = graph.add_const_u32(switch_branch_1, 0);
        let switch_branch_1_output_1 = graph.add_const_u32(switch_branch_1, 9);
        graph.reconnect_region_result(
            switch_branch_1,
            0,
            ValueOrigin::Output {
                producer: switch_branch_1_output_0,
                output: 0,
            },
        );
        graph.reconnect_region_result(
            switch_branch_1,
            1,
            ValueOrigin::Output {
                producer: switch_branch_1_output_1,
                output: 0,
            },
        );

        let mut analysis = ContextualValueAnalysis::new(&graph, function);
        let root_context = analysis.root_context();

        assert_eq!(
            analysis.evaluate_value(
                ValueKey::new(
                    graph[switch].region(),
                    ValueOrigin::Output {
                        producer: switch,
                        output: 0,
                    }
                ),
                root_context
            ),
            AbstractValue::from(AbstractU32::from_intervals([0..=1]))
        );
        assert_eq!(
            analysis.evaluate_value(
                ValueKey::new(
                    body,
                    ValueOrigin::Output {
                        producer: switch,
                        output: 1,
                    }
                ),
                root_context,
            ),
            AbstractValue::from(AbstractU32::from_intervals([8..=9]))
        );

        let branch_0_context = analysis
            .assume_branch(root_context, switch, 0)
            .expect_context();

        assert_eq!(
            analysis
                .evaluate_value(
                    ValueKey::new(
                        body,
                        ValueOrigin::Output {
                            producer: switch,
                            output: 0,
                        }
                    ),
                    branch_0_context
                )
                .to_scalar_constant(),
            Some(ScalarConstant::U32(1))
        );
        assert_eq!(
            analysis
                .evaluate_branch_result(switch, 1, 0, root_context)
                .to_scalar_constant(),
            Some(ScalarConstant::U32(8))
        );

        let branch_1_context = analysis
            .assume_branch(root_context, switch, 1)
            .expect_context();

        assert_eq!(
            analysis
                .evaluate_value(
                    ValueKey::new(
                        body,
                        ValueOrigin::Output {
                            producer: switch,
                            output: 0,
                        }
                    ),
                    branch_1_context
                )
                .to_scalar_constant(),
            Some(ScalarConstant::U32(0))
        );
        assert_eq!(
            analysis
                .evaluate_branch_result(switch, 1, 1, root_context)
                .to_scalar_constant(),
            Some(ScalarConstant::U32(9))
        );
    }

    #[test]
    fn default_subtracts_cases_from_available_bound() {
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
                ret_ty: None,
            },
        );

        let mut graph = Rvsdg::new(module.ty.clone());
        let (_, body) = graph.register_function(&module, function, iter::empty());

        let switch_selector = graph.add_op_case_to_branch_selector(
            body,
            ValueInput::argument(TY_U32, 0),
            Int::U32,
            [2u32, 3, 6, 12].map(BranchCase::from),
        );
        let switch = graph.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, switch_selector, 0)],
            Vec::new(),
            None,
        );
        graph.add_switch_branch(switch);
        graph.add_switch_branch(switch);
        graph.add_switch_branch(switch);
        graph.add_switch_branch(switch);
        graph.add_switch_branch(switch);

        let limit = graph.add_const_u32(body, 10);

        let comparison = graph.add_op_binary(
            body,
            BinaryOperator::LtEq,
            ValueInput::argument(TY_U32, 0),
            ValueInput::output(TY_U32, limit, 0),
        );

        let bound_selector =
            graph.add_op_bool_to_branch_selector(body, ValueInput::output(TY_BOOL, comparison, 0));
        let bounded = graph.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, bound_selector, 0)],
            Vec::new(),
            None,
        );
        graph.add_switch_branch(bounded);
        graph.add_switch_branch(bounded);

        let key = ValueKey::new(body, ValueOrigin::Argument(0));

        let mut analysis = ContextualValueAnalysis::new(&graph, function);
        let root_context = analysis.root_context();

        let bounded_branch_0_context = analysis
            .assume_branch(root_context, bounded, 0)
            .expect_context();

        let bounded_branch_0_switch_default_branch_context = analysis
            .assume_branch(bounded_branch_0_context, switch, 4)
            .expect_context();

        assert_eq!(
            analysis.evaluate_value(key, bounded_branch_0_switch_default_branch_context),
            AbstractValue::from(AbstractU32::from_intervals([0..=1, 4..=5, 7..=10]))
        );
    }

    #[test]
    fn unrepresentable_default_keeps_input_unconstrained() {
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
                ret_ty: None,
            },
        );

        let mut graph = Rvsdg::new(module.ty.clone());
        let (_, body) = graph.register_function(&module, function, iter::empty());

        let cases: Vec<u32> = (0..24).step_by(2).collect();

        let switch_selector = graph.add_op_case_to_branch_selector(
            body,
            ValueInput::argument(TY_U32, 0),
            Int::U32,
            cases.iter().copied().map(BranchCase::from),
        );
        let switch = graph.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, switch_selector, 0)],
            Vec::new(),
            None,
        );

        for _ in 0..=cases.len() {
            graph.add_switch_branch(switch);
        }

        let mut analysis = ContextualValueAnalysis::new(&graph, function);
        let root_context = analysis.root_context();

        let switch_default_branch_context = analysis
            .assume_branch(root_context, switch, cases.len())
            .expect_context();

        // Assuming the default branch for the switch node would allow us to exclude all case values
        // from the OpCaseToBranchSelector's input. However, representing that constraint requires
        // more intervals than we support for a `u32` value, so we currently fall back to the
        // prior constraints (in this case: "top").

        assert!(
            analysis
                .evaluate_value(
                    ValueKey::new(body, ValueOrigin::Argument(0)),
                    switch_default_branch_context
                )
                .is_top()
        );
    }

    #[test]
    fn saturation_propagates_past_upstream_switch_node() {
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
                args: vec![
                    FnArg {
                        ty: TY_BOOL,
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

        let mut graph = Rvsdg::new(module.ty.clone());
        let (_, body) = graph.register_function(&module, function, iter::empty());

        let producer_selector =
            graph.add_op_bool_to_branch_selector(body, ValueInput::argument(TY_BOOL, 0));
        let producer = graph.add_switch(
            body,
            vec![
                ValueInput::output(TY_PREDICATE, producer_selector, 0),
                ValueInput::argument(TY_U32, 1),
            ],
            vec![ValueOutput::new(TY_U32), ValueOutput::new(TY_U32)],
            None,
        );

        let producer_branch_0 = graph.add_switch_branch(producer);
        let producer_branch_0_output_1 = graph.add_const_u32(producer_branch_0, 8);
        graph.reconnect_region_result(producer_branch_0, 0, ValueOrigin::Argument(0));
        graph.reconnect_region_result(
            producer_branch_0,
            1,
            ValueOrigin::Output {
                producer: producer_branch_0_output_1,
                output: 0,
            },
        );

        let producer_branch_1 = graph.add_switch_branch(producer);
        let producer_branch_1_output_0 = graph.add_const_u32(producer_branch_1, 1);
        let producer_branch_1_output_1 = graph.add_const_u32(producer_branch_1, 9);
        graph.reconnect_region_result(
            producer_branch_1,
            0,
            ValueOrigin::Output {
                producer: producer_branch_1_output_0,
                output: 0,
            },
        );
        graph.reconnect_region_result(
            producer_branch_1,
            1,
            ValueOrigin::Output {
                producer: producer_branch_1_output_1,
                output: 0,
            },
        );

        let consumer_selector = graph.add_op_case_to_branch_selector(
            body,
            ValueInput::output(TY_U32, producer, 0),
            Int::U32,
            [0u32].map(BranchCase::from),
        );
        let consumer = graph.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, consumer_selector, 0)],
            Vec::new(),
            None,
        );
        graph.add_switch_branch(consumer);
        graph.add_switch_branch(consumer);

        let conflicting_consumer_selector = graph.add_op_case_to_branch_selector(
            body,
            ValueInput::output(TY_U32, producer, 1),
            Int::U32,
            [9u32].map(BranchCase::from),
        );
        let conflicting_consumer = graph.add_switch(
            body,
            vec![ValueInput::output(
                TY_PREDICATE,
                conflicting_consumer_selector,
                0,
            )],
            Vec::new(),
            None,
        );
        graph.add_switch_branch(conflicting_consumer);
        graph.add_switch_branch(conflicting_consumer);

        let mut analysis = ContextualValueAnalysis::new(&graph, function);
        let root_context = analysis.root_context();

        let consumer_branch_0_context = analysis
            .assume_branch(root_context, consumer, 0)
            .expect_context();

        assert_eq!(
            analysis
                .evaluate_value(
                    ValueKey::new(
                        body,
                        ValueOrigin::Output {
                            producer,
                            output: 1,
                        },
                    ),
                    consumer_branch_0_context,
                )
                .to_scalar_constant(),
            Some(ScalarConstant::U32(8))
        );
        assert_eq!(
            analysis
                .evaluate_value(
                    ValueKey::new(body, ValueOrigin::Argument(0)),
                    consumer_branch_0_context,
                )
                .to_scalar_constant(),
            Some(ScalarConstant::Bool(true))
        );
        assert_eq!(
            analysis
                .evaluate_value(
                    ValueKey::new(body, ValueOrigin::Argument(1)),
                    consumer_branch_0_context,
                )
                .to_scalar_constant(),
            Some(ScalarConstant::U32(0))
        );

        assert_eq!(
            analysis.assume_branch(consumer_branch_0_context, conflicting_consumer, 0),
            AssumptionResult::Contradictory
        );
    }

    #[test]
    fn constant_condition_excludes_unreachable_branch() {
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

        let mut graph = Rvsdg::new(module.ty.clone());
        let (_, body) = graph.register_function(&module, function, iter::empty());

        let condition = graph.add_const_bool(body, false);

        let selector =
            graph.add_op_bool_to_branch_selector(body, ValueInput::output(TY_BOOL, condition, 0));
        let switch = graph.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, selector, 0)],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let branch_0 = graph.add_switch_branch(switch);
        let three = graph.add_const_u32(branch_0, 3);
        graph.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: three,
                output: 0,
            },
        );

        let branch_1 = graph.add_switch_branch(switch);
        let seven = graph.add_const_u32(branch_1, 7);
        graph.reconnect_region_result(
            branch_1,
            0,
            ValueOrigin::Output {
                producer: seven,
                output: 0,
            },
        );

        let mut analysis = ContextualValueAnalysis::new(&graph, function);
        let root_context = analysis.root_context();

        assert_eq!(
            analysis
                .evaluate_value(
                    ValueKey::new(
                        body,
                        ValueOrigin::Output {
                            producer: switch,
                            output: 0,
                        }
                    ),
                    root_context,
                )
                .to_scalar_constant(),
            Some(ScalarConstant::U32(7))
        );
        assert_eq!(
            analysis.assume_branch(root_context, switch, 0),
            AssumptionResult::Contradictory
        );
        assert!(
            analysis
                .evaluate_branch_result(switch, 0, 0, root_context)
                .is_bottom()
        );
        assert_eq!(
            analysis
                .evaluate_branch_result(switch, 0, 1, root_context)
                .to_scalar_constant(),
            Some(ScalarConstant::U32(7))
        );
    }

    #[test]
    fn inner_branch_assumptions_imply_enclosing_branch() {
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
                args: vec![
                    FnArg {
                        ty: TY_BOOL,
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

        let mut graph = Rvsdg::new(module.ty.clone());
        let (_, body) = graph.register_function(&module, function, iter::empty());

        let outer_selector =
            graph.add_op_bool_to_branch_selector(body, ValueInput::argument(TY_BOOL, 0));
        let outer_switch = graph.add_switch(
            body,
            vec![
                ValueInput::output(TY_PREDICATE, outer_selector, 0),
                ValueInput::argument(TY_U32, 1),
            ],
            Vec::new(),
            None,
        );

        let outer_branch_0 = graph.add_switch_branch(outer_switch);

        let inner_selector = graph.add_op_case_to_branch_selector(
            outer_branch_0,
            ValueInput::argument(TY_U32, 0),
            Int::U32,
            [5u32].map(BranchCase::from),
        );
        let inner_switch = graph.add_switch(
            outer_branch_0,
            vec![ValueInput::output(TY_PREDICATE, inner_selector, 0)],
            Vec::new(),
            None,
        );
        graph.add_switch_branch(inner_switch);
        graph.add_switch_branch(inner_switch);

        let outer_branch_1 = graph.add_switch_branch(outer_switch);

        let mut analysis = ContextualValueAnalysis::new(&graph, function);
        let root_context = analysis.root_context();

        let inner_switch_branch_0_context = analysis
            .assume_branch(root_context, inner_switch, 0)
            .expect_context();

        assert_eq!(
            analysis
                .evaluate_value(
                    ValueKey::new(body, ValueOrigin::Argument(0)),
                    inner_switch_branch_0_context
                )
                .to_scalar_constant(),
            Some(ScalarConstant::Bool(true))
        );
        assert_eq!(
            analysis
                .evaluate_value(
                    ValueKey::new(body, ValueOrigin::Argument(1)),
                    inner_switch_branch_0_context
                )
                .to_scalar_constant(),
            Some(ScalarConstant::U32(5))
        );
        assert_eq!(
            analysis
                .evaluate_value(
                    ValueKey::new(outer_branch_0, ValueOrigin::Argument(0)),
                    inner_switch_branch_0_context
                )
                .to_scalar_constant(),
            Some(ScalarConstant::U32(5))
        );
        assert!(
            analysis
                .evaluate_value(
                    ValueKey::new(outer_branch_1, ValueOrigin::Argument(0)),
                    inner_switch_branch_0_context
                )
                .is_bottom()
        );

        let outer_switch_branch_branch_1 = analysis
            .assume_branch(root_context, outer_switch, 1)
            .expect_context();

        assert_eq!(
            analysis.assume_branch(outer_switch_branch_branch_1, inner_switch, 0),
            AssumptionResult::Contradictory
        );
    }

    #[test]
    fn zero_work_budget_produces_conservative_values() {
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

        let mut graph = Rvsdg::new(module.ty.clone());
        let (_, body) = graph.register_function(&module, function, iter::empty());

        let a = graph.add_const_u32(body, 1);
        let b = graph.add_const_u32(body, 2);

        let sum = graph.add_op_binary(
            body,
            BinaryOperator::Add,
            ValueInput::output(TY_U32, a, 0),
            ValueInput::output(TY_U32, b, 0),
        );

        let selector = graph.add_op_bool_to_branch_selector(body, ValueInput::argument(TY_BOOL, 0));
        let switch = graph.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, selector, 0)],
            Vec::new(),
            None,
        );
        graph.add_switch_branch(switch);
        graph.add_switch_branch(switch);

        // First verify that with sufficient work budget, we derive constrained values.

        let mut analysis = ContextualValueAnalysis::new(&graph, function);
        let root_context = analysis.root_context();

        assert_eq!(
            analysis
                .evaluate_value(
                    ValueKey::new(
                        body,
                        ValueOrigin::Output {
                            producer: sum,
                            output: 0,
                        },
                    ),
                    root_context
                )
                .to_scalar_constant(),
            Some(ScalarConstant::U32(3))
        );

        let switch_branch_0_context = analysis
            .assume_branch(root_context, switch, 0)
            .expect_context();

        assert_eq!(
            analysis
                .evaluate_value(
                    ValueKey::new(body, ValueOrigin::Argument(0)),
                    switch_branch_0_context
                )
                .to_scalar_constant(),
            Some(ScalarConstant::Bool(true))
        );

        // Now verify that with a limited work budget, we get "top" fallback values for the same
        // queries.

        let mut limited = ContextualValueAnalysis::with_work_budget(&graph, function, 0);
        let root_context = limited.root_context();

        assert!(
            limited
                .evaluate_value(
                    ValueKey::new(
                        body,
                        ValueOrigin::Output {
                            producer: sum,
                            output: 0,
                        },
                    ),
                    root_context
                )
                .is_top()
        );

        let switch_branch_0_context = limited
            .assume_branch(root_context, switch, 0)
            .expect_context();

        assert!(
            limited
                .evaluate_value(
                    ValueKey::new(body, ValueOrigin::Argument(0)),
                    switch_branch_0_context
                )
                .is_top()
        );
    }
}
