use rustc_hash::FxHashSet;

use super::domain::AbstractPredicate;
use super::{AbstractValue, Environment, QueryCache, branches_matching_selector};
use crate::rvsdg::analyse::ValueKey;
use crate::rvsdg::analyse::scalar_constant::ScalarConstant;
use crate::rvsdg::{NodeKind, Rvsdg, SimpleNode, ValueOrigin};

const MAX_DEPTH: usize = 64;

/// Recursion protection belongs to the query, not its discardable caches.
pub struct Engine<'a> {
    pub rvsdg: &'a Rvsdg,
    pub remaining_work_budget: usize,

    /// A number that increments every time we have to fall back to a conservative result because we
    /// ran out of work budget or exceeded the analysis depth limit.
    ///
    /// We store a copy of this value before running a query, then check the copy against the
    /// actual value after the query to determine whether we had to fallback. This then informs the
    /// caching decision, as we don't want to cache "incomplete" results that we may be able to
    /// improve later.
    conservative_fallback_revision: usize,

    /// A set that logs the active value queries during the current analysis, used to guard against
    /// recursion.
    active_value_queries: FxHashSet<ValueKey>,
}

impl<'a> Engine<'a> {
    pub fn new(rvsdg: &'a Rvsdg, remaining: usize) -> Self {
        Self {
            rvsdg,
            remaining_work_budget: remaining,
            conservative_fallback_revision: 0,
            active_value_queries: Default::default(),
        }
    }

    pub fn evaluate_value(
        &mut self,
        key: ValueKey,
        env: &Environment,
        cache: &mut QueryCache,
    ) -> AbstractValue {
        let ty = key.ty(self.rvsdg);

        if let value @ AbstractValue::Unsupported(_) = AbstractValue::top(ty) {
            return value;
        }

        // If the value does not occur within a region that is part of the currently assumed
        // control-flow, then the value never exists and is contradictory; we'll signal this with a
        // "bottom" value.
        if !env.is_in_context(self.rvsdg, key.region) {
            return AbstractValue::bottom(ty);
        }

        let key = key.canonicalize(self.rvsdg);

        if let Some(value) = cache.value_constraints.get(&key) {
            // We've successfully evaluated this value before, so we can return the cached result.

            return value.clone();
        }

        if !self.try_take_analysis_step() {
            // We ran out of work budget or exceeded the depth limit; conservatively return "top".

            return AbstractValue::top(ty);
        }

        if !self.active_value_queries.insert(key) {
            // We seem to have recursed to a value we encountered earlier in this same query chain;
            // conservatively return "top".

            self.conservative_fallback_revision += 1;

            return AbstractValue::top(ty);
        }

        let revision = self.conservative_fallback_revision;
        let value = self.forward_infer_constraints(key, env, cache);
        let value = value.refine(&env.assumption_derived_constraints(self.rvsdg, key));

        self.active_value_queries.remove(&key);

        // Cache the result, but only if the revision number was not incremented during the
        // evaluation of the query; if the revision was incremented, the value may be incomplete,
        // and we don't want to cache incomplete results that we may be able to improve later.
        if revision == self.conservative_fallback_revision {
            cache.value_constraints.insert(key, value.clone());
        }

        value
    }

    fn try_consume_work_unit(&mut self) -> bool {
        if self.remaining_work_budget == 0 {
            self.conservative_fallback_revision += 1;

            false
        } else {
            self.remaining_work_budget -= 1;

            true
        }
    }

    fn try_take_analysis_step(&mut self) -> bool {
        if self.active_value_queries.len() >= MAX_DEPTH {
            self.conservative_fallback_revision += 1;

            return false;
        }

        self.try_consume_work_unit()
    }

    /// Infers a value's constraints from its producer node and the constraints on the producer
    /// node's inputs.
    ///
    /// This does not represent the full set of constraints on the value: we may have derived
    /// additional constraints on the value by backward propagation of the control-flow assumptions
    /// made by the context environment. Such constraints are not included here.
    fn forward_infer_constraints(
        &mut self,
        key: ValueKey,
        env: &Environment,
        cache: &mut QueryCache,
    ) -> AbstractValue {
        let ty = key.ty(self.rvsdg);

        let ValueOrigin::Output { producer, output } = key.origin else {
            return AbstractValue::top(ty);
        };

        if let Some(constant) = ScalarConstant::from_node(self.rvsdg, producer, output) {
            return AbstractValue::from_scalar_constant(constant);
        }

        if self.rvsdg[producer].is_switch() {
            let selector = self.evaluate_value(
                ValueKey::for_node_input(self.rvsdg, producer, 0),
                env,
                cache,
            );
            let count = self.rvsdg[producer].expect_switch().branches().len();
            let branches = branches_matching_selector(&selector, count);

            let mut result = AbstractValue::bottom(ty);

            for branch in branches {
                let branch_result_key =
                    ValueKey::for_branch_result(self.rvsdg, producer, output, branch);
                let branch_result_value = self.evaluate_value(branch_result_key, env, cache);

                result = result.join(&branch_result_value);
            }

            return result;
        }

        if output != 0 {
            return AbstractValue::top(ty);
        }

        match self.rvsdg[producer].kind() {
            NodeKind::Simple(SimpleNode::OpBinary(op)) => {
                let op = op.operator();
                let lhs = self.evaluate_value(
                    ValueKey::for_node_input(self.rvsdg, producer, 0),
                    env,
                    cache,
                );
                let rhs = self.evaluate_value(
                    ValueKey::for_node_input(self.rvsdg, producer, 1),
                    env,
                    cache,
                );

                lhs.abstract_binary_op(op, &rhs)
            }
            NodeKind::Simple(SimpleNode::OpUnary(op)) => {
                let op = op.operator();
                let value = self.evaluate_value(
                    ValueKey::for_node_input(self.rvsdg, producer, 0),
                    env,
                    cache,
                );

                value.abstract_unary_op(op)
            }
            NodeKind::Simple(SimpleNode::OpConvertToU32(_)) => {
                let value = self.evaluate_value(
                    ValueKey::for_node_input(self.rvsdg, producer, 0),
                    env,
                    cache,
                );

                value.to_abstract_u32().into()
            }
            NodeKind::Simple(SimpleNode::OpConvertToI32(_)) => {
                let value = self.evaluate_value(
                    ValueKey::for_node_input(self.rvsdg, producer, 0),
                    env,
                    cache,
                );

                value.to_abstract_i32().into()
            }
            NodeKind::Simple(SimpleNode::OpConvertToF32(_)) => {
                let value = self.evaluate_value(
                    ValueKey::for_node_input(self.rvsdg, producer, 0),
                    env,
                    cache,
                );

                value.to_abstract_f32().into()
            }
            NodeKind::Simple(SimpleNode::OpConvertToBool(_)) => {
                let value = self.evaluate_value(
                    ValueKey::for_node_input(self.rvsdg, producer, 0),
                    env,
                    cache,
                );

                value.to_abstract_bool().into()
            }
            NodeKind::Simple(SimpleNode::OpBoolToBranchSelector(_)) => {
                let value = self.evaluate_value(
                    ValueKey::for_node_input(self.rvsdg, producer, 0),
                    env,
                    cache,
                );

                AbstractPredicate::abstract_from_bool(value.expect_bool()).into()
            }
            NodeKind::Simple(SimpleNode::OpCaseToBranchSelector(selector)) => {
                let source = ValueKey::for_node_input(self.rvsdg, producer, 0);
                let value = self.evaluate_value(source, env, cache);

                let predicate = match &value {
                    AbstractValue::I32(value) => {
                        AbstractPredicate::abstract_from_case_i32(value, selector.cases())
                    }
                    AbstractValue::U32(value) => {
                        AbstractPredicate::abstract_from_case_u32(value, selector.cases())
                    }
                    _ => unreachable!("case selector input is an integer"),
                };

                predicate.into()
            }

            // Loop outputs and the outputs on unsupported simple nodes are not evaluated; we
            // conservatively return "top" to indicate that we have not derived any constraints.
            _ => AbstractValue::top(ty),
        }
    }
}
