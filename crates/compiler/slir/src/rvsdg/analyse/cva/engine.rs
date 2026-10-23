use std::mem;

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
    force_retry: bool,
    did_stop_early: bool,
    hit_depth_or_recursion_limit: bool,

    /// A set that logs the active value queries during the current analysis, used to guard against
    /// recursion.
    active_value_queries: FxHashSet<ValueKey>,
}

impl<'a> Engine<'a> {
    pub fn new(rvsdg: &'a Rvsdg, remaining: usize, force_retry: bool) -> Self {
        Self {
            rvsdg,
            remaining_work_budget: remaining,
            force_retry,
            did_stop_early: false,
            hit_depth_or_recursion_limit: false,
            active_value_queries: Default::default(),
        }
    }

    /// Evaluates a supported scalar value. Recursive value-flow traversal must preserve type support.
    pub fn evaluate_value(
        &mut self,
        key: ValueKey,
        env: &Environment,
        cache: &mut QueryCache,
    ) -> AbstractValue {
        let ty = key.ty(self.rvsdg);

        debug_assert!(AbstractValue::supports_type(ty));

        // If the value does not occur within a region that is part of the currently assumed
        // control-flow, then the value never exists and is contradictory; we'll signal this with a
        // "bottom" value.
        if !env.is_in_context(self.rvsdg, key.region) {
            return AbstractValue::bottom(ty);
        }

        let key = key.canonicalize(self.rvsdg);

        if let Some(value) = cache.get_complete(key) {
            // We've successfully evaluated this value before, so we can return the cached result.
            return value.clone();
        }

        let budget = self.remaining_work_budget;

        if !self.force_retry {
            if let Some(previous) = cache.get_incomplete(key) {
                if previous
                    .largest_requested_budget
                    .is_some_and(|b| b >= budget)
                    && self.active_value_queries.len() < MAX_DEPTH
                    && !self.active_value_queries.contains(&key)
                {
                    self.did_stop_early = true;

                    return previous.constraints.clone();
                }
            }
        }

        let can_expand = self.try_take_analysis_step();

        if !can_expand || !self.active_value_queries.insert(key) {
            let recursive = self.active_value_queries.contains(&key);

            if recursive {
                self.did_stop_early = true;
                self.hit_depth_or_recursion_limit = true;
            }

            let value = cache
                .get_incomplete(key)
                .map_or_else(|| AbstractValue::top(ty), |value| value.constraints.clone());

            if !can_expand && self.active_value_queries.len() < MAX_DEPTH && !recursive {
                cache.record_incomplete(key, &value, Some(budget));
            }

            return value;
        }

        let parent_did_stop_early = mem::take(&mut self.did_stop_early);
        let parent_hit_depth_or_recursion_limit = mem::take(&mut self.hit_depth_or_recursion_limit);

        let value = self.forward_infer_constraints(key, env, cache);
        let value = value.refine(&env.assumption_derived_constraints(self.rvsdg, key));

        // Integer refinement uses a fallback mechanism where, if the maximum supported intervals
        // are exceeded, it falls back to the prior `self` value. We therefore use the `previous`
        // value as the `self` value here to ensure we at least never regress the result.
        let value = if let Some(previous) = cache.get_incomplete(key) {
            if previous.constraints.is_top() || previous.constraints == value {
                value
            } else {
                previous.constraints.refine(&value)
            }
        } else {
            value
        };

        self.active_value_queries.remove(&key);

        if !self.did_stop_early {
            cache.record_complete(key, value.clone());
        } else {
            let budget = (!self.hit_depth_or_recursion_limit).then_some(budget);
            cache.record_incomplete(key, &value, budget);
        }

        self.did_stop_early |= parent_did_stop_early;
        self.hit_depth_or_recursion_limit |= parent_hit_depth_or_recursion_limit;

        value
    }

    fn try_consume_work_unit(&mut self) -> bool {
        if self.remaining_work_budget == 0 {
            self.did_stop_early = true;

            false
        } else {
            self.remaining_work_budget -= 1;

            true
        }
    }

    fn try_take_analysis_step(&mut self) -> bool {
        if self.active_value_queries.len() >= MAX_DEPTH {
            self.did_stop_early = true;
            self.hit_depth_or_recursion_limit = true;

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
