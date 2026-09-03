//! Replaces fallback switch-branch results with a common scalar constant or branch argument.
//!
//! This pass assumes that [proxy_node_elimination] has already removed all value-proxy nodes.
//!
//! The compiler is allowed to replace fallback values with any value of a matching type. We inspect
//! the region results that correspond to the same switch output. If:
//!
//! - At least one branch returns a scalar constant or branch argument
//! - At least one branch returns a fallback value
//! - All other branches can be unified with that same value or are fallback values
//!
//! Values can be unified if they are the same branch argument or matching constant values. If all
//! values can be unified to the same branch argument, then in each branch where the result
//! connects to a fallback value, we reconnect the result to this branch argument. If all values
//! can be unified to a constant value, then in each branch where the result connects to a fallback
//! value, we add a node to represent this constant value and reconnect the result to this new node.
//! If the values can both be unified to the same branch argument and to the same constant, then
//! unifying to the branch argument takes precedence.
//!
//! After this pass, either [passthrough_elimination] (when unifying to a branch argument) or
//! [switch_output_extraction] (when unifying to a constant) can eliminate the switch output value.
//!
//! [passthrough_elimination]: crate::rvsdg::transform::passthrough_elimination
//! [proxy_node_elimination]: crate::rvsdg::transform::proxy_node_elimination
//! [switch_output_extraction]: crate::rvsdg::transform::switch_output_extraction

use crate::Module;
use crate::rvsdg::analyse::scalar_constant::ScalarConstant;
use crate::rvsdg::analyse::value_resolution::{ResolvedValue, ValueResolver};
use crate::rvsdg::visit::region_nodes::RegionNodesVisitor;
use crate::rvsdg::{Connectivity, Node, NodeKind, Region, Rvsdg, ValueOrigin, visit};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Unifiable {
    Unknown,
    Argument {
        index: u32,
        resolved_constant: Option<ScalarConstant>,
    },
    Constant(ScalarConstant),
    Impossible,
}

impl Unifiable {
    fn meet(self, other: Self) -> Self {
        match (self, other) {
            // If either side is impossible, then the result is always impossible
            (Self::Impossible, _) | (_, Self::Impossible) => Self::Impossible,

            // If either side is unknown, then the other side must be at least as constraining, so
            // we return that.
            (Self::Unknown, value) | (value, Self::Unknown) => value,

            // If both sides are constant, then we can unify if they have the same value. Different
            // value constants can never be unified.
            (Self::Constant(left), Self::Constant(right)) => {
                if left == right {
                    Self::Constant(left)
                } else {
                    Self::Impossible
                }
            }

            // If both sides resolve to the same branch argument, then we can unify to this branch
            // argument.
            (
                Self::Argument {
                    index: left_index,
                    resolved_constant: left_constant,
                },
                Self::Argument {
                    index: right_index,
                    resolved_constant: right_constant,
                },
            ) if left_index == right_index => {
                // Try to unify the constant values resolved for both arguments, if any. If a later
                // unification fails to unify to the branch argument, we can fallback to trying to
                // unifying to the constant value.
                let resolved_constant = match (left_constant, right_constant) {
                    (Some(left), Some(right)) if left == right => Some(left),
                    _ => None,
                };

                Self::Argument {
                    index: left_index,
                    resolved_constant,
                }
            }

            // If both sides resolve to a branch argument, but we failed to unify them to the same
            // branch argument above, then attempt to unify them to a constant value.
            (
                Self::Argument {
                    resolved_constant: Some(left),
                    ..
                },
                Self::Argument {
                    resolved_constant: Some(right),
                    ..
                },
            ) if left == right => Self::Constant(left),

            // Both sides resolve to a branch argument, but we failed to unify them to a branch
            // argument or a constant value above; unification is apparently impossible.
            (Self::Argument { .. }, Self::Argument { .. }) => Self::Impossible,

            // One side resolves to a branch argument and one side resolves to a constant value. If
            // the branch argument itself also resolved to a constant value, then we can unify if
            // both constant values are equal.
            (
                Self::Argument {
                    resolved_constant: Some(argument_constant),
                    ..
                },
                Self::Constant(constant),
            )
            | (
                Self::Constant(constant),
                Self::Argument {
                    resolved_constant: Some(argument_constant),
                    ..
                },
            ) if argument_constant == constant => Self::Constant(constant),

            // In all remaining cases, unification is impossible.
            _ => Self::Impossible,
        }
    }
}

struct ClassifiedResult {
    unifiable: Unifiable,
    is_fallback: bool,
}

struct ResultClassifier {
    resolver: ValueResolver,
}

impl ResultClassifier {
    fn new() -> Self {
        Self {
            resolver: ValueResolver::new(),
        }
    }

    fn classify(&mut self, rvsdg: &Rvsdg, branch: Region, origin: ValueOrigin) -> ClassifiedResult {
        let resolved = self.resolver.resolve(rvsdg, branch, origin);

        match (origin, resolved) {
            (_, ResolvedValue::Fallback) => ClassifiedResult {
                unifiable: Unifiable::Unknown,
                is_fallback: true,
            },
            (ValueOrigin::Argument(index), ResolvedValue::Constant(constant)) => ClassifiedResult {
                unifiable: Unifiable::Argument {
                    index,
                    resolved_constant: Some(constant),
                },
                is_fallback: false,
            },
            (ValueOrigin::Argument(index), ResolvedValue::Opaque { .. }) => ClassifiedResult {
                unifiable: Unifiable::Argument {
                    index,
                    resolved_constant: None,
                },
                is_fallback: false,
            },
            (_, ResolvedValue::Constant(constant)) => ClassifiedResult {
                unifiable: Unifiable::Constant(constant),
                is_fallback: false,
            },
            (_, ResolvedValue::Opaque { .. }) => ClassifiedResult {
                unifiable: Unifiable::Impossible,
                is_fallback: false,
            },
        }
    }
}

struct SwitchNodeCollector<'a> {
    queue: &'a mut Vec<Node>,
}

impl RegionNodesVisitor for SwitchNodeCollector<'_> {
    fn visit_node(&mut self, rvsdg: &Rvsdg, node: Node) {
        if let NodeKind::Switch(_) = rvsdg[node].kind() {
            self.queue.push(node);
        }

        visit::region_nodes::visit_node(self, rvsdg, node);
    }
}

pub struct SwitchFallbackUnifier {
    result_classifier: ResultClassifier,
    switch_node_queue: Vec<Node>,
    switch_branches: Vec<Region>,

    // Records the subset of the switch_branches that are classified as "Fallback".
    fallback_branches: Vec<Region>,
}

impl SwitchFallbackUnifier {
    pub fn new() -> Self {
        Self {
            result_classifier: ResultClassifier::new(),
            switch_node_queue: Vec::new(),
            switch_branches: Vec::new(),
            fallback_branches: Vec::new(),
        }
    }

    pub fn process_region(&mut self, rvsdg: &mut Rvsdg, region: Region) -> bool {
        self.switch_node_queue.clear();

        SwitchNodeCollector {
            queue: &mut self.switch_node_queue,
        }
        .visit_region(rvsdg, region);

        let mut changed = false;

        while let Some(switch_node) = self.switch_node_queue.pop() {
            changed |= self.process_switch_node(rvsdg, switch_node);
        }

        changed
    }

    fn process_switch_node(&mut self, rvsdg: &mut Rvsdg, switch_node: Node) -> bool {
        self.switch_branches.clear();

        let switch = rvsdg[switch_node].expect_switch();
        let output_count = switch.value_outputs().len();

        self.switch_branches.extend_from_slice(switch.branches());

        let mut changed = false;

        for output in 0..output_count {
            self.fallback_branches.clear();

            let mut result_meet = Unifiable::Unknown;

            for &branch in &self.switch_branches {
                let origin = rvsdg[branch].value_results()[output].origin;
                let result = self.result_classifier.classify(rvsdg, branch, origin);

                if result.is_fallback {
                    self.fallback_branches.push(branch);
                }

                result_meet = result_meet.meet(result.unifiable);

                if result_meet == Unifiable::Impossible {
                    break;
                }
            }

            if self.fallback_branches.is_empty() {
                continue;
            }

            for &branch in &self.fallback_branches {
                let replacement = match result_meet {
                    Unifiable::Argument { index, .. } => ValueOrigin::Argument(index),
                    Unifiable::Constant(constant) => {
                        let replacement_node = constant.add_to_region(rvsdg, branch);

                        ValueOrigin::Output {
                            producer: replacement_node,
                            output: 0,
                        }
                    }
                    Unifiable::Unknown | Unifiable::Impossible => continue,
                };

                rvsdg.reconnect_region_result(branch, output as u32, replacement);
            }

            changed |= matches!(
                result_meet,
                Unifiable::Argument { .. } | Unifiable::Constant(_)
            );
        }

        changed
    }
}

pub fn transform_entry_points(module: &Module, rvsdg: &mut Rvsdg) -> bool {
    let mut unifier = SwitchFallbackUnifier::new();
    let mut changed = false;

    for (entry_point, _) in module.entry_points.iter() {
        let fn_node = rvsdg
            .get_function_node(entry_point)
            .expect("function not registered");
        let body_region = rvsdg[fn_node].expect_function().body_region();

        changed |= unifier.process_region(rvsdg, body_region);
    }

    changed
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::rvsdg::{ValueInput, ValueOutput};
    use crate::ty::{TY_DUMMY, TY_PREDICATE, TY_U32};
    use crate::{FnSig, Function, Symbol};

    #[test]
    fn unifiable_meet() {
        let constant_1 = Unifiable::Constant(ScalarConstant::U32(1));
        let constant_2 = Unifiable::Constant(ScalarConstant::U32(2));
        let argument_0 = Unifiable::Argument {
            index: 0,
            resolved_constant: None,
        };
        let argument_0_constant_1 = Unifiable::Argument {
            index: 0,
            resolved_constant: Some(ScalarConstant::U32(1)),
        };
        let argument_0_constant_2 = Unifiable::Argument {
            index: 0,
            resolved_constant: Some(ScalarConstant::U32(2)),
        };
        let argument_1_constant_1 = Unifiable::Argument {
            index: 1,
            resolved_constant: Some(ScalarConstant::U32(1)),
        };

        assert_eq!(Unifiable::Unknown.meet(argument_0), argument_0);
        assert_eq!(argument_0.meet(Unifiable::Unknown), argument_0);

        assert_eq!(
            Unifiable::Impossible.meet(argument_0),
            Unifiable::Impossible
        );
        assert_eq!(
            argument_0.meet(Unifiable::Impossible),
            Unifiable::Impossible
        );

        assert_eq!(constant_1.meet(constant_1), constant_1);
        assert_eq!(constant_1.meet(constant_2), Unifiable::Impossible);

        assert_eq!(
            argument_0_constant_1.meet(argument_0_constant_1),
            argument_0_constant_1
        );
        assert_eq!(argument_0_constant_1.meet(argument_0), argument_0);
        assert_eq!(
            argument_0_constant_1.meet(argument_0_constant_2),
            argument_0
        );

        assert_eq!(
            argument_0_constant_1.meet(argument_1_constant_1),
            constant_1
        );
        assert_eq!(argument_0_constant_1.meet(constant_1), constant_1);
        assert_eq!(constant_1.meet(argument_0_constant_1), constant_1);

        assert_eq!(argument_0.meet(constant_1), Unifiable::Impossible);
        assert_eq!(constant_1.meet(argument_0), Unifiable::Impossible);
        assert_eq!(
            constant_1.meet(argument_0_constant_2),
            Unifiable::Impossible
        );

        assert_eq!(
            Unifiable::Impossible.meet(constant_1),
            Unifiable::Impossible
        );
    }

    #[test]
    fn unifies_constant_and_fallback_results() {
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

        let selector = rvsdg.add_const_predicate(body, 0);
        let switch = rvsdg.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, selector, 0)],
            vec![ValueOutput::new(TY_U32)],
            None,
        );
        let branch_0 = rvsdg.add_switch_branch(switch);
        let constant = rvsdg.add_const_u32(branch_0, 1);

        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: constant,
                output: 0,
            },
        );

        let branch_1 = rvsdg.add_switch_branch(switch);
        let fallback = rvsdg.add_const_fallback(branch_1, TY_U32);

        rvsdg.reconnect_region_result(
            branch_1,
            0,
            ValueOrigin::Output {
                producer: fallback,
                output: 0,
            },
        );

        let mut unifier = SwitchFallbackUnifier::new();

        assert!(unifier.process_region(&mut rvsdg, body));

        let ValueOrigin::Output {
            producer: replacement,
            output: 0,
        } = rvsdg[branch_1].value_results()[0].origin
        else {
            panic!("expected the fallback result to connect to a constant node");
        };

        assert_eq!(rvsdg[replacement].expect_const_u32().value(), 1);
    }

    #[test]
    fn unifies_shared_argument_and_fallback_results() {
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

        let selector = rvsdg.add_const_predicate(body, 0);
        let input = rvsdg.add_const_u32(body, 7);
        let switch = rvsdg.add_switch(
            body,
            vec![
                ValueInput::output(TY_PREDICATE, selector, 0),
                ValueInput::output(TY_U32, input, 0),
            ],
            vec![ValueOutput::new(TY_U32)],
            None,
        );
        let branch_0 = rvsdg.add_switch_branch(switch);

        rvsdg.reconnect_region_result(branch_0, 0, ValueOrigin::Argument(0));

        let branch_1 = rvsdg.add_switch_branch(switch);
        let fallback = rvsdg.add_const_fallback(branch_1, TY_U32);

        rvsdg.reconnect_region_result(
            branch_1,
            0,
            ValueOrigin::Output {
                producer: fallback,
                output: 0,
            },
        );

        let mut unifier = SwitchFallbackUnifier::new();

        assert!(unifier.process_region(&mut rvsdg, body));

        assert_eq!(
            rvsdg[branch_1].value_results()[0].origin,
            ValueOrigin::Argument(0)
        );
    }

    #[test]
    fn unifies_direct_and_argument_resolved_constants() {
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

        let selector = rvsdg.add_const_predicate(body, 0);
        let argument_constant = rvsdg.add_const_u32(body, 7);
        let switch = rvsdg.add_switch(
            body,
            vec![
                ValueInput::output(TY_PREDICATE, selector, 0),
                ValueInput::output(TY_U32, argument_constant, 0),
            ],
            vec![ValueOutput::new(TY_U32)],
            None,
        );
        let branch_0 = rvsdg.add_switch_branch(switch);
        let direct_constant = rvsdg.add_const_u32(branch_0, 7);

        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: direct_constant,
                output: 0,
            },
        );

        let branch_1 = rvsdg.add_switch_branch(switch);

        rvsdg.reconnect_region_result(branch_1, 0, ValueOrigin::Argument(0));

        let branch_2 = rvsdg.add_switch_branch(switch);
        let fallback = rvsdg.add_const_fallback(branch_2, TY_U32);

        rvsdg.reconnect_region_result(
            branch_2,
            0,
            ValueOrigin::Output {
                producer: fallback,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(
            body,
            0,
            ValueOrigin::Output {
                producer: switch,
                output: 0,
            },
        );

        let mut unifier = SwitchFallbackUnifier::new();

        assert!(unifier.process_region(&mut rvsdg, body));

        let ValueOrigin::Output {
            producer: replacement,
            output: 0,
        } = rvsdg[branch_2].value_results()[0].origin
        else {
            panic!("expected the fallback result to connect to a constant node");
        };

        assert_eq!(rvsdg[replacement].expect_const_u32().value(), 7);
    }

    #[test]
    fn rejects_non_unifiable_result_sets() {
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

        let selector = rvsdg.add_const_predicate(body, 0);
        let switch = rvsdg.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, selector, 0)],
            vec![ValueOutput::new(TY_U32)],
            None,
        );
        let branch_0 = rvsdg.add_switch_branch(switch);
        let zero = rvsdg.add_const_u32(branch_0, 0);

        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: zero,
                output: 0,
            },
        );

        let branch_1 = rvsdg.add_switch_branch(switch);
        let one = rvsdg.add_const_u32(branch_1, 1);

        rvsdg.reconnect_region_result(
            branch_1,
            0,
            ValueOrigin::Output {
                producer: one,
                output: 0,
            },
        );

        let branch_2 = rvsdg.add_switch_branch(switch);
        let fallback = rvsdg.add_const_fallback(branch_2, TY_U32);

        rvsdg.reconnect_region_result(
            branch_2,
            0,
            ValueOrigin::Output {
                producer: fallback,
                output: 0,
            },
        );

        let mut unifier = SwitchFallbackUnifier::new();

        assert!(!unifier.process_region(&mut rvsdg, body));

        assert_eq!(
            rvsdg[branch_2].value_results()[0].origin,
            ValueOrigin::Output {
                producer: fallback,
                output: 0,
            }
        );
    }
}
