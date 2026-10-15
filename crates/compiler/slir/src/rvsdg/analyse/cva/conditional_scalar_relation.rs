//! Provides analysis to determine the relation between a boolean scalar value and the branch
//! selector of an upstream switch node.
//!
//! See [query_boolean_relation] for details.

use crate::rvsdg::Node;
use crate::rvsdg::analyse::ValueKey;
use crate::rvsdg::analyse::cva::domain::AbstractBool;
use crate::rvsdg::analyse::cva::{
    AbstractValue, AssumptionResult, ContextId, ContextualValueAnalysis,
};

/// The relation between a boolean scalar value and the boolean input controlling a 2-branch switch.
///
/// See also [query_boolean_relation].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BooleanRelation {
    /// The scalar value is equal to the switch's boolean input (`v == p`).
    Input,
    /// The scalar value is equal to the complement of the switch's boolean input (`v == !p`).
    Complement,
    /// The scalar value is a uniform constant boolean across all reachable producer alternatives.
    Constant(bool),
}

/// Evaluates the boolean scalar identified by `v_key` under each alternative of a 2-branch `switch`
/// in `context`.
///
/// Returns:
///
/// - `Some(BooleanRelation::Constant(b))` if all reachable alternatives evaluate to `Const(b)`.
/// - `Some(BooleanRelation::Input)` if both branches `0` and `1` evaluate to `true` and `false`
///   respectively; the scalar value is equal to the switch's boolean branch-selector.
/// - `Some(BooleanRelation::Complement)` if both branches `0` and `1`  evaluate to `false` and
///   `true` respectively; the scalar value is the complement/inverse of the switch's boolean
///   branch-selector.
/// - `None` if assumptions fail, values are unconstrained/bottom/non-constant, or both branch
///   assumptions are contradictory.
pub fn query_boolean_relation(
    analysis: &mut ContextualValueAnalysis,
    v_key: ValueKey,
    switch: Node,
    context: ContextId,
) -> Option<BooleanRelation> {
    let mut branch_results = [None, None];

    for branch in 0..2 {
        match analysis.assume_branch(context, switch, branch) {
            AssumptionResult::Unknown => return None,
            AssumptionResult::Contradictory => {}
            AssumptionResult::Context(child) => match analysis.evaluate_value(v_key, child) {
                AbstractValue::Bool(AbstractBool::Const(b)) => {
                    branch_results[branch] = Some(b);
                }
                _ => return None,
            },
        }
    }

    match (branch_results[0], branch_results[1]) {
        (Some(b), None) | (None, Some(b)) => Some(BooleanRelation::Constant(b)),
        (Some(true), Some(true)) => Some(BooleanRelation::Constant(true)),
        (Some(false), Some(false)) => Some(BooleanRelation::Constant(false)),
        (Some(true), Some(false)) => Some(BooleanRelation::Input),
        (Some(false), Some(true)) => Some(BooleanRelation::Complement),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::rvsdg::{Rvsdg, ValueInput, ValueOrigin, ValueOutput};
    use crate::ty::{TY_BOOL, TY_DUMMY, TY_PREDICATE};
    use crate::{FnArg, FnSig, Function, Module, Symbol};

    #[test]
    fn returns_input() {
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
        let selector = graph.add_op_bool_to_branch_selector(body, ValueInput::argument(TY_BOOL, 0));
        let switch = graph.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, selector, 0)],
            vec![ValueOutput::new(TY_BOOL)],
            None,
        );

        let branch_0 = graph.add_switch_branch(switch);
        let value_0 = graph.add_const_bool(branch_0, true);
        graph.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: value_0,
                output: 0,
            },
        );

        let branch_1 = graph.add_switch_branch(switch);
        let value_1 = graph.add_const_bool(branch_1, false);
        graph.reconnect_region_result(
            branch_1,
            0,
            ValueOrigin::Output {
                producer: value_1,
                output: 0,
            },
        );

        let key = ValueKey::new(
            body,
            ValueOrigin::Output {
                producer: switch,
                output: 0,
            },
        );
        let mut analysis = ContextualValueAnalysis::new(&graph, function);
        let root = analysis.root_context();

        assert_eq!(
            query_boolean_relation(&mut analysis, key, switch, root),
            Some(BooleanRelation::Input)
        );
    }

    #[test]
    fn returns_complement() {
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
        let selector = graph.add_op_bool_to_branch_selector(body, ValueInput::argument(TY_BOOL, 0));
        let switch = graph.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, selector, 0)],
            vec![ValueOutput::new(TY_BOOL)],
            None,
        );

        let branch_0 = graph.add_switch_branch(switch);
        let value_0 = graph.add_const_bool(branch_0, false);
        graph.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: value_0,
                output: 0,
            },
        );

        let branch_1 = graph.add_switch_branch(switch);
        let value_1 = graph.add_const_bool(branch_1, true);
        graph.reconnect_region_result(
            branch_1,
            0,
            ValueOrigin::Output {
                producer: value_1,
                output: 0,
            },
        );

        let key = ValueKey::new(
            body,
            ValueOrigin::Output {
                producer: switch,
                output: 0,
            },
        );
        let mut analysis = ContextualValueAnalysis::new(&graph, function);
        let root = analysis.root_context();

        assert_eq!(
            query_boolean_relation(&mut analysis, key, switch, root),
            Some(BooleanRelation::Complement)
        );
    }

    #[test]
    fn returns_same_constant_in_both_branches() {
        for value in [true, false] {
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
            let selector =
                graph.add_op_bool_to_branch_selector(body, ValueInput::argument(TY_BOOL, 0));
            let switch = graph.add_switch(
                body,
                vec![ValueInput::output(TY_PREDICATE, selector, 0)],
                vec![ValueOutput::new(TY_BOOL)],
                None,
            );

            let branch_0 = graph.add_switch_branch(switch);
            let value_0 = graph.add_const_bool(branch_0, value);
            graph.reconnect_region_result(
                branch_0,
                0,
                ValueOrigin::Output {
                    producer: value_0,
                    output: 0,
                },
            );

            let branch_1 = graph.add_switch_branch(switch);
            let value_1 = graph.add_const_bool(branch_1, value);
            graph.reconnect_region_result(
                branch_1,
                0,
                ValueOrigin::Output {
                    producer: value_1,
                    output: 0,
                },
            );

            let key = ValueKey::new(
                body,
                ValueOrigin::Output {
                    producer: switch,
                    output: 0,
                },
            );
            let mut analysis = ContextualValueAnalysis::new(&graph, function);
            let root = analysis.root_context();

            assert_eq!(
                query_boolean_relation(&mut analysis, key, switch, root),
                Some(BooleanRelation::Constant(value))
            );
        }
    }

    #[test]
    fn returns_constant_when_only_one_branch_is_reachable() {
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
        let condition = graph.add_const_bool(body, true);
        let selector =
            graph.add_op_bool_to_branch_selector(body, ValueInput::output(TY_BOOL, condition, 0));
        let switch = graph.add_switch(
            body,
            vec![
                ValueInput::output(TY_PREDICATE, selector, 0),
                ValueInput::argument(TY_BOOL, 0),
            ],
            vec![ValueOutput::new(TY_BOOL)],
            None,
        );

        let branch_0 = graph.add_switch_branch(switch);
        let value_0 = graph.add_const_bool(branch_0, true);
        graph.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: value_0,
                output: 0,
            },
        );

        let branch_1 = graph.add_switch_branch(switch);
        graph.reconnect_region_result(branch_1, 0, ValueOrigin::Argument(0));

        let key = ValueKey::new(
            body,
            ValueOrigin::Output {
                producer: switch,
                output: 0,
            },
        );
        let mut analysis = ContextualValueAnalysis::new(&graph, function);
        let root = analysis.root_context();

        assert_eq!(
            query_boolean_relation(&mut analysis, key, switch, root),
            Some(BooleanRelation::Constant(true))
        );
    }

    #[test]
    fn returns_none_when_a_reachable_branch_is_unconstrained() {
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
                        ty: TY_BOOL,
                        shader_io_binding: None,
                    },
                ],
                ret_ty: None,
            },
        );

        let mut graph = Rvsdg::new(module.ty.clone());
        let (_, body) = graph.register_function(&module, function, iter::empty());
        let selector = graph.add_op_bool_to_branch_selector(body, ValueInput::argument(TY_BOOL, 0));
        let switch = graph.add_switch(
            body,
            vec![
                ValueInput::output(TY_PREDICATE, selector, 0),
                ValueInput::argument(TY_BOOL, 1),
            ],
            vec![ValueOutput::new(TY_BOOL)],
            None,
        );

        let branch_0 = graph.add_switch_branch(switch);
        let value_0 = graph.add_const_bool(branch_0, true);
        graph.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: value_0,
                output: 0,
            },
        );

        let branch_1 = graph.add_switch_branch(switch);
        graph.reconnect_region_result(branch_1, 0, ValueOrigin::Argument(0));

        let key = ValueKey::new(
            body,
            ValueOrigin::Output {
                producer: switch,
                output: 0,
            },
        );
        let mut analysis = ContextualValueAnalysis::new(&graph, function);
        let root = analysis.root_context();

        assert_eq!(
            query_boolean_relation(&mut analysis, key, switch, root),
            None
        );
    }

    #[test]
    fn returns_none_when_neither_branch_is_reachable() {
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
                        ty: TY_BOOL,
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
                ValueInput::argument(TY_BOOL, 1),
            ],
            Vec::new(),
            None,
        );

        let outer_branch_0 = graph.add_switch_branch(outer_switch);
        let inner_selector =
            graph.add_op_bool_to_branch_selector(outer_branch_0, ValueInput::argument(TY_BOOL, 0));
        let inner_switch = graph.add_switch(
            outer_branch_0,
            vec![ValueInput::output(TY_PREDICATE, inner_selector, 0)],
            vec![ValueOutput::new(TY_BOOL)],
            None,
        );

        let inner_branch_0 = graph.add_switch_branch(inner_switch);
        let inner_value_0 = graph.add_const_bool(inner_branch_0, true);
        graph.reconnect_region_result(
            inner_branch_0,
            0,
            ValueOrigin::Output {
                producer: inner_value_0,
                output: 0,
            },
        );

        let inner_branch_1 = graph.add_switch_branch(inner_switch);
        let inner_value_1 = graph.add_const_bool(inner_branch_1, true);
        graph.reconnect_region_result(
            inner_branch_1,
            0,
            ValueOrigin::Output {
                producer: inner_value_1,
                output: 0,
            },
        );

        graph.add_switch_branch(outer_switch);

        let key = ValueKey::new(
            outer_branch_0,
            ValueOrigin::Output {
                producer: inner_switch,
                output: 0,
            },
        );
        let mut analysis = ContextualValueAnalysis::new(&graph, function);
        let root = analysis.root_context();
        let context = analysis
            .assume_branch(root, outer_switch, 1)
            .expect_context();

        assert_eq!(
            query_boolean_relation(&mut analysis, key, inner_switch, context),
            None
        );
    }
}
