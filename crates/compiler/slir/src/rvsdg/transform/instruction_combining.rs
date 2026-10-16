use std::collections::VecDeque;

use crate::rvsdg::analyse::ValueKey;
use crate::rvsdg::visit::region_nodes::RegionNodesVisitor;
use crate::rvsdg::visit::reverse_value_flow::ReverseValueFlowVisitor;
use crate::rvsdg::{
    Connectivity, Node, NodeKind, Region, Rvsdg, SimpleNode, ValueInput, ValueOrigin, ValueUser,
    visit,
};
use crate::{Function, UnaryOperator};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum CandidateKind {
    OpUnaryNot,
    Switch,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Candidate {
    node: Node,
    kind: CandidateKind,
}

impl Candidate {
    fn try_from_node(rvsdg: &Rvsdg, node: Node) -> Option<Candidate> {
        use NodeKind::*;
        use SimpleNode::*;

        let kind = match rvsdg[node].kind() {
            Simple(OpUnary(op)) => match op.operator() {
                UnaryOperator::Not => CandidateKind::OpUnaryNot,
                _ => return None,
            },
            Switch(_) => CandidateKind::Switch,
            _ => return None,
        };

        Some(Candidate { node, kind })
    }
}

struct CandidateCollector<'a> {
    candidates: &'a mut VecDeque<Candidate>,
}

impl RegionNodesVisitor for CandidateCollector<'_> {
    fn visit_node(&mut self, rvsdg: &Rvsdg, node: Node) {
        if let Some(candidate) = Candidate::try_from_node(rvsdg, node) {
            self.candidates.push_back(candidate);
        }

        visit::region_nodes::visit_node(self, rvsdg, node);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum TraceStepKind {
    Switch,
    Loop,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct TraceStep {
    region: Region,
    region_owner: Node,
    kind: TraceStepKind,
}

struct TargetValueTracer<'a> {
    target: Option<(Node, u32)>,
    regions: &'a mut Vec<TraceStep>,
}

impl<'a> TargetValueTracer<'a> {
    fn new(regions: &'a mut Vec<TraceStep>) -> Self {
        Self {
            target: None,
            regions,
        }
    }

    fn trace(&mut self, rvsdg: &Rvsdg, target: ValueKey) -> Option<(Node, u32)> {
        self.regions.clear();
        self.target = None;

        self.visit_value_origin(rvsdg, target.region, target.origin);

        self.target
    }
}

impl ReverseValueFlowVisitor for TargetValueTracer<'_> {
    fn should_visit(&mut self, _region: Region, _origin: ValueOrigin) -> bool {
        true
    }

    fn visit_region_argument(&mut self, rvsdg: &Rvsdg, region: Region, argument: u32) {
        use NodeKind::*;

        let owner = rvsdg[region].owner();

        match rvsdg[owner].kind() {
            Switch(_) => self.regions.push(TraceStep {
                region,
                region_owner: owner,
                kind: TraceStepKind::Switch,
            }),
            Loop(loop_data) => {
                let loop_region = loop_data.loop_region();
                let result_origin =
                    rvsdg[loop_region].value_results()[argument as usize + 1].origin;
                let is_loop_invariant = result_origin == ValueOrigin::Argument(argument);

                if is_loop_invariant {
                    self.regions.push(TraceStep {
                        region,
                        region_owner: owner,
                        kind: TraceStepKind::Loop,
                    });
                } else {
                    return;
                }
            }
            _ => {}
        }

        visit::reverse_value_flow::visit_region_argument(self, rvsdg, region, argument);
    }

    fn visit_value_output(&mut self, rvsdg: &Rvsdg, node: Node, output: u32) {
        if rvsdg[node].is_value_proxy() {
            visit::reverse_value_flow::visit_value_output(self, rvsdg, node, output);
        } else {
            self.target = Some((node, output));
        }
    }
}

fn trace_target_value(
    trace: &mut Vec<TraceStep>,
    rvsdg: &Rvsdg,
    target: ValueKey,
) -> Option<(Node, u32)> {
    TargetValueTracer::new(trace).trace(rvsdg, target)
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct CombineResult {
    changed: bool,
    new_candidate: Option<Candidate>,
}

impl CombineResult {
    fn noop() -> Self {
        CombineResult {
            changed: false,
            new_candidate: None,
        }
    }
}

pub struct InstructionCombiner {
    queue: VecDeque<Candidate>,
    region_trace: Vec<TraceStep>,
}

impl InstructionCombiner {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            region_trace: Vec::new(),
        }
    }

    pub fn combine_in_fn(&mut self, rvsdg: &mut Rvsdg, function: Function) -> bool {
        let fn_node = rvsdg
            .get_function_node(function)
            .expect("function not registered");
        let body_region = rvsdg[fn_node].expect_function().body_region();

        CandidateCollector {
            candidates: &mut self.queue,
        }
        .visit_region(rvsdg, body_region);

        let mut changed = false;

        // For the time being we don't really care about the order; for the currently supported
        // node kinds, any order should produce the same result. If it becomes relevant in the
        // future, we could consider assigning priority numbers to different candidate kinds, and
        // then sorting the queue before processing (this would also require a secondary "staging"
        // queue to requeuing new candidates resulting from a combination; we would swap the
        // queues when the "active" queue is drained).
        while let Some(candidate) = self.queue.pop_front() {
            let res = self.try_combine_candidate(rvsdg, candidate);

            if let Some(candidate) = res.new_candidate {
                self.queue.push_back(candidate);
            }

            changed |= res.changed;
        }

        changed
    }

    fn try_combine_candidate(&mut self, rvsdg: &mut Rvsdg, candidate: Candidate) -> CombineResult {
        // Candidate may have already been combined into an earlier candidate, in which case there
        // is nothing left to do here.
        if !rvsdg.is_live_node(candidate.node) {
            return CombineResult::noop();
        }

        match candidate.kind {
            CandidateKind::OpUnaryNot => self.try_combine_op_unary_not(rvsdg, candidate.node),
            CandidateKind::Switch => self.try_combine_switch(rvsdg, candidate.node),
        }
    }

    fn try_combine_op_unary_not(&mut self, rvsdg: &mut Rvsdg, node: Node) -> CombineResult {
        use NodeKind::*;
        use SimpleNode::*;

        let Some((target_node, 0)) = trace_target_value(
            &mut self.region_trace,
            rvsdg,
            ValueKey::for_node_input(rvsdg, node, 0),
        ) else {
            return CombineResult::noop();
        };

        if let Simple(OpUnary(op)) = rvsdg[target_node].kind()
            && op.operator() == UnaryOperator::Not
        {
            let new_value = rvsdg[target_node].value_inputs()[0];
            let region = rvsdg[node].region();
            let new_origin = self.route_value(rvsdg, new_value);

            rvsdg.reconnect_value_users(
                region,
                ValueOrigin::Output {
                    producer: node,
                    output: 0,
                },
                new_origin,
            );

            rvsdg.remove_node(node);

            CombineResult {
                changed: true,
                new_candidate: None,
            }
        } else {
            CombineResult::noop()
        }
    }

    fn try_combine_switch(&mut self, rvsdg: &mut Rvsdg, node: Node) -> CombineResult {
        use NodeKind::*;
        use SimpleNode::*;

        let ValueOrigin::Output {
            producer: selector_producer,
            output: 0,
        } = rvsdg[node].value_inputs()[0].origin
        else {
            return CombineResult::noop();
        };

        if !rvsdg[selector_producer].is_op_bool_to_branch_selector() {
            return CombineResult::noop();
        }

        let Some((target_node, 0)) = trace_target_value(
            &mut self.region_trace,
            rvsdg,
            ValueKey::for_node_input(rvsdg, selector_producer, 0),
        ) else {
            return CombineResult::noop();
        };

        if let Simple(OpUnary(op)) = rvsdg[target_node].kind()
            && op.operator() == UnaryOperator::Not
        {
            let new_value = rvsdg[target_node].value_inputs()[0];
            let new_origin = self.route_value(rvsdg, new_value);

            // We assume that this transform runs after branch-selector-normalizaztion and that any
            // selector-producer has only a single user.
            assert_eq!(
                rvsdg[selector_producer].value_outputs()[0].users.len(),
                1,
                "expected RVSDG to be in predicate-normal-form (see branch-selector-normalization)"
            );

            assert_eq!(
                rvsdg[node].expect_switch().branches().len(),
                2,
                "a switch node with a boolean selector should have exactly 2 branches"
            );

            rvsdg.reconnect_value_input(selector_producer, 0, new_origin);
            rvsdg.permute_switch_branches(node, &[1, 0]);

            CombineResult {
                changed: true,

                // We requeue the switch node as a candidate, as we changed it rather than removed
                // it, and the new condition value may again be combinable.
                new_candidate: Some(Candidate {
                    node,
                    kind: CandidateKind::Switch,
                }),
            }
        } else {
            CombineResult::noop()
        }
    }

    fn route_value(&mut self, rvsdg: &mut Rvsdg, mut value: ValueInput) -> ValueOrigin {
        if self.region_trace.is_empty() {
            return value.origin;
        }

        for trace_step in self.region_trace.iter().copied().rev() {
            let outer_region = rvsdg[trace_step.region_owner].region();

            match trace_step.kind {
                TraceStepKind::Switch => {
                    if let Some(arg) = rvsdg
                        .value_users(outer_region, value.origin)
                        .iter()
                        .find_map(|user| {
                            if let &ValueUser::Input { consumer, input } = user
                                && consumer == trace_step.region_owner
                                && input != 0
                            {
                                Some(input - 1)
                            } else {
                                None
                            }
                        })
                    {
                        // The value is already available inside the region, reuse the existing
                        // argument.

                        value = ValueInput::argument(value.ty, arg);
                    } else {
                        let input = rvsdg.add_switch_input(trace_step.region_owner, value);

                        value = ValueInput::argument(value.ty, input - 1);
                    }
                }
                TraceStepKind::Loop => {
                    if let Some(arg) = rvsdg
                        .value_users(outer_region, value.origin)
                        .iter()
                        .find_map(|user| {
                            if let &ValueUser::Input { consumer, input } = user
                                && consumer == trace_step.region_owner
                            {
                                let result_origin = rvsdg[trace_step.region].value_results()
                                    [input as usize + 1]
                                    .origin;
                                let is_loop_invariant =
                                    result_origin == ValueOrigin::Argument(input);

                                if is_loop_invariant { Some(input) } else { None }
                            } else {
                                None
                            }
                        })
                    {
                        // The value is already available as a loop-invariant loop-value, reuse the
                        // existing argument

                        value = ValueInput::argument(value.ty, arg);
                    } else {
                        let input = rvsdg.add_loop_input(trace_step.region_owner, value);

                        // The result that was created for the new loop-value is initially connected
                        // to a placeholder value. Reconnect it to the new argument to make it into
                        // a loop-invariant value.
                        rvsdg.reconnect_region_result(
                            trace_step.region,
                            input + 1,
                            ValueOrigin::Argument(input),
                        );

                        value = ValueInput::argument(value.ty, input);
                    }
                }
            }
        }

        value.origin
    }
}

#[cfg(test)]
mod tests {
    use super::InstructionCombiner;
    use crate::rvsdg::{Connectivity, Rvsdg, ValueInput, ValueOutput, ValueUser};
    use crate::ty::{TY_BOOL, TY_DUMMY, TY_PREDICATE};
    use crate::{FnArg, FnSig, Function, Module, Symbol, UnaryOperator, thin_set};

    #[test]
    fn combine_unary_not() {
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

        let not_0 =
            rvsdg.add_op_unary(region, UnaryOperator::Not, ValueInput::argument(TY_BOOL, 0));

        let not_1 = rvsdg.add_op_unary(
            region,
            UnaryOperator::Not,
            ValueInput::output(TY_BOOL, not_0, 0),
        );
        rvsdg.reconnect_region_result(region, 0, ValueInput::output(TY_BOOL, not_1, 0).origin);

        let mut optimizer = InstructionCombiner::new();

        assert!(optimizer.combine_in_fn(&mut rvsdg, function));

        assert!(!rvsdg.is_live_node(not_1));
        assert_eq!(
            rvsdg[region].value_results()[0],
            ValueInput::argument(TY_BOOL, 0)
        );
        assert!(rvsdg[not_0].value_outputs()[0].users.is_empty());
        assert_eq!(
            rvsdg[region].value_arguments()[0].users,
            thin_set![
                ValueUser::Input {
                    consumer: not_0,
                    input: 0,
                },
                ValueUser::Result(0)
            ]
        );

        assert!(!optimizer.combine_in_fn(&mut rvsdg, function));
    }

    #[test]
    fn combine_switch_with_negated_boolean_selector() {
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

        let not_0 =
            rvsdg.add_op_unary(region, UnaryOperator::Not, ValueInput::argument(TY_BOOL, 0));

        let selector =
            rvsdg.add_op_bool_to_branch_selector(region, ValueInput::output(TY_BOOL, not_0, 0));
        let switch = rvsdg.add_switch(
            region,
            vec![ValueInput::output(TY_PREDICATE, selector, 0)],
            vec![ValueOutput::new(TY_BOOL)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch);

        let const_true = rvsdg.add_const_bool(branch_0, true);
        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueInput::output(TY_BOOL, const_true, 0).origin,
        );

        let branch_1 = rvsdg.add_switch_branch(switch);

        let const_false = rvsdg.add_const_bool(branch_1, false);
        rvsdg.reconnect_region_result(
            branch_1,
            0,
            ValueInput::output(TY_BOOL, const_false, 0).origin,
        );

        rvsdg.reconnect_region_result(region, 0, ValueInput::output(TY_BOOL, switch, 0).origin);

        let mut optimizer = InstructionCombiner::new();

        assert!(optimizer.combine_in_fn(&mut rvsdg, function));

        assert_eq!(
            rvsdg[selector].value_inputs(),
            &[ValueInput::argument(TY_BOOL, 0)]
        );
        assert_eq!(
            rvsdg[switch].value_inputs(),
            &[ValueInput::output(TY_PREDICATE, selector, 0)]
        );
        assert_eq!(
            rvsdg[switch].expect_switch().branches(),
            &[branch_1, branch_0]
        );

        assert_eq!(
            rvsdg[branch_0].value_results()[0],
            ValueInput::output(TY_BOOL, const_true, 0)
        );
        assert_eq!(
            rvsdg[branch_1].value_results()[0],
            ValueInput::output(TY_BOOL, const_false, 0)
        );

        assert!(rvsdg[not_0].value_outputs()[0].users.is_empty());
        assert_eq!(
            rvsdg[selector].value_outputs()[0].users,
            thin_set![ValueUser::Input {
                consumer: switch,
                input: 0,
            }]
        );
        assert_eq!(
            rvsdg[region].value_arguments()[0].users,
            thin_set![
                ValueUser::Input {
                    consumer: not_0,
                    input: 0,
                },
                ValueUser::Input {
                    consumer: selector,
                    input: 0,
                }
            ]
        );
        assert_eq!(
            rvsdg[region].value_results()[0],
            ValueInput::output(TY_BOOL, switch, 0)
        );

        assert!(!optimizer.combine_in_fn(&mut rvsdg, function));
    }

    #[test]
    fn combines_through_switch_branch_argument() {
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

        let (_, region) = rvsdg.register_function(&module, function, []);

        let not_0 =
            rvsdg.add_op_unary(region, UnaryOperator::Not, ValueInput::argument(TY_BOOL, 0));

        let selector =
            rvsdg.add_op_bool_to_branch_selector(region, ValueInput::argument(TY_BOOL, 1));
        let switch = rvsdg.add_switch(
            region,
            vec![
                ValueInput::output(TY_PREDICATE, selector, 0),
                ValueInput::output(TY_BOOL, not_0, 0),
            ],
            vec![ValueOutput::new(TY_BOOL)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch);

        let not_1 = rvsdg.add_op_unary(
            branch_0,
            UnaryOperator::Not,
            ValueInput::argument(TY_BOOL, 0),
        );
        rvsdg.reconnect_region_result(branch_0, 0, ValueInput::output(TY_BOOL, not_1, 0).origin);

        let branch_1 = rvsdg.add_switch_branch(switch);
        rvsdg.reconnect_region_result(branch_1, 0, ValueInput::argument(TY_BOOL, 0).origin);

        rvsdg.reconnect_region_result(region, 0, ValueInput::output(TY_BOOL, switch, 0).origin);

        let mut optimizer = InstructionCombiner::new();

        assert!(optimizer.combine_in_fn(&mut rvsdg, function));

        assert!(!rvsdg.is_live_node(not_1));
        assert!(rvsdg.is_live_node(not_0));

        assert_eq!(
            rvsdg[switch].value_inputs(),
            &[
                ValueInput::output(TY_PREDICATE, selector, 0),
                ValueInput::output(TY_BOOL, not_0, 0),
                ValueInput::argument(TY_BOOL, 0),
            ]
        );

        assert_eq!(
            rvsdg[branch_0].value_results()[0],
            ValueInput::argument(TY_BOOL, 1)
        );
        assert!(rvsdg[branch_0].value_arguments()[0].users.is_empty());
        assert_eq!(
            rvsdg[branch_0].value_arguments()[1].users,
            thin_set![ValueUser::Result(0)]
        );

        assert_eq!(
            rvsdg[branch_1].value_results()[0],
            ValueInput::argument(TY_BOOL, 0)
        );
        assert_eq!(
            rvsdg[branch_1].value_arguments()[0].users,
            thin_set![ValueUser::Result(0)]
        );
        assert!(rvsdg[branch_1].value_arguments()[1].users.is_empty());

        assert_eq!(
            rvsdg[not_0].value_outputs()[0].users,
            thin_set![ValueUser::Input {
                consumer: switch,
                input: 1,
            }]
        );
        assert_eq!(
            rvsdg[region].value_arguments()[0].users,
            thin_set![
                ValueUser::Input {
                    consumer: not_0,
                    input: 0,
                },
                ValueUser::Input {
                    consumer: switch,
                    input: 2,
                }
            ]
        );

        assert!(!optimizer.combine_in_fn(&mut rvsdg, function));
    }

    #[test]
    fn combines_through_loop_invariant_loop_region_argument() {
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

        let not_0 =
            rvsdg.add_op_unary(region, UnaryOperator::Not, ValueInput::argument(TY_BOOL, 0));

        let (loop_node, loop_body) =
            rvsdg.add_loop(region, vec![ValueInput::output(TY_BOOL, not_0, 0)], None);
        rvsdg.reconnect_region_result(loop_body, 1, ValueInput::argument(TY_BOOL, 0).origin);

        let not_1 = rvsdg.add_op_unary(
            loop_body,
            UnaryOperator::Not,
            ValueInput::argument(TY_BOOL, 0),
        );
        rvsdg.reconnect_region_result(loop_body, 0, ValueInput::output(TY_BOOL, not_1, 0).origin);

        rvsdg.reconnect_region_result(region, 0, ValueInput::output(TY_BOOL, loop_node, 0).origin);

        let mut optimizer = InstructionCombiner::new();

        assert!(optimizer.combine_in_fn(&mut rvsdg, function));

        assert!(!rvsdg.is_live_node(not_1));
        assert!(rvsdg.is_live_node(not_0));

        assert_eq!(
            rvsdg[loop_node].value_inputs(),
            &[
                ValueInput::output(TY_BOOL, not_0, 0),
                ValueInput::argument(TY_BOOL, 0),
            ]
        );

        assert_eq!(
            rvsdg[loop_body].value_results(),
            &[
                ValueInput::argument(TY_BOOL, 1),
                ValueInput::argument(TY_BOOL, 0),
                ValueInput::argument(TY_BOOL, 1),
            ]
        );
        assert_eq!(
            rvsdg[loop_body].value_arguments()[0].users,
            thin_set![ValueUser::Result(1)]
        );
        assert_eq!(
            rvsdg[loop_body].value_arguments()[1].users,
            thin_set![ValueUser::Result(0), ValueUser::Result(2)]
        );

        assert_eq!(
            rvsdg[not_0].value_outputs()[0].users,
            thin_set![ValueUser::Input {
                consumer: loop_node,
                input: 0,
            }]
        );
        assert_eq!(
            rvsdg[region].value_arguments()[0].users,
            thin_set![
                ValueUser::Input {
                    consumer: not_0,
                    input: 0,
                },
                ValueUser::Input {
                    consumer: loop_node,
                    input: 1,
                }
            ]
        );

        assert!(!optimizer.combine_in_fn(&mut rvsdg, function));
    }

    #[test]
    fn does_not_combine_through_non_loop_invariant_loop_region_argument() {
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

        let not_0 =
            rvsdg.add_op_unary(region, UnaryOperator::Not, ValueInput::argument(TY_BOOL, 0));

        let (loop_node, loop_body) =
            rvsdg.add_loop(region, vec![ValueInput::output(TY_BOOL, not_0, 0)], None);

        let not_1 = rvsdg.add_op_unary(
            loop_body,
            UnaryOperator::Not,
            ValueInput::argument(TY_BOOL, 0),
        );
        rvsdg.reconnect_region_result(loop_body, 0, ValueInput::output(TY_BOOL, not_1, 0).origin);
        rvsdg.reconnect_region_result(loop_body, 1, ValueInput::output(TY_BOOL, not_1, 0).origin);

        rvsdg.reconnect_region_result(region, 0, ValueInput::output(TY_BOOL, loop_node, 0).origin);

        let mut optimizer = InstructionCombiner::new();

        assert!(!optimizer.combine_in_fn(&mut rvsdg, function));

        assert!(rvsdg.is_live_node(not_0));
        assert!(rvsdg.is_live_node(not_1));

        assert_eq!(
            rvsdg[not_0].value_inputs(),
            &[ValueInput::argument(TY_BOOL, 0)]
        );
        assert_eq!(
            rvsdg[not_1].value_inputs(),
            &[ValueInput::argument(TY_BOOL, 0)]
        );
        assert_eq!(
            rvsdg[loop_node].value_inputs(),
            &[ValueInput::output(TY_BOOL, not_0, 0)]
        );

        assert_eq!(
            rvsdg[loop_body].value_results(),
            &[
                ValueInput::output(TY_BOOL, not_1, 0),
                ValueInput::output(TY_BOOL, not_1, 0),
            ]
        );
        assert_eq!(rvsdg[loop_body].value_arguments().len(), 1);
        assert_eq!(
            rvsdg[loop_body].value_arguments()[0].users,
            thin_set![ValueUser::Input {
                consumer: not_1,
                input: 0,
            }]
        );

        assert_eq!(
            rvsdg[not_0].value_outputs()[0].users,
            thin_set![ValueUser::Input {
                consumer: loop_node,
                input: 0,
            }]
        );
        assert_eq!(
            rvsdg[not_1].value_outputs()[0].users,
            thin_set![ValueUser::Result(0), ValueUser::Result(1)]
        );

        assert_eq!(
            rvsdg[region].value_arguments()[0].users,
            thin_set![ValueUser::Input {
                consumer: not_0,
                input: 0,
            }]
        );
        assert_eq!(
            rvsdg[region].value_results()[0],
            ValueInput::output(TY_BOOL, loop_node, 0)
        );
    }
}
