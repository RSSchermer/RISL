//! This module implements the branch-selector normalization transform.
//!
//! Branch-selector normalization ensures that every [SwitchNode][0] connects directly (without
//! intermediate connections) to a node that produces its branch-selector, and that the
//! [SwitchNode][0] is this node's only user. After [dead_value_elimination][1], this achieves an
//! RVSDG that is in "predicate continuation form", as described by [Bahmann et al., 2015][2].
//!
//! Predicate continuation form simplifies follow-up transformations, such as
//! [conditional_ub_elimination][3]. It also simplifies the [rvsdg_to_scf][4] conversion.
//!
//! [0]: crate::rvsdg::SwitchNode
//! [1]: crate::rvsdg::transform::dead_value_elimination
//! [2]: https://doi.org/10.1145/269326
//! [3]: crate::rvsdg::transform::conditional_ub_elimination
//! [4]: crate::rvsdg_to_scf

use std::mem;

use rustc_hash::FxHashMap;

use crate::Module;
use crate::rvsdg::visit::region_nodes::RegionNodesVisitor;
use crate::rvsdg::{
    Connectivity, Node, NodeKind, Region, Rvsdg, SimpleNode, ValueInput, ValueOrigin, visit,
};
use crate::ty::{TY_BOOL, TY_U32};

#[derive(Clone, Debug)]
enum ResolvedValue {
    Fallback,
    Bool(ValueInput),
    Case { value: ValueInput, cases: Vec<u32> },
}

impl ResolvedValue {
    fn value_input(&self) -> Option<ValueInput> {
        match self {
            ResolvedValue::Fallback => None,
            ResolvedValue::Bool(value) => Some(*value),
            ResolvedValue::Case { value, .. } => Some(*value),
        }
    }

    fn with_input(self, value: ValueInput) -> Self {
        match self {
            ResolvedValue::Fallback => ResolvedValue::Fallback,
            ResolvedValue::Bool(_) => ResolvedValue::Bool(value),
            ResolvedValue::Case { cases, .. } => ResolvedValue::Case { value, cases },
        }
    }

    fn fold(self, other: ResolvedValue) -> ResolvedValue {
        match (self, other) {
            (ResolvedValue::Fallback, value) => value,
            (value, ResolvedValue::Fallback) => value,
            (ResolvedValue::Bool(a), ResolvedValue::Bool(_)) => ResolvedValue::Bool(a),
            (
                ResolvedValue::Case {
                    value: a,
                    cases: a_cases,
                },
                ResolvedValue::Case { cases: b_cases, .. },
            ) => {
                if a_cases == b_cases {
                    ResolvedValue::Case {
                        value: a,
                        cases: a_cases,
                    }
                } else {
                    ResolvedValue::Fallback
                }
            }
            _ => ResolvedValue::Fallback,
        }
    }
}

pub struct BranchSelectorNormalizer {
    cache: FxHashMap<(Region, ValueOrigin), ResolvedValue>,
    collector: SwitchCollector,
}

impl BranchSelectorNormalizer {
    pub fn new() -> Self {
        Self {
            cache: FxHashMap::default(),
            collector: SwitchCollector::new(),
        }
    }

    pub fn transform_region(&mut self, rvsdg: &mut Rvsdg, region: Region) {
        self.collector.collect(rvsdg, region);

        let mut switches = mem::replace(&mut self.collector.switches, Vec::new());

        for switch in switches.drain(..) {
            self.normalize_switch_branch_selector(rvsdg, switch);
        }

        // Place the switches Vec back in the collector so that its allocation can be reused for
        // the next region.
        mem::swap(&mut self.collector.switches, &mut switches);
    }

    fn resolve_value(
        &mut self,
        rvsdg: &mut Rvsdg,
        region: Region,
        origin: ValueOrigin,
    ) -> ResolvedValue {
        if let Some(resolved) = self.cache.get(&(region, origin)) {
            return resolved.clone();
        }

        let resolved = self.route_value(rvsdg, region, origin);

        self.cache.insert((region, origin), resolved.clone());

        resolved
    }

    fn route_value(
        &mut self,
        rvsdg: &mut Rvsdg,
        region: Region,
        origin: ValueOrigin,
    ) -> ResolvedValue {
        match origin {
            ValueOrigin::Output { producer, output } => {
                self.route_node_output(rvsdg, producer, output)
            }
            ValueOrigin::Argument(arg) => self.route_argument(rvsdg, region, arg),
        }
    }

    fn route_node_output(&mut self, rvsdg: &mut Rvsdg, node: Node, output: u32) -> ResolvedValue {
        match rvsdg[node].kind() {
            NodeKind::Simple(SimpleNode::OpBoolToBranchSelector(_)) => {
                self.route_op_bool_to_branch_selector_output(rvsdg, node)
            }
            NodeKind::Simple(SimpleNode::OpCaseToBranchSelector(_)) => {
                self.route_op_case_to_branch_selector_output(rvsdg, node)
            }
            NodeKind::Simple(SimpleNode::ConstFallback(_)) => ResolvedValue::Fallback,
            NodeKind::Switch(_) => self.route_switch_output(rvsdg, node, output),
            NodeKind::Loop(_) => self.route_loop_output(rvsdg, node, output),
            kind => panic!(
                "predicate origin is not a normalization node, switch, or loop: node={:?}, \
                kind={:?}",
                node, kind
            ),
        }
    }

    fn route_op_bool_to_branch_selector_output(
        &self,
        rvsdg: &mut Rvsdg,
        node: Node,
    ) -> ResolvedValue {
        ResolvedValue::Bool(rvsdg[node].value_inputs()[0])
    }

    fn route_op_case_to_branch_selector_output(
        &self,
        rvsdg: &mut Rvsdg,
        node: Node,
    ) -> ResolvedValue {
        let op = rvsdg[node].expect_op_case_to_branch_selector();

        ResolvedValue::Case {
            value: rvsdg[node].value_inputs()[0],
            cases: op.cases().to_vec(),
        }
    }

    fn route_switch_output(&mut self, rvsdg: &mut Rvsdg, node: Node, output: u32) -> ResolvedValue {
        let branch_count = rvsdg[node].expect_switch().branches().len();
        let mut resolved_origins = Vec::with_capacity(branch_count);
        let mut folded_value = ResolvedValue::Fallback;

        for i in 0..branch_count {
            let branch_region = rvsdg[node].expect_switch().branches()[i];
            let branch_origin = rvsdg[branch_region].value_results()[output as usize].origin;
            let resolved = self.resolve_value(rvsdg, branch_region, branch_origin);

            resolved_origins.push(resolved.value_input().map(|v| v.origin));
            folded_value = folded_value.fold(resolved);
        }

        let value_ty = match folded_value {
            ResolvedValue::Bool(_) => TY_BOOL,
            ResolvedValue::Case { .. } => TY_U32,
            ResolvedValue::Fallback => {
                panic!("folded value must resolve to something other than a fallback value")
            }
        };

        let new_output = rvsdg.add_switch_output(node, value_ty);

        for i in 0..branch_count {
            let branch_region = rvsdg[node].expect_switch().branches()[i];

            let origin = resolved_origins[i].unwrap_or_else(|| {
                // Apparently, the original predicate result for this branch connects to a fallback
                // node. We assume the value will never actually be used and may provide any value
                // as a valid value.

                let node = if value_ty == TY_BOOL {
                    rvsdg.add_const_bool(branch_region, false)
                } else {
                    rvsdg.add_const_u32(branch_region, 0)
                };

                ValueOrigin::Output {
                    producer: node,
                    output: 0,
                }
            });

            rvsdg.reconnect_region_result(branch_region, new_output, origin);
        }

        folded_value.with_input(ValueInput::output(value_ty, node, new_output))
    }

    fn route_loop_output(&mut self, rvsdg: &mut Rvsdg, node: Node, output: u32) -> ResolvedValue {
        let loop_node = rvsdg[node].expect_loop();
        let loop_region = loop_node.loop_region();

        self.route_loop_value(rvsdg, loop_region, node, output);

        let outer_region = rvsdg[node].region();
        let output_origin = ValueOrigin::Output {
            producer: node,
            output,
        };

        self.cache
            .get(&(outer_region, output_origin))
            .expect("value should be cached by route_loop_value")
            .clone()
    }

    fn route_argument(&mut self, rvsdg: &mut Rvsdg, region: Region, arg: u32) -> ResolvedValue {
        let owner = rvsdg[region].owner();

        match rvsdg[owner].kind() {
            NodeKind::Switch(_) => self.route_switch_argument(rvsdg, owner, arg),
            NodeKind::Loop(_) => self.route_loop_argument(rvsdg, region, owner, arg),
            kind => panic!(
                "predicate origin is a region argument but owner is not a switch or loop: \
                owner={:?}, arg={}, kind={:?}",
                owner, arg, kind
            ),
        }
    }

    fn route_switch_argument(&mut self, rvsdg: &mut Rvsdg, node: Node, arg: u32) -> ResolvedValue {
        let outer_region = rvsdg[node].region();
        let input = arg + 1;
        let input_origin = rvsdg[node].value_inputs()[input as usize].origin;
        let resolved_value = self.resolve_value(rvsdg, outer_region, input_origin);
        let value_input = resolved_value
            .value_input()
            .expect("a switch input should not connect a fallback predicate");

        let new_input = rvsdg.add_switch_input(node, value_input);
        let new_arg = new_input - 1;

        let branch_count = rvsdg[node].expect_switch().branches().len();

        let predicate_arg = ValueOrigin::Argument(arg);
        let branch_resolved_value =
            resolved_value.with_input(ValueInput::argument(value_input.ty, new_arg));

        for i in 0..branch_count {
            let branch_region = rvsdg[node].expect_switch().branches()[i];
            self.cache.insert(
                (branch_region, predicate_arg),
                branch_resolved_value.clone(),
            );
        }

        branch_resolved_value
    }

    fn route_loop_argument(
        &mut self,
        rvsdg: &mut Rvsdg,
        loop_region: Region,
        node: Node,
        arg: u32,
    ) -> ResolvedValue {
        self.route_loop_value(rvsdg, loop_region, node, arg);

        let arg = ValueOrigin::Argument(arg);

        self.cache
            .get(&(loop_region, arg))
            .expect("value should be cached by route_loop_value")
            .clone()
    }

    fn route_loop_value(
        &mut self,
        rvsdg: &mut Rvsdg,
        loop_region: Region,
        node: Node,
        loop_value_index: u32,
    ) {
        let outer_region = rvsdg[node].region();
        let input_origin = rvsdg[node].value_inputs()[loop_value_index as usize].origin;
        let input_resolved = self.resolve_value(rvsdg, outer_region, input_origin);
        let value_input = input_resolved
            .value_input()
            .expect("a loop input should not connect a fallback predicate");

        let new_index = rvsdg.add_loop_input(node, value_input);

        // Cache a new resolved value for the predicate argument
        let predicate_arg = ValueOrigin::Argument(loop_value_index);
        let resolved_arg = input_resolved
            .clone()
            .with_input(ValueInput::argument(value_input.ty, new_index));
        self.cache
            .insert((loop_region, predicate_arg), resolved_arg);

        // Cache a new resolved value for the predicate output
        let predicate_output = ValueOrigin::Output {
            producer: node,
            output: loop_value_index,
        };
        let resolved_output =
            input_resolved
                .clone()
                .with_input(ValueInput::output(value_input.ty, node, new_index));
        self.cache
            .insert((outer_region, predicate_output), resolved_output);

        // The first region result is the reentry condition. The loop-value results come after that.
        // Therefore, add 1.
        let result_origin =
            rvsdg[loop_region].value_results()[(loop_value_index + 1) as usize].origin;
        let result_resolved = self.resolve_value(rvsdg, loop_region, result_origin);
        let result_origin = result_resolved
            .value_input()
            .expect("a loop result should not connect a fallback predicate")
            .origin;

        if matches!(
            input_resolved.fold(result_resolved),
            ResolvedValue::Fallback
        ) {
            panic!("incompatible branch selector kinds at loop");
        }

        rvsdg.reconnect_region_result(loop_region, new_index + 1, result_origin);
    }

    fn normalize_switch_branch_selector(&mut self, rvsdg: &mut Rvsdg, switch_node: Node) {
        let region = rvsdg[switch_node].region();
        let branch_selector_input = rvsdg[switch_node].expect_switch().branch_selector();
        let origin = branch_selector_input.origin;

        let needs_normalization = match origin {
            ValueOrigin::Output { producer, output } => {
                let is_normalization_node = rvsdg[producer].is_op_bool_to_branch_selector()
                    || rvsdg[producer].is_op_case_to_branch_selector();

                if !is_normalization_node {
                    true
                } else {
                    rvsdg[producer].value_outputs()[output as usize].users.len() > 1
                }
            }
            ValueOrigin::Argument(_) => true,
        };

        if needs_normalization {
            let resolved = self.resolve_value(rvsdg, region, origin);

            let normalization_node = match resolved {
                ResolvedValue::Bool(value) => rvsdg.add_op_bool_to_branch_selector(region, value),
                ResolvedValue::Case { value, cases } => {
                    rvsdg.add_op_case_to_branch_selector(region, value, cases)
                }
                ResolvedValue::Fallback => panic!("cannot normalize a fallback value"),
            };

            let output_origin = ValueOrigin::Output {
                producer: normalization_node,
                output: 0,
            };

            rvsdg.reconnect_value_input(switch_node, 0, output_origin);
        }
    }
}

struct SwitchCollector {
    switches: Vec<Node>,
}

impl SwitchCollector {
    fn new() -> Self {
        Self {
            switches: Vec::new(),
        }
    }

    fn collect(&mut self, rvsdg: &Rvsdg, region: Region) {
        self.switches.clear();

        self.visit_region(rvsdg, region);
    }
}

impl RegionNodesVisitor for SwitchCollector {
    fn visit_node(&mut self, rvsdg: &Rvsdg, node: Node) {
        if rvsdg[node].is_switch() {
            self.switches.push(node);
        }

        visit::region_nodes::visit_node(self, rvsdg, node);
    }
}

pub fn transform_entry_points(module: &Module, rvsdg: &mut Rvsdg) {
    let mut normalizer = BranchSelectorNormalizer::new();

    for (entry_point, _) in module.entry_points.iter() {
        let fn_node = rvsdg
            .get_function_node(entry_point)
            .expect("function not registered");
        let body_region = rvsdg[fn_node].expect_function().body_region();

        normalizer.transform_region(rvsdg, body_region);
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::rvsdg::{ValueOrigin, ValueOutput};
    use crate::ty::{TY_BOOL, TY_DUMMY, TY_PREDICATE};
    use crate::{FnSig, Function, Symbol};

    #[test]
    fn test_normalize_switch_branch_selector() {
        let mut module = Module::new(Symbol::from_ref("test"));
        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let function = Function {
            name: Symbol::from_ref("f"),
            module: Symbol::from_ref("test"),
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

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let bool_value = rvsdg.add_const_bool(region, true);
        let bool_to_pred = rvsdg
            .add_op_bool_to_branch_selector(region, ValueInput::output(TY_BOOL, bool_value, 0));

        let switch_node = rvsdg.add_switch(
            region,
            vec![ValueInput::output(TY_PREDICATE, bool_to_pred, 0)],
            vec![],
            None,
        );
        rvsdg.add_switch_branch(switch_node);
        rvsdg.add_switch_branch(switch_node);

        // Add another user to the predicate to trigger normalization
        let another_switch = rvsdg.add_switch(
            region,
            vec![ValueInput::output(TY_PREDICATE, bool_to_pred, 0)],
            vec![],
            None,
        );
        rvsdg.add_switch_branch(another_switch);
        rvsdg.add_switch_branch(another_switch);

        // Verify that bool_to_pred has 2 users
        assert_eq!(rvsdg[bool_to_pred].value_outputs()[0].users.len(), 2);

        let mut normalizer = BranchSelectorNormalizer::new();

        normalizer.transform_region(&mut rvsdg, region);

        let branch_selector = rvsdg[switch_node].expect_switch().branch_selector();

        let ValueOrigin::Output {
            producer: switch_branch_selector,
            output: 0,
        } = branch_selector.origin
        else {
            panic!("expected output origin");
        };

        assert_ne!(switch_branch_selector, bool_to_pred);

        let switch_branch_selector =
            rvsdg[switch_branch_selector].expect_op_bool_to_branch_selector();

        assert_eq!(switch_branch_selector.value_outputs()[0].users.len(), 1);

        // Verify that the new branch selector uses the correct input value
        let ValueOrigin::Output {
            producer: switch_branch_selector_input_producer,
            output: 0,
        } = switch_branch_selector.value_inputs()[0].origin
        else {
            panic!("expected output origin for new branch selector input");
        };
        assert_eq!(switch_branch_selector_input_producer, bool_value);

        let branch_selector_2 = rvsdg[another_switch].expect_switch().branch_selector();

        let ValueOrigin::Output {
            producer: another_switch_branch_selector,
            output: 0,
        } = branch_selector_2.origin
        else {
            panic!("expected output origin");
        };

        // `another_switch` still uses `bool_to_pred` as it became the sole user after the other
        // switch was normalized.
        assert_eq!(another_switch_branch_selector, bool_to_pred);
        assert_eq!(
            rvsdg[another_switch_branch_selector].value_outputs()[0]
                .users
                .len(),
            1
        );
    }

    #[test]
    fn test_normalize_switch_routing_through_switch() {
        let mut module = Module::new(Symbol::from_ref("test"));
        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let function = Function {
            name: Symbol::from_ref("f"),
            module: Symbol::from_ref("test"),
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

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let bool_value = rvsdg.add_const_bool(region, true);
        let bool_to_pred = rvsdg
            .add_op_bool_to_branch_selector(region, ValueInput::output(TY_BOOL, bool_value, 0));

        // Switch_0 that returns the predicate
        let switch_0 = rvsdg.add_switch(
            region,
            vec![ValueInput::output(TY_PREDICATE, bool_to_pred, 0)],
            vec![ValueOutput::new(TY_PREDICATE)],
            None,
        );

        // Pass the predicate as an additional argument to switch_0
        rvsdg.add_switch_input(switch_0, ValueInput::output(TY_PREDICATE, bool_to_pred, 0));

        let switch_0_branch_0 = rvsdg.add_switch_branch(switch_0);
        rvsdg.reconnect_region_result(switch_0_branch_0, 0, ValueOrigin::Argument(0));

        let switch_0_branch_1 = rvsdg.add_switch_branch(switch_0);
        rvsdg.reconnect_region_result(switch_0_branch_1, 0, ValueOrigin::Argument(0));

        // Another switch that uses the predicate returned by switch_0
        let switch_1 = rvsdg.add_switch(
            region,
            vec![ValueInput::output(TY_PREDICATE, switch_0, 0)],
            vec![],
            None,
        );
        rvsdg.add_switch_branch(switch_1);
        rvsdg.add_switch_branch(switch_1);

        let mut normalizer = BranchSelectorNormalizer::new();

        normalizer.transform_region(&mut rvsdg, region);

        let branch_selector = rvsdg[switch_1].expect_switch().branch_selector();

        let ValueOrigin::Output {
            producer: switch_1_branch_selector,
            output: 0,
        } = branch_selector.origin
        else {
            panic!("expected output origin");
        };

        assert!(rvsdg[switch_1_branch_selector].is_op_bool_to_branch_selector());

        // The input to this to-branch-selector node should be a new output of switch_0
        let ValueOrigin::Output {
            producer: switch_1_branch_selector_prod,
            output: 1,
        } = rvsdg[switch_1_branch_selector].value_inputs()[0].origin
        else {
            panic!("expected node output");
        };
        assert_eq!(switch_1_branch_selector_prod, switch_0);

        // Check that the new output of switch_0 has type BOOL
        assert_eq!(rvsdg[switch_0].value_outputs()[1].ty, TY_BOOL);

        // Verify that the new results of switch_0 are connected to the correct argument origin
        let b0_result_origin = rvsdg[switch_0_branch_0].value_results()[1].origin;
        assert_eq!(b0_result_origin, ValueOrigin::Argument(1));
        let b1_result_origin = rvsdg[switch_0_branch_1].value_results()[1].origin;
        assert_eq!(b1_result_origin, ValueOrigin::Argument(1));

        // Verify that the new input of switch_0 is connected to the bool_value
        let ValueOrigin::Output {
            producer: switch_0_new_input_producer,
            output: 0,
        } = rvsdg[switch_0].value_inputs()[2].origin
        else {
            panic!("expected output origin for switch_0 new input");
        };
        assert_eq!(switch_0_new_input_producer, bool_value);
    }

    #[test]
    fn test_normalize_switch_routing_through_loop() {
        let mut module = Module::new(Symbol::from_ref("test"));
        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let function = Function {
            name: Symbol::from_ref("test_func"),
            module: Symbol::from_ref("test"),
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

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        // bool_value = true
        let bool_value = rvsdg.add_const_bool(region, true);

        // bool_to_pred = OpBoolToBranchSelector(bool_value)
        let bool_to_pred = rvsdg
            .add_op_bool_to_branch_selector(region, ValueInput::output(TY_BOOL, bool_value, 0));

        // loop_node: passes predicate through
        let (loop_node, loop_region) = rvsdg.add_loop(
            region,
            vec![ValueInput::output(TY_PREDICATE, bool_to_pred, 0)],
            None,
        );

        // Reentry condition: always false (run once)
        let false_const = rvsdg.add_const_bool(loop_region, false);
        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: false_const,
                output: 0,
            },
        );

        // Loop result: return the predicate argument
        rvsdg.reconnect_region_result(loop_region, 1, ValueOrigin::Argument(0));

        // Switch that uses the predicate returned by loop_node
        let switch_node = rvsdg.add_switch(
            region,
            vec![ValueInput::output(TY_PREDICATE, loop_node, 0)],
            vec![],
            None,
        );
        rvsdg.add_switch_branch(switch_node);
        rvsdg.add_switch_branch(switch_node);

        let mut normalizer = BranchSelectorNormalizer::new();
        normalizer.transform_region(&mut rvsdg, region);

        let branch_selector = rvsdg[switch_node].expect_switch().branch_selector();

        let ValueOrigin::Output {
            producer: switch_branch_selector,
            output: 0,
        } = branch_selector.origin
        else {
            panic!("expected output origin for switch branch selector");
        };

        assert!(rvsdg[switch_branch_selector].is_op_bool_to_branch_selector());

        // The input to this to-branch-selector node should be a new output of loop_node
        let origin = rvsdg[switch_branch_selector].value_inputs()[0].origin;
        let ValueOrigin::Output {
            producer: loop_node_prod,
            output: loop_output_index,
        } = origin
        else {
            panic!("expected loop output origin");
        };
        assert_eq!(loop_node_prod, loop_node);
        assert_eq!(loop_output_index, 1);

        // Check that the new output of loop_node has type BOOL
        assert_eq!(rvsdg[loop_node].value_outputs()[1].ty, TY_BOOL);

        // Verify that the new loop region result is connected to the new loop region argument
        let loop_new_result_origin = rvsdg[loop_region].value_results()[2].origin;
        assert_eq!(loop_new_result_origin, ValueOrigin::Argument(1));

        // Verify that the new input of loop_node is connected to the bool_value
        let ValueOrigin::Output {
            producer: loop_new_input_producer,
            output: 0,
        } = rvsdg[loop_node].value_inputs()[1].origin
        else {
            panic!("expected output origin for loop new input");
        };
        assert_eq!(loop_new_input_producer, bool_value);
    }
}
