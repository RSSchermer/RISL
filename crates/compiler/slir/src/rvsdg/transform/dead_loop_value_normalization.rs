//! Normalizes dead loop-values to prepare them for elimination.
//!
//! The `i`th loop-value consists of four parts:
//!
//! 1. The [LoopNode]'s `i`th value-input: this initializes value for the first iteration.
//! 2. The [LoopNode]'s `i`th value-output: this represents the value after the loop completes.
//! 3. The `i`th loop-region argument: this represents the value for each iteration.
//! 4. The `i+1`th loop-region result: this either sets the argument value for the next iteration if
//!    the reentry condition is `true`, or sets the value-output if the reentry condition is
//!    `false`.
//!
//! Note that the index of the result is `i+1`, as the first result of a loop-region is always the
//! reentry condition; the loop-value results follow.
//!
//! A loop-value is "used" if either:
//!
//! - The loop value-output is used.
//! - The loop-region argument is used.
//!
//! We consider a loop value-output "used" if its user-set (see [ValueOutput::users]) is not empty.
//!
//! To determine if the loop-region argument is used, we perform a value-flow analysis starting at
//! the loop-region argument. The loop-region argument is considered "used" if value-flow can reach
//! any loop-region result for which the corresponding loop value-output is "used", or if value-flow
//! can reach the loop-region's state-result.
//!
//! If a loop-value is not "used", then it is considered "dead". This transform does not directly
//! eliminate dead loop-values. Instead, the loop-value's result is reconnected directly to the
//! loop-value's argument. This breaks the value-flow cycle and will allow a subsequent
//! [dead_value_elimination](super::dead_value_elimination) pass to eliminate the loop-value and
//! any nodes that were part of its value-flow.

use std::collections::VecDeque;
use std::mem;

use rustc_hash::FxHashSet;

use crate::Module;
use crate::rvsdg::visit::region_nodes::RegionNodesVisitor;
use crate::rvsdg::{Connectivity, Node, NodeKind, Region, Rvsdg, ValueOrigin, ValueUser, visit};

struct LoopNodeCollector<'a> {
    candidates: &'a mut Vec<Node>,
}

impl RegionNodesVisitor for LoopNodeCollector<'_> {
    fn visit_node(&mut self, rvsdg: &Rvsdg, node: Node) {
        if rvsdg[node].is_loop() {
            self.candidates.push(node);
        }

        visit::region_nodes::visit_node(self, rvsdg, node);
    }
}

struct LoopValueUseAnalyzer {
    visited: FxHashSet<(Region, ValueUser)>,
    queue: VecDeque<(Region, ValueUser)>,
    loop_node: Node,
    used: bool,
}

impl LoopValueUseAnalyzer {
    fn new() -> Self {
        Self {
            visited: FxHashSet::default(),
            queue: Default::default(),
            loop_node: Default::default(),
            used: false,
        }
    }

    fn loop_value_is_used(&mut self, rvsdg: &Rvsdg, loop_node: Node, index: u32) -> bool {
        self.visited.clear();
        self.queue.clear();
        self.loop_node = loop_node;

        let data = rvsdg[loop_node].expect_loop();

        self.used = !data.value_outputs()[index as usize].users.is_empty();

        let loop_region = data.loop_region();

        self.visit_region_argument(rvsdg, loop_region, index);

        while !self.used
            && let Some((region, user)) = self.queue.pop_front()
        {
            self.visit_value_user(rvsdg, region, user);
        }

        self.used
    }

    fn visit_region_argument(&mut self, rvsdg: &Rvsdg, region: Region, arg: u32) {
        for user in &rvsdg[region].value_arguments()[arg as usize].users {
            self.queue.push_back((region, *user));
        }
    }

    fn visit_node_output(&mut self, rvsdg: &Rvsdg, node: Node, output: u32) {
        let data = &rvsdg[node];
        let region = data.region();
        let users = &data.value_outputs()[output as usize].users;

        if node == self.loop_node && !users.is_empty() {
            self.used = true;

            return;
        }

        for user in users {
            self.queue.push_back((region, *user));
        }
    }

    fn visit_value_user(&mut self, rvsdg: &Rvsdg, region: Region, user: ValueUser) {
        if self.visited.insert((region, user)) {
            match user {
                ValueUser::Result(res) => self.visit_region_result(rvsdg, region, res),
                ValueUser::Input { consumer, input } => {
                    self.visit_node_input(rvsdg, consumer, input)
                }
            }
        }
    }

    fn visit_region_result(&mut self, rvsdg: &Rvsdg, region: Region, result: u32) {
        let owner = rvsdg[region].owner();
        let owner_data = &rvsdg[owner];

        match owner_data.kind() {
            NodeKind::Switch(_) => self.visit_node_output(rvsdg, owner, result),
            NodeKind::Loop(_) => {
                if result == 0 {
                    self.used = true;
                } else {
                    self.visit_node_output(rvsdg, owner, result - 1)
                }
            }
            NodeKind::Simple(_)
            | NodeKind::UniformBinding(_)
            | NodeKind::StorageBinding(_)
            | NodeKind::WorkgroupBinding(_)
            | NodeKind::Constant(_)
            | NodeKind::Function(_) => {
                unreachable!("cannot own a region nested inside a loop-node")
            }
        }
    }

    fn visit_node_input(&mut self, rvsdg: &Rvsdg, node: Node, input: u32) {
        let data = &rvsdg[node];

        match data.kind() {
            NodeKind::Switch(switch_data) => {
                if input == 0 {
                    self.used = true;
                } else {
                    let arg = input - 1;

                    for branch in switch_data.branches() {
                        self.visit_region_argument(rvsdg, *branch, arg);
                    }
                }
            }
            NodeKind::Loop(loop_data) => {
                let loop_region = loop_data.loop_region();

                self.visit_region_argument(rvsdg, loop_region, input);
            }
            NodeKind::Simple(simple_node_data) => {
                if simple_node_data.state().is_some() {
                    // We assume a valid state chain, meaning the state chain should connect the
                    // function-region's state argument to its state result. Therefore, if a
                    // simple node is part of the state chain at all, then it must also be
                    // connected to the state result of the loop region that we're currently
                    // analyzing. Consequently, we don't have to trace the state chain and can
                    // immediately conclude that the loop-value is "used".

                    self.used = true;
                } else {
                    let output_count = simple_node_data.value_outputs().len() as u32;

                    for i in 0..output_count {
                        self.visit_node_output(rvsdg, node, i);
                    }
                }
            }
            NodeKind::UniformBinding(_)
            | NodeKind::StorageBinding(_)
            | NodeKind::WorkgroupBinding(_)
            | NodeKind::Constant(_)
            | NodeKind::Function(_) => unreachable!("node kind cannot exist inside a loop-region"),
        }
    }
}

pub struct DeadLoopValueNormalizer {
    candidates: Vec<Node>,
    analyzer: LoopValueUseAnalyzer,
}

impl DeadLoopValueNormalizer {
    pub fn new() -> Self {
        Self {
            candidates: Vec::new(),
            analyzer: LoopValueUseAnalyzer::new(),
        }
    }

    pub fn transform_region(&mut self, rvsdg: &mut Rvsdg, region: Region) {
        let mut collector = LoopNodeCollector {
            candidates: &mut self.candidates,
        };

        collector.visit_region(rvsdg, region);

        let mut candidates = mem::replace(&mut self.candidates, Vec::new());

        for node in candidates.drain(..) {
            self.normalize_loop_values(rvsdg, node);
        }

        self.candidates = candidates;
    }

    fn normalize_loop_values(&mut self, rvsdg: &mut Rvsdg, loop_node: Node) {
        let loop_data = rvsdg[loop_node].expect_loop();
        let loop_region = loop_data.loop_region();
        let loop_value_count = loop_data.value_outputs().len() as u32;

        for i in 0..loop_value_count {
            if !self.analyzer.loop_value_is_used(rvsdg, loop_node, i) {
                let result = i + 1;

                rvsdg.reconnect_region_result(loop_region, result, ValueOrigin::Argument(i));
            }
        }
    }
}

pub fn transform_entry_points(module: &Module, rvsdg: &mut Rvsdg) {
    let mut normalizer = DeadLoopValueNormalizer::new();

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
    use crate::rvsdg::{StateOrigin, ValueInput, ValueOrigin, ValueOutput};
    use crate::ty::{TY_DUMMY, TY_PREDICATE, TY_PTR_U32, TY_U32};
    use crate::{FnSig, Function, Module, Symbol};

    #[test]
    fn test_loop_value_use_analyzer_simple_used_value() {
        let mut module = Module::new(Symbol::from_ref("m"));
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

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let const_val = rvsdg.add_const_u32(region, 0);
        let val_input = ValueInput::output(TY_U32, const_val, 0);

        // Create a loop with 1 loop-value.
        let (loop_node, loop_region) = rvsdg.add_loop(region, vec![val_input], None);

        // Connect loop-region result 1 (the 0th loop-value) to loop-region argument 0.
        rvsdg.reconnect_region_result(loop_region, 1, ValueOrigin::Argument(0));

        // Use the loop's 0th value-output.
        let _user_node = rvsdg.add_value_proxy(region, ValueInput::output(TY_U32, loop_node, 0));

        let mut analyzer = LoopValueUseAnalyzer::new();
        assert!(analyzer.loop_value_is_used(&rvsdg, loop_node, 0));
    }

    #[test]
    fn test_loop_value_use_analyzer_simple_unused_value() {
        let mut module = Module::new(Symbol::from_ref("m"));
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

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let const_val = rvsdg.add_const_u32(region, 0);
        let val_input = ValueInput::output(TY_U32, const_val, 0);

        // Create a loop with 1 loop-value.
        let (loop_node, loop_region) = rvsdg.add_loop(region, vec![val_input], None);

        // Connect loop-region result 1 (the 0th loop-value) to loop-region argument 0.
        rvsdg.reconnect_region_result(loop_region, 1, ValueOrigin::Argument(0));

        // Note: NO user for the loop's 0th value-output.

        let mut analyzer = LoopValueUseAnalyzer::new();
        assert!(!analyzer.loop_value_is_used(&rvsdg, loop_node, 0));
    }

    #[test]
    fn test_loop_value_use_analyzer_proxy_unused_value() {
        let mut module = Module::new(Symbol::from_ref("m"));
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

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let const_val = rvsdg.add_const_u32(region, 0);
        let val_input = ValueInput::output(TY_U32, const_val, 0);

        // Create a loop with 1 loop-value.
        let (loop_node, loop_region) = rvsdg.add_loop(region, vec![val_input], None);

        // Inside the loop: Argument(0) -> Proxy -> Result(1)
        let proxy_node = rvsdg.add_value_proxy(loop_region, ValueInput::argument(TY_U32, 0));
        rvsdg.reconnect_region_result(
            loop_region,
            1,
            ValueOrigin::Output {
                producer: proxy_node,
                output: 0,
            },
        );

        // Note: NO user for the loop's 0th value-output.

        let mut analyzer = LoopValueUseAnalyzer::new();
        assert!(!analyzer.loop_value_is_used(&rvsdg, loop_node, 0));
    }

    #[test]
    fn test_loop_value_use_analyzer_used_by_different_output() {
        let mut module = Module::new(Symbol::from_ref("m"));
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

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let const_val = rvsdg.add_const_u32(region, 0);
        let val_input0 = ValueInput::output(TY_U32, const_val, 0);
        let val_input1 = ValueInput::output(TY_U32, const_val, 0);

        // Create a loop with 2 loop-values.
        let (loop_node, loop_region) = rvsdg.add_loop(region, vec![val_input0, val_input1], None);

        // Inside the loop:
        // Argument(0) -> Proxy -> Result(2) (Result 2 is the 1st loop-value's result)
        // Argument(0) -> Result(1) (simple loop-back for the 0th loop-value)
        let proxy_node = rvsdg.add_value_proxy(loop_region, ValueInput::argument(TY_U32, 0));

        // Result 1 corresponds to loop-value 0.
        rvsdg.reconnect_region_result(loop_region, 1, ValueOrigin::Argument(0));

        // Result 2 corresponds to loop-value 1.
        rvsdg.reconnect_region_result(
            loop_region,
            2,
            ValueOrigin::Output {
                producer: proxy_node,
                output: 0,
            },
        );

        // Use the loop's 1st value-output (Value 1).
        let _user_node = rvsdg.add_value_proxy(region, ValueInput::output(TY_U32, loop_node, 1));

        // Note: NO user for the loop's 0th value-output (Value 0).

        let mut analyzer = LoopValueUseAnalyzer::new();

        // Even though loop-output 0 is unused, argument 0 flows to loop-output 1, which is used.
        // Thus, loop-value 0 should be "used" because its argument is used.
        assert!(analyzer.loop_value_is_used(&rvsdg, loop_node, 0));
    }

    #[test]
    fn test_loop_value_use_analyzer_load_used_value() {
        let mut module = Module::new(Symbol::from_ref("m"));
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

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        // We need a pointer for OpLoad.
        let const_val = rvsdg.add_const_u32(region, 0);
        let const_ptr =
            rvsdg.add_const_ptr(region, TY_U32, ValueInput::output(TY_U32, const_val, 0));
        let val_input = ValueInput::output(TY_PTR_U32, const_ptr, 0);

        // Create a loop with 1 loop-value. Link the loop into the function's state chain.
        let (loop_node, loop_region) =
            rvsdg.add_loop(region, vec![val_input], Some(StateOrigin::Argument));

        // Inside the loop: Argument(0) is used by an OpLoad node.
        let _load_node = rvsdg.add_op_load(
            loop_region,
            ValueInput::argument(TY_PTR_U32, 0),
            StateOrigin::Argument,
        );

        // Connect loop-region result 1 (the 0th loop-value) back to loop-region argument 0.
        rvsdg.reconnect_region_result(loop_region, 1, ValueOrigin::Argument(0));

        // Note: NO user for the loop's 0th value-output.

        let mut analyzer = LoopValueUseAnalyzer::new();

        // The loop-value is "used" because its argument is used by OpLoad, which is a stateful node.
        assert!(analyzer.loop_value_is_used(&rvsdg, loop_node, 0));
    }

    #[test]
    fn test_loop_value_use_analyzer_switch_unused_value() {
        let mut module = Module::new(Symbol::from_ref("m"));
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

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let const_val = rvsdg.add_const_u32(region, 0);
        let val_input = ValueInput::output(TY_U32, const_val, 0);

        // Create a loop with 1 loop-value.
        let (loop_node, loop_region) = rvsdg.add_loop(region, vec![val_input], None);

        // Inside the loop, create a SwitchNode with 2 branches. It uses the loop-value as an input.
        let const_pred = rvsdg.add_const_predicate(loop_region, 0);
        let switch_node = rvsdg.add_switch(
            loop_region,
            vec![
                ValueInput::output(TY_PREDICATE, const_pred, 0),
                ValueInput::argument(TY_U32, 0),
            ],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        // Branch 0: pass the value through, unused.
        let branch0 = rvsdg.add_switch_branch(switch_node);
        rvsdg.reconnect_region_result(branch0, 0, ValueOrigin::Argument(0));

        // Branch 1: pass the value through, unused.
        let branch1 = rvsdg.add_switch_branch(switch_node);
        rvsdg.reconnect_region_result(branch1, 0, ValueOrigin::Argument(0));

        // Connect loop-region result 1 (the 0th loop-value) to the switch output.
        rvsdg.reconnect_region_result(
            loop_region,
            1,
            ValueOrigin::Output {
                producer: switch_node,
                output: 0,
            },
        );

        // Note: NO user for the loop's 0th value-output.

        let mut analyzer = LoopValueUseAnalyzer::new();
        // The value flow passes through a SwitchNode, but since it's remains unused inside the
        // switch node, it should be marked as unused.
        assert!(!analyzer.loop_value_is_used(&rvsdg, loop_node, 0));
    }

    #[test]
    fn test_loop_value_use_analyzer_switch_used_value() {
        let mut module = Module::new(Symbol::from_ref("m"));
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

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let const_val = rvsdg.add_const_u32(region, 0);
        let const_ptr =
            rvsdg.add_const_ptr(region, TY_U32, ValueInput::output(TY_U32, const_val, 0));
        let val_input = ValueInput::output(TY_PTR_U32, const_ptr, 0);

        // Create a loop with 1 loop-value, part of the state chain.
        let (loop_node, loop_region) =
            rvsdg.add_loop(region, vec![val_input], Some(StateOrigin::Argument));

        // Inside the loop, create a SwitchNode with 2 branches.
        let const_pred = rvsdg.add_const_predicate(loop_region, 0);
        let switch_node = rvsdg.add_switch(
            loop_region,
            vec![
                ValueInput::output(TY_PREDICATE, const_pred, 0),
                ValueInput::argument(TY_PTR_U32, 0),
            ],
            vec![],
            Some(StateOrigin::Argument),
        );

        // Branch 0: contains an OpLoad that uses the value.
        let branch0 = rvsdg.add_switch_branch(switch_node);
        rvsdg.add_op_load(
            branch0,
            ValueInput::argument(TY_PTR_U32, 0),
            StateOrigin::Argument,
        );

        // Branch 1: does not use the value.
        let _branch1 = rvsdg.add_switch_branch(switch_node);

        // Connect loop-region result 1 (the 0th loop-value) to the loop-region argument.
        rvsdg.reconnect_region_result(loop_region, 1, ValueOrigin::Argument(0));

        let mut analyzer = LoopValueUseAnalyzer::new();
        // The value is used by the OpLoad in Branch 0.
        assert!(analyzer.loop_value_is_used(&rvsdg, loop_node, 0));
    }

    #[test]
    fn test_loop_value_use_analyzer_nested_loop_unused_value() {
        let mut module = Module::new(Symbol::from_ref("m"));
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

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let const_val = rvsdg.add_const_u32(region, 0);
        let val_input = ValueInput::output(TY_U32, const_val, 0);

        // Create an outer loop with 1 loop-value.
        let (outer_loop_node, outer_loop_region) = rvsdg.add_loop(region, vec![val_input], None);

        // Inside the outer loop, create a nested loop.
        let nested_loop_input = ValueInput::argument(TY_U32, 0);
        let (nested_loop_node, nested_loop_region) =
            rvsdg.add_loop(outer_loop_region, vec![nested_loop_input], None);

        // Inside the nested loop: Argument(0) -> Result(1)
        rvsdg.reconnect_region_result(nested_loop_region, 1, ValueOrigin::Argument(0));

        // Connect outer loop-region result 1 (the 0th loop-value) to the nested loop's output.
        rvsdg.reconnect_region_result(
            outer_loop_region,
            1,
            ValueOrigin::Output {
                producer: nested_loop_node,
                output: 0,
            },
        );

        // Note: NO user for the outer loop's 0th value-output.

        let mut analyzer = LoopValueUseAnalyzer::new();
        // The value flow passes through a nested LoopNode, but since it's remains unused inside the
        // nested loop, it should be marked as unused.
        assert!(!analyzer.loop_value_is_used(&rvsdg, outer_loop_node, 0));
    }

    #[test]
    fn test_loop_value_use_analyzer_nested_loop_used_value() {
        let mut module = Module::new(Symbol::from_ref("m"));
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

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let const_val = rvsdg.add_const_fallback(region, TY_PTR_U32);
        let val_input = ValueInput::output(TY_PTR_U32, const_val, 0);

        // Create an outer loop with 1 loop-value and state.
        let (outer_loop_node, outer_loop_region) =
            rvsdg.add_loop(region, vec![val_input], Some(StateOrigin::Argument));

        // Inside the outer loop, create a nested loop with state.
        let nested_loop_input = ValueInput::argument(TY_PTR_U32, 0);
        let (nested_loop_node, nested_loop_region) = rvsdg.add_loop(
            outer_loop_region,
            vec![nested_loop_input],
            Some(StateOrigin::Argument),
        );

        // Inside the nested loop, use the value with an OpLoad.
        rvsdg.add_op_load(
            nested_loop_region,
            ValueInput::argument(TY_PTR_U32, 0),
            StateOrigin::Argument,
        );

        // Inside the nested loop: Argument(0) -> Result(1)
        rvsdg.reconnect_region_result(nested_loop_region, 1, ValueOrigin::Argument(0));

        // Connect outer loop-region result 1 (the 0th loop-value) to the nested loop's output.
        rvsdg.reconnect_region_result(
            outer_loop_region,
            1,
            ValueOrigin::Output {
                producer: nested_loop_node,
                output: 0,
            },
        );

        let mut analyzer = LoopValueUseAnalyzer::new();
        // The value flow passes through a nested LoopNode, where it's used by an OpLoad.
        assert!(analyzer.loop_value_is_used(&rvsdg, outer_loop_node, 0));
    }

    #[test]
    fn test_dead_loop_value_normalizer() {
        let mut module = Module::new(Symbol::from_ref("m"));
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

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let const_val = rvsdg.add_const_u32(region, 0);
        let val_inputs = vec![ValueInput::output(TY_U32, const_val, 0); 4];

        // Create a loop with 4 loop-values.
        let (loop_node, loop_region) = rvsdg.add_loop(region, val_inputs, None);

        // Inside the loop region, create 4 proxy nodes, one for each argument.
        let proxy_0 = rvsdg.add_value_proxy(loop_region, ValueInput::argument(TY_U32, 0));
        let proxy_1 = rvsdg.add_value_proxy(loop_region, ValueInput::argument(TY_U32, 1));
        let proxy_2 = rvsdg.add_value_proxy(loop_region, ValueInput::argument(TY_U32, 2));
        let proxy_3 = rvsdg.add_value_proxy(loop_region, ValueInput::argument(TY_U32, 3));

        // Connect loop results to the proxy's output.
        rvsdg.reconnect_region_result(
            loop_region,
            1,
            ValueOrigin::Output {
                producer: proxy_0,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            loop_region,
            2,
            ValueOrigin::Output {
                producer: proxy_1,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            loop_region,
            3,
            ValueOrigin::Output {
                producer: proxy_2,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            loop_region,
            4,
            ValueOrigin::Output {
                producer: proxy_3,
                output: 0,
            },
        );

        // Connect value-outputs 0 and 2 to users outside the loop.
        rvsdg.add_value_proxy(region, ValueInput::output(TY_U32, loop_node, 0));
        rvsdg.add_value_proxy(region, ValueInput::output(TY_U32, loop_node, 2));

        // The value-outputs for loop-values 1 and 3 are unused.

        let mut normalizer = DeadLoopValueNormalizer::new();
        normalizer.transform_region(&mut rvsdg, region);

        // Verify that results 2 and 4 (loop-values 1 and 3) now connect directly to their
        // arguments, and that the other results remain connected to their respective proxy nodes.

        let loop_data = rvsdg[loop_node].expect_loop();
        let loop_region_after = loop_data.loop_region();

        assert_eq!(
            rvsdg[loop_region_after].value_results()[1].origin,
            ValueOrigin::Output {
                producer: proxy_0,
                output: 0,
            }
        );
        assert_eq!(
            rvsdg[loop_region_after].value_results()[2].origin,
            ValueOrigin::Argument(1)
        );
        assert_eq!(
            rvsdg[loop_region_after].value_results()[3].origin,
            ValueOrigin::Output {
                producer: proxy_2,
                output: 0,
            }
        );
        assert_eq!(
            rvsdg[loop_region_after].value_results()[4].origin,
            ValueOrigin::Argument(3)
        );

        // Verify that proxy nodes for loop-values 1 and 3 no longer have any users.
        assert!(rvsdg[proxy_1].value_outputs()[0].users.is_empty());
        assert!(rvsdg[proxy_3].value_outputs()[0].users.is_empty());
    }
}
