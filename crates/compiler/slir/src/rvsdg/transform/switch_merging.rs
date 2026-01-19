use std::collections::VecDeque;

use crate::rvsdg::analyse::region_stratification::RegionStratifier;
use crate::rvsdg::transform::region_replication::replicate_region;
use crate::rvsdg::visit::region_nodes::RegionNodesVisitor;
use crate::rvsdg::{
    Connectivity, Node, NodeKind, Region, Rvsdg, StateOrigin, ValueOrigin, ValueUser, visit,
};
use crate::{Function, Module};

fn merge_switch_nodes(
    module: &mut Module,
    rvsdg: &mut Rvsdg,
    region: Region,
    node_0: Node,
    node_1: Node,
) {
    // Note: we merge node_0 into node_1, then remove node_0 from the graph. That way, if there
    // are yet more candidate switch nodes that share a branch selector predicate with node_0 and
    // node_1, i.e. node_2, then we can continue to iteratively merge node_1 into node_2, etc.

    let node_0_data = rvsdg[node_0].expect_switch();
    let node_1_data = rvsdg[node_1].expect_switch();
    let node_0_input_count = node_0_data.value_inputs().len();
    let node_0_arg_count = node_0_input_count - 1;
    let node_0_output_count = node_0_data.value_outputs().len();
    let node_0_state_origin = node_0_data.state().map(|s| s.origin);
    let node_1_has_state = node_1_data.state().is_some();
    let prior_node_1_input_count = rvsdg[node_1].value_inputs().len();
    let prior_node_1_arg_count = prior_node_1_input_count - 1;
    let prior_node_1_result_count = node_1_data.value_outputs().len();
    let branch_count = node_0_data.branches().len();

    assert_eq!(
        branch_count,
        node_1_data.branches().len(),
        "switch nodes that share a branch selector predicate must have the same number of branches"
    );

    // Add new inputs to node_1 for each entry value input of node_0.
    for i in 1..node_0_input_count {
        let input = rvsdg[node_0].value_inputs()[i];

        rvsdg.add_switch_input(node_1, input);
    }

    // Add new outputs to node_1 for each output of node_0.
    for i in 0..node_0_output_count {
        let node_0_output = &rvsdg[node_0].value_outputs()[i];
        let ty = node_0_output.ty;
        let user_count = node_0_output.users.len();

        let output = rvsdg.add_switch_output(node_1, ty);

        // Transfer all the users of the node_0 output to the node_1 output, expect if node_1 is
        // the user.
        for j in (0..user_count).rev() {
            let user = rvsdg[node_0].value_outputs()[i].users[j];

            match user {
                ValueUser::Input { consumer, .. } if consumer == node_1 => (),
                _ => rvsdg.reconnect_value_user(
                    region,
                    user,
                    ValueOrigin::Output {
                        producer: node_1,
                        output,
                    },
                ),
            }
        }
    }

    // If node_0 is part of the state chain, but node_1 is not, then link node_1 into the state
    // chain adjacent to node_0 (we'll remove node_0 later, so node_1 will essentially take its
    // place in the state chain).
    if let Some(node_0_state_origin) = node_0_state_origin
        && !node_1_has_state
    {
        rvsdg.link_switch_state(node_1, node_0_state_origin);
    }

    let arg_mapping_start = prior_node_1_arg_count;
    let arg_mapping_end = prior_node_1_arg_count + node_0_arg_count;
    let arg_mapping = (arg_mapping_start..arg_mapping_end)
        .map(|i| ValueOrigin::Argument(i as u32))
        .collect::<Vec<_>>();
    let state_mapping = Some(StateOrigin::Argument);

    // Replicate each of node_0's branches into the corresponding branch regions of node_1
    for i in 0..branch_count {
        let src_branch = rvsdg[node_0].expect_switch().branches()[i];
        let dst_branch = rvsdg[node_1].expect_switch().branches()[i];

        let result_mapping = replicate_region(
            module,
            rvsdg,
            src_branch,
            dst_branch,
            arg_mapping.clone(),
            state_mapping,
        );

        // Connect the results of the replicated region to the newly added results of node_1's
        // branch region.
        for i in 0..result_mapping.len() {
            let result = prior_node_1_result_count + i;

            rvsdg.reconnect_region_result(dst_branch, result as u32, result_mapping[i]);
        }

        // Some inputs of node_1 may be users of outputs on node_0. If there are any such inputs,
        // reconnect users of the corresponding arguments to the mapped results of the replicated
        // region nodes.
        for i in 0..prior_node_1_arg_count {
            let input_origin = rvsdg[node_1].expect_switch().entry_inputs()[i].origin;

            if let ValueOrigin::Output { producer, output } = input_origin
                && producer == node_0
            {
                rvsdg.reconnect_value_users(
                    dst_branch,
                    ValueOrigin::Argument(i as u32),
                    result_mapping[output as usize],
                );
            }
        }
    }

    // If there were any inputs of node_1 that were users of outputs on node_0, then none of their
    // corresponding arguments should have any users now, so we can remove those inputs.
    for i in (1..prior_node_1_input_count).rev() {
        let origin = rvsdg[node_1].value_inputs()[i].origin;

        if let ValueOrigin::Output { producer, .. } = origin
            && producer == node_0
        {
            rvsdg.remove_switch_input(node_1, i as u32);
        }
    }

    // None of node_0's outputs should have any users now, so we can remove node_0
    rvsdg.remove_node(node_0);
}

struct Candidate {
    switch_node: Node,
    stratum: usize,
    predicate_origin: ValueOrigin,
}

struct RegionSwitchMerger {
    candidates: VecDeque<Candidate>,
    region_stratifier: RegionStratifier,
}

impl RegionSwitchMerger {
    fn new() -> Self {
        Self {
            candidates: VecDeque::new(),
            region_stratifier: RegionStratifier::new(),
        }
    }

    fn collect_candidates(&mut self, rvsdg: &Rvsdg, region: Region) {
        self.region_stratifier
            .stratify(rvsdg, region, |node, stratum| {
                if let NodeKind::Switch(n) = rvsdg[node].kind() {
                    self.candidates.push_back(Candidate {
                        switch_node: node,
                        stratum,
                        predicate_origin: n.predicate().origin,
                    });
                }
            });
    }

    fn merge_in_region(&mut self, module: &mut Module, rvsdg: &mut Rvsdg, region: Region) {
        self.collect_candidates(rvsdg, region);

        while let Some(candidate) = self.candidates.pop_front() {
            let max_stratum = candidate.stratum + 1;

            for other in &self.candidates {
                // The candidate node list was collected in such a way that the candidates will be
                // ordered by "stratum" (see analyze::region_stratification). We allow two switch
                // nodes to merge if they are either in the same stratum or in consecutive strata:
                // if this is the case, then we are assured that there are no connections between
                // the two switch nodes that travel through another node (only direct connections,
                // if any).

                if other.stratum > max_stratum {
                    // The stratum gap is too large and, because of the ordering, will only get
                    // larger for later candidates, so we can stop looking for candidates here.
                    break;
                }

                if other.predicate_origin == candidate.predicate_origin {
                    merge_switch_nodes(
                        module,
                        rvsdg,
                        region,
                        candidate.switch_node,
                        other.switch_node,
                    );
                }
            }
        }
    }
}

struct RegionCollector<'a> {
    region_stack: &'a mut Vec<Region>,
}

impl RegionNodesVisitor for RegionCollector<'_> {
    fn visit_region(&mut self, rvsdg: &Rvsdg, region: Region) {
        self.region_stack.push(region);

        visit::region_nodes::visit_region(self, rvsdg, region);
    }
}

pub struct SwitchMerger {
    region_stack: Vec<Region>,
    region_switch_merger: RegionSwitchMerger,
}

impl SwitchMerger {
    pub fn new() -> Self {
        Self {
            region_stack: Vec::new(),
            region_switch_merger: RegionSwitchMerger::new(),
        }
    }

    pub fn merge_in_fn(&mut self, module: &mut Module, rvsdg: &mut Rvsdg, function: Function) {
        let fn_node = rvsdg
            .get_function_node(function)
            .expect("function not registered");
        let mut collector = RegionCollector {
            region_stack: &mut self.region_stack,
        };

        collector.visit_node(rvsdg, fn_node);

        while let Some(region) = self.region_stack.pop() {
            self.region_switch_merger
                .merge_in_region(module, rvsdg, region);
        }
    }
}

pub fn transform_entry_points(module: &mut Module, rvsdg: &mut Rvsdg) {
    let mut merger = SwitchMerger::new();
    let entry_points = module
        .entry_points
        .iter()
        .map(|(f, _)| f)
        .collect::<Vec<_>>();

    for entry_point in entry_points {
        merger.merge_in_fn(module, rvsdg, entry_point);
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::rvsdg::{ValueInput, ValueOutput};
    use crate::ty::{TY_DUMMY, TY_PREDICATE, TY_U32};
    use crate::{BinaryOperator, FnSig, Symbol};

    #[test]
    fn test_merge_same_stratum_switch_nodes() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref(""),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Default::default(),
                ty: TY_DUMMY,
                args: vec![],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let pred_0_node = rvsdg.add_const_predicate(region, 0);
        let pred_1_node = rvsdg.add_const_predicate(region, 0);

        let value_0_node = rvsdg.add_const_u32(region, 0);
        let value_1_node = rvsdg.add_const_u32(region, 1);
        let value_2_node = rvsdg.add_const_u32(region, 2);

        let switch_0_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::output(TY_PREDICATE, pred_0_node, 0),
                ValueInput::output(TY_U32, value_0_node, 0),
            ],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let switch_0_branch_0 = rvsdg.add_switch_branch(switch_0_node);

        rvsdg.reconnect_region_result(switch_0_branch_0, 0, ValueOrigin::Argument(0));

        let switch_0_branch_1 = rvsdg.add_switch_branch(switch_0_node);

        let switch_0_fallback_node = rvsdg.add_const_u32(switch_0_branch_1, 0);

        rvsdg.reconnect_region_result(
            switch_0_branch_1,
            0,
            ValueOrigin::Output {
                producer: switch_0_fallback_node,
                output: 0,
            },
        );

        let switch_1_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::output(TY_PREDICATE, pred_0_node, 0),
                ValueInput::output(TY_U32, value_1_node, 0),
            ],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let switch_1_branch_0 = rvsdg.add_switch_branch(switch_1_node);

        rvsdg.reconnect_region_result(switch_1_branch_0, 0, ValueOrigin::Argument(0));

        let switch_1_branch_1 = rvsdg.add_switch_branch(switch_1_node);

        let switch_1_fallback_node = rvsdg.add_const_u32(switch_1_branch_1, 1);

        rvsdg.reconnect_region_result(
            switch_1_branch_1,
            0,
            ValueOrigin::Output {
                producer: switch_1_fallback_node,
                output: 0,
            },
        );

        let switch_2_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::output(TY_PREDICATE, pred_1_node, 0),
                ValueInput::output(TY_U32, value_2_node, 0),
            ],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let switch_2_branch_0 = rvsdg.add_switch_branch(switch_2_node);

        rvsdg.reconnect_region_result(switch_2_branch_0, 0, ValueOrigin::Argument(0));

        let switch_2_branch_1 = rvsdg.add_switch_branch(switch_2_node);

        let switch_2_fallback_node = rvsdg.add_const_u32(switch_2_branch_1, 0);

        rvsdg.reconnect_region_result(
            switch_2_branch_1,
            0,
            ValueOrigin::Output {
                producer: switch_2_fallback_node,
                output: 0,
            },
        );

        let add_node = rvsdg.add_op_binary(
            region,
            BinaryOperator::Add,
            ValueInput::output(TY_U32, switch_0_node, 0),
            ValueInput::output(TY_U32, switch_1_node, 0),
        );

        let switch_3_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::output(TY_PREDICATE, pred_0_node, 0),
                ValueInput::output(TY_U32, add_node, 0),
                ValueInput::output(TY_U32, switch_2_node, 0),
            ],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let switch_3_branch_0 = rvsdg.add_switch_branch(switch_3_node);

        rvsdg.reconnect_region_result(switch_3_branch_0, 0, ValueOrigin::Argument(0));

        let switch_3_branch_1 = rvsdg.add_switch_branch(switch_3_node);

        rvsdg.reconnect_region_result(switch_3_branch_1, 0, ValueOrigin::Argument(1));

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: switch_3_node,
                output: 0,
            },
        );

        let mut merger = SwitchMerger::new();

        merger.merge_in_fn(&mut module, &mut rvsdg, function);

        let switch_1_data = rvsdg[switch_1_node].expect_switch();

        assert!(
            !rvsdg.is_live_node(switch_0_node),
            "switch-0 should have been removed"
        );

        assert_eq!(
            switch_1_data.value_inputs(),
            &[
                ValueInput::output(TY_PREDICATE, pred_0_node, 0),
                ValueInput::output(TY_U32, value_1_node, 0),
                ValueInput::output(TY_U32, value_0_node, 0),
            ],
            "switch-0's inputs should be added to switch-1"
        );

        assert_eq!(
            rvsdg[switch_1_branch_0].value_results()[0].origin,
            ValueOrigin::Argument(0)
        );
        assert_eq!(
            rvsdg[switch_1_branch_0].value_results()[1].origin,
            ValueOrigin::Argument(1)
        );

        assert_eq!(
            rvsdg[switch_1_branch_1].value_results()[0].origin,
            ValueOrigin::Output {
                producer: switch_1_fallback_node,
                output: 0,
            }
        );

        let ValueOrigin::Output {
            producer: fallback_0_repl_node,
            output: 0,
        } = rvsdg[switch_1_branch_1].value_results()[1].origin
        else {
            panic!("switch-1's second result should be connected to a node output");
        };

        let fallback_0_repl_data = rvsdg[fallback_0_repl_node].expect_const_u32();

        assert_eq!(
            fallback_0_repl_data.value(),
            0,
            "the replicated fallback node should have the correct value"
        );

        assert!(
            rvsdg.is_live_node(switch_2_node),
            "switch-2 should not have been removed"
        );
        assert_eq!(
            rvsdg[switch_2_node].value_inputs(),
            &[
                ValueInput::output(TY_PREDICATE, pred_1_node, 0),
                ValueInput::output(TY_U32, value_2_node, 0),
            ],
            "switch-2's inputs should be unchanged"
        );

        let add_data = rvsdg[add_node].expect_op_binary();

        assert_eq!(
            add_data.lhs_input(),
            &ValueInput::output(TY_U32, switch_1_node, 1),
            "the add node's lhs input should now be switch-1's second output"
        );
        assert_eq!(
            add_data.rhs_input(),
            &ValueInput::output(TY_U32, switch_1_node, 0),
            "the add node's lhs input should still be switch-1's first output"
        );

        assert_eq!(
            rvsdg[switch_3_node].value_inputs(),
            &[
                ValueInput::output(TY_PREDICATE, pred_0_node, 0),
                ValueInput::output(TY_U32, add_node, 0),
                ValueInput::output(TY_U32, switch_2_node, 0),
            ],
            "switch-3's inputs should be unchanged"
        );
    }

    #[test]
    fn test_merge_adjacent_stratum_switch_nodes() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref(""),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Default::default(),
                ty: TY_DUMMY,
                args: vec![],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let pred_node = rvsdg.add_const_predicate(region, 0);

        let value_0_node = rvsdg.add_const_u32(region, 0);
        let value_1_node = rvsdg.add_const_u32(region, 1);
        let value_2_node = rvsdg.add_const_u32(region, 2);

        let switch_0_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::output(TY_PREDICATE, pred_node, 0),
                ValueInput::output(TY_U32, value_0_node, 0),
                ValueInput::output(TY_U32, value_1_node, 0),
            ],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let switch_0_branch_0 = rvsdg.add_switch_branch(switch_0_node);

        let add_0_node = rvsdg.add_op_binary(
            switch_0_branch_0,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 0),
            ValueInput::argument(TY_U32, 1),
        );

        rvsdg.reconnect_region_result(
            switch_0_branch_0,
            0,
            ValueOrigin::Output {
                producer: add_0_node,
                output: 0,
            },
        );

        let switch_0_branch_1 = rvsdg.add_switch_branch(switch_0_node);

        let switch_0_fallback_node = rvsdg.add_const_u32(switch_0_branch_1, 0);

        rvsdg.reconnect_region_result(
            switch_0_branch_1,
            0,
            ValueOrigin::Output {
                producer: switch_0_fallback_node,
                output: 0,
            },
        );

        let switch_1_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::output(TY_PREDICATE, pred_node, 0),
                ValueInput::output(TY_U32, switch_0_node, 0),
                ValueInput::output(TY_U32, value_2_node, 0),
            ],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let switch_1_branch_0 = rvsdg.add_switch_branch(switch_1_node);

        let add_1_node = rvsdg.add_op_binary(
            switch_1_branch_0,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 0),
            ValueInput::argument(TY_U32, 1),
        );

        rvsdg.reconnect_region_result(
            switch_1_branch_0,
            0,
            ValueOrigin::Output {
                producer: add_1_node,
                output: 0,
            },
        );

        let switch_1_branch_1 = rvsdg.add_switch_branch(switch_1_node);

        let switch_1_fallback_node = rvsdg.add_const_u32(switch_1_branch_1, 1);

        rvsdg.reconnect_region_result(
            switch_1_branch_1,
            0,
            ValueOrigin::Output {
                producer: switch_1_fallback_node,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: switch_1_node,
                output: 0,
            },
        );

        let mut merger = SwitchMerger::new();

        merger.merge_in_fn(&mut module, &mut rvsdg, function);

        assert!(
            !rvsdg.is_live_node(switch_0_node),
            "switch-0 should have been removed"
        );

        let switch_1_data = rvsdg[switch_1_node].expect_switch();

        assert_eq!(
            switch_1_data.value_inputs(),
            &[
                ValueInput::output(TY_PREDICATE, pred_node, 0),
                ValueInput::output(TY_U32, value_2_node, 0),
                ValueInput::output(TY_U32, value_0_node, 0),
                ValueInput::output(TY_U32, value_1_node, 0),
            ],
            "switch-0's inputs should have been added to switch-1"
        );

        assert_eq!(
            rvsdg[switch_1_branch_0].value_results()[0].origin,
            ValueOrigin::Output {
                producer: add_1_node,
                output: 0,
            },
            "switch-1's first branch's first result should still be connect to the add-1 node"
        );

        let add_1_data = rvsdg[add_1_node].expect_op_binary();

        let ValueOrigin::Output {
            producer: add_0_repl_node,
            output: 0,
        } = add_1_data.lhs_input().origin
        else {
            panic!("add-1's lhs input should be connected to a node output");
        };

        let add_0_repl_data = rvsdg[add_0_repl_node].expect_op_binary();

        assert_eq!(
            add_0_repl_data.lhs_input(),
            &ValueInput::argument(TY_U32, 1),
            "the replicated add node's lhs input should be the newly added second argument"
        );
        assert_eq!(
            add_0_repl_data.rhs_input(),
            &ValueInput::argument(TY_U32, 2),
            "the replicated add node's rhs input should be the newly added third argument"
        );

        assert_eq!(
            rvsdg[switch_1_branch_0].value_results()[1].origin,
            ValueOrigin::Output {
                producer: add_0_repl_node,
                output: 0
            },
            "switch-1's first branch's second result should now be connected to the replicated add \
            node"
        );

        assert_eq!(
            rvsdg[switch_1_branch_1].value_results()[0].origin,
            ValueOrigin::Output {
                producer: switch_1_fallback_node,
                output: 0,
            },
            "switch-1's second branch's first result should still be connected to the switch-1 \
            fallback node"
        );

        let ValueOrigin::Output {
            producer: fallback_0_repl_node,
            output: 0,
        } = rvsdg[switch_1_branch_1].value_results()[1].origin
        else {
            panic!("switch-1's second branch's second result should be connected to a node output");
        };

        let fallback_0_repl_data = rvsdg[fallback_0_repl_node].expect_const_u32();

        assert_eq!(
            fallback_0_repl_data.value(),
            0,
            "the replicated fallback node should have the correct value"
        );

        assert_eq!(
            rvsdg[region].value_results()[0].origin,
            ValueOrigin::Output {
                producer: switch_1_node,
                output: 0,
            },
            "the region's result should still be connected to switch-1's first output"
        );

        assert!(
            switch_1_data.value_outputs()[1].users.is_empty(),
            "switch-1's newly added output should not have any users"
        );
    }
}
