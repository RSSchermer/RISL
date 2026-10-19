use rustc_hash::FxHashSet;

use crate::rvsdg::visit::region_nodes::RegionNodesVisitor;
use crate::rvsdg::visit::value_flow::ValueFlowVisitor;
use crate::rvsdg::{
    Connectivity, Node, NodeKind, Region, Rvsdg, SimpleNode, ValueOrigin, ValueUser, visit,
};
use crate::{BranchCase, Function, Module};

struct NodeCollector<'a> {
    candidates: &'a mut Vec<Node>,
}

impl RegionNodesVisitor for NodeCollector<'_> {
    fn visit_node(&mut self, rvsdg: &Rvsdg, node: Node) {
        use NodeKind::*;
        use SimpleNode::*;

        match rvsdg[node].kind() {
            Simple(OpBranchSelectorToCase(_)) => self.candidates.push(node),
            _ => (),
        }

        visit::region_nodes::visit_node(self, rvsdg, node);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CasesMatch<'a> {
    Permutation(&'a [usize]),
    NoMatch,
}

struct CasesMatcher {
    permutation: Vec<usize>,
}

impl CasesMatcher {
    fn new() -> Self {
        Self {
            permutation: Vec::new(),
        }
    }

    fn match_cases(
        &mut self,
        pred_to_case: &[BranchCase],
        case_to_pred: &[BranchCase],
    ) -> CasesMatch<'_> {
        self.permutation.clear();

        for i in 0..pred_to_case.len() {
            // Resolve the index of the branch target the `i`th case in `pred_to_case` will map to.
            // If we find it in the `case_to_pred` list, then the index of the branch target will be
            // the entry's index. Otherwise, the index of the branch will be the index of the
            // "default" branch, which is `case_to_pred.len()`.
            let j = case_to_pred
                .iter()
                .position(|&x| x == pred_to_case[i])
                .unwrap_or(case_to_pred.len());

            // The same branch target cannot be mapped to multiple cases: the resulting branch
            // permutation would imply a branch duplication, which we don't wish to do.
            if self.permutation.contains(&j) {
                return CasesMatch::NoMatch;
            }

            self.permutation.push(j);
        }

        CasesMatch::Permutation(&self.permutation)
    }
}

struct DependentFinder {
    dependents: Vec<Node>,
    visited: FxHashSet<(Region, ValueUser)>,
    all_uses_supported: bool,
}

impl DependentFinder {
    fn new() -> Self {
        Self {
            dependents: Vec::new(),
            visited: FxHashSet::default(),
            all_uses_supported: true,
        }
    }

    fn find_dependents(&mut self, rvsdg: &Rvsdg, node: Node) -> Option<&[Node]> {
        self.dependents.clear();
        self.visited.clear();
        self.all_uses_supported = true;

        self.visit_value_output(rvsdg, node, 0);

        self.all_uses_supported.then_some(&self.dependents)
    }
}

impl ValueFlowVisitor for DependentFinder {
    fn should_visit(&mut self, region: Region, user: ValueUser) -> bool {
        self.all_uses_supported && self.visited.insert((region, user))
    }

    fn visit_region_result(&mut self, rvsdg: &Rvsdg, region: Region, result: u32) {
        // We don't follow values that flow out of switch or loop nodes. If the value is a proper
        // "passthrough" value (a value that passes through a loop or switch node unaltered), then a
        // prior passthrough-elimination pass should have normalized such value-flow to bypass the
        // switch/loop node. There is one exception: we do follow values into loop-regions, if the
        // loop-value is loop-invariant. In such cases the value may reach the loop-region argument
        // without invalidating the merge, but only if the corresponding loop output is unused.

        let owner = rvsdg[region].owner();

        if let NodeKind::Loop(_) = rvsdg[owner].kind()
            && result > 0
            && rvsdg[region].value_results()[result as usize].origin
                == ValueOrigin::Argument(result - 1)
            && rvsdg[owner].value_outputs()[result as usize - 1]
                .users
                .is_empty()
        {
            return;
        }

        self.all_uses_supported = false;
    }

    fn visit_value_input(&mut self, rvsdg: &Rvsdg, node: Node, input: u32) {
        use NodeKind::*;
        use SimpleNode::*;

        match rvsdg[node].kind() {
            Simple(OpBranchSelectorToCase(_)) => self.dependents.push(node),
            Switch(_) if input == 0 => self.dependents.push(node),
            Switch(_) => visit::value_flow::visit_value_input(self, rvsdg, node, input),
            Loop(n) => {
                let region = n.loop_region();

                // See the comment in visit_region_result
                if rvsdg[region].value_results()[input as usize + 1].origin
                    == ValueOrigin::Argument(input)
                    && rvsdg[node].value_outputs()[input as usize].users.is_empty()
                {
                    self.visit_region_argument(rvsdg, region, input);
                } else {
                    self.all_uses_supported = false;
                }
            }
            _ => self.all_uses_supported = false,
        }
    }
}

struct Merger {
    case_matcher: CasesMatcher,
    dependent_finder: DependentFinder,
}

impl Merger {
    fn new() -> Self {
        Self {
            case_matcher: CasesMatcher::new(),
            dependent_finder: DependentFinder::new(),
        }
    }

    fn try_merge_pred_to_case_to_pred(&mut self, rvsdg: &mut Rvsdg, pred_to_case: Node) {
        use NodeKind::*;
        use SimpleNode::*;

        let region = rvsdg[pred_to_case].region();
        let data = rvsdg[pred_to_case].expect_op_branch_selector_to_case();
        let pred_origin = data.value_input().origin;
        let user_count = data.value_output().users.len();

        let mut merged_user_count = 0;

        for i in (0..user_count).rev() {
            let data = rvsdg[pred_to_case].expect_op_branch_selector_to_case();
            let user = data.value_output().users[i];

            if let ValueUser::Input {
                consumer: case_to_pred,
                input: 0,
            } = user
                && let Simple(OpCaseToBranchSelector(n)) = rvsdg[case_to_pred].kind()
                && data.encoding().backend_ty() == n.encoding().backend_ty()
            {
                let case_match = self.case_matcher.match_cases(data.cases(), n.cases());

                if matches!(case_match, CasesMatch::NoMatch) {
                    continue;
                }

                // Apply the permutation to this OpCaseToBranchSelector's dependents. If the
                // permutation is the identity permutation, then we can skip this step.
                if let CasesMatch::Permutation(permutation) = case_match
                    && permutation.iter().enumerate().any(|(i, &j)| i != j)
                {
                    let Some(dependents) =
                        self.dependent_finder.find_dependents(rvsdg, case_to_pred)
                    else {
                        // If the dependent_finder returned None, then at least one of the
                        // downstream users doesn't support the merge, so we abort merging this
                        // OpCaseToBranchSelector.
                        continue;
                    };

                    for dependent in dependents.iter().copied() {
                        // A previous permutation may have removed a branch containing this node.
                        if !rvsdg.is_live_node(dependent) {
                            continue;
                        }

                        match rvsdg[dependent].kind() {
                            Switch(_) => rvsdg.permute_switch_branches(dependent, permutation),
                            Simple(OpBranchSelectorToCase(_)) => rvsdg
                                .permute_op_branch_selector_to_case_cases(dependent, permutation),
                            _ => unreachable!(
                                "find_dependents should not have found other node kinds"
                            ),
                        }
                    }
                }

                rvsdg.reconnect_value_users(
                    region,
                    ValueOrigin::Output {
                        producer: case_to_pred,
                        output: 0,
                    },
                    pred_origin,
                );

                rvsdg.remove_node(case_to_pred);

                merged_user_count += 1;
            }
        }

        if merged_user_count == user_count {
            rvsdg.remove_node(pred_to_case);
        }
    }
}

pub struct PredToCaseToPredMerger {
    candidates: Vec<Node>,
    merger: Merger,
}

impl PredToCaseToPredMerger {
    pub fn new() -> Self {
        Self {
            candidates: Vec::new(),
            merger: Merger::new(),
        }
    }

    pub fn merge_in_fn(&mut self, rvsdg: &mut Rvsdg, function: Function) {
        let fn_node = rvsdg
            .get_function_node(function)
            .expect("function not registered");
        let mut collector = NodeCollector {
            candidates: &mut self.candidates,
        };

        collector.visit_node(rvsdg, fn_node);

        for node in self.candidates.drain(..) {
            if rvsdg.is_live_node(node) {
                self.merger.try_merge_pred_to_case_to_pred(rvsdg, node);
            }
        }
    }
}

pub fn transform_entry_points(module: &Module, rvsdg: &mut Rvsdg) {
    let mut merger = PredToCaseToPredMerger::new();

    for (function, _) in module.entry_points.iter() {
        merger.merge_in_fn(rvsdg, function);
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::rvsdg::{ValueInput, ValueOutput};
    use crate::ty::{Int, TY_BOOL, TY_DUMMY, TY_PREDICATE, TY_U32};
    use crate::{FnSig, Symbol};

    #[test]
    fn test_case_match() {
        let mut matcher = CasesMatcher::new();

        let cases_0 = [1u32, 2, 3].map(BranchCase::from);
        let cases_1 = [1u32, 2, 3].map(BranchCase::from);

        assert_eq!(
            matcher.match_cases(&cases_0, &cases_1),
            CasesMatch::Permutation(&[0, 1, 2])
        );

        let cases_0 = [1u32, 2, 3].map(BranchCase::from);
        let cases_1 = [3u32, 1, 2].map(BranchCase::from);

        assert_eq!(
            matcher.match_cases(&cases_0, &cases_1),
            CasesMatch::Permutation(&[1, 2, 0])
        );

        let cases_0 = [1u32, 2, 3].map(BranchCase::from);
        let cases_1 = [1u32, 2, 4].map(BranchCase::from);

        assert_eq!(
            matcher.match_cases(&cases_0, &cases_1),
            CasesMatch::Permutation(&[0, 1, 3])
        );

        let cases_0 = [1u32, 2, 3].map(BranchCase::from);
        let cases_1 = [1u32, 2].map(BranchCase::from);

        assert_eq!(
            matcher.match_cases(&cases_0, &cases_1),
            CasesMatch::Permutation(&[0, 1, 2])
        );

        let cases_0 = [1u32, 2].map(BranchCase::from);
        let cases_1 = [1u32, 2, 3].map(BranchCase::from);

        assert_eq!(
            matcher.match_cases(&cases_0, &cases_1),
            CasesMatch::Permutation(&[0, 1])
        );

        let cases_0 = [1u32, 2, 1].map(BranchCase::from);
        let cases_1 = [1u32, 2, 3].map(BranchCase::from);

        assert_eq!(matcher.match_cases(&cases_0, &cases_1), CasesMatch::NoMatch);
    }

    #[test]
    fn test_merge_exact_match() {
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

        let bool_node = rvsdg.add_const_bool(region, false);
        let bool_pred_node =
            rvsdg.add_op_bool_to_branch_selector(region, ValueInput::output(TY_BOOL, bool_node, 0));
        let pred_to_case_node = rvsdg.add_op_branch_selector_to_case(
            region,
            ValueInput::output(TY_PREDICATE, bool_pred_node, 0),
            Int::U32,
            [0u32, 1].map(BranchCase::from),
        );
        let case_to_pred_node = rvsdg.add_op_case_to_branch_selector(
            region,
            ValueInput::output(TY_U32, pred_to_case_node, 0),
            Int::U32,
            [0u32, 1].map(BranchCase::from),
        );
        let switch_node = rvsdg.add_switch(
            region,
            vec![ValueInput::output(TY_PREDICATE, case_to_pred_node, 0)],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);

        let value_0 = rvsdg.add_const_u32(branch_0, 0);

        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: value_0,
                output: 0,
            },
        );

        let branch_1 = rvsdg.add_switch_branch(switch_node);

        let value_1 = rvsdg.add_const_u32(branch_1, 1);

        rvsdg.reconnect_region_result(
            branch_1,
            0,
            ValueOrigin::Output {
                producer: value_1,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: switch_node,
                output: 0,
            },
        );

        let mut merger = PredToCaseToPredMerger::new();

        merger.merge_in_fn(&mut rvsdg, function);

        assert!(
            !rvsdg.is_live_node(pred_to_case_node),
            "the pred-to-case node was removed"
        );
        assert!(
            !rvsdg.is_live_node(case_to_pred_node),
            "the case-to-pred node was removed"
        );

        assert_eq!(
            rvsdg[switch_node].expect_switch().branch_selector().origin,
            ValueOrigin::Output {
                producer: bool_pred_node,
                output: 0
            },
            "the switch node's predicate input now connects directly to the bool-to-pred node"
        );
    }

    #[test]
    fn test_merge_permutation_match() {
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

        let bool_node = rvsdg.add_const_bool(region, false);
        let bool_pred_node =
            rvsdg.add_op_bool_to_branch_selector(region, ValueInput::output(TY_BOOL, bool_node, 0));
        let pred_to_case_node = rvsdg.add_op_branch_selector_to_case(
            region,
            ValueInput::output(TY_PREDICATE, bool_pred_node, 0),
            Int::U32,
            [0u32, 1].map(BranchCase::from),
        );
        let case_to_pred_node = rvsdg.add_op_case_to_branch_selector(
            region,
            ValueInput::output(TY_U32, pred_to_case_node, 0),
            Int::U32,
            [1u32, 0].map(BranchCase::from),
        );
        let switch_node = rvsdg.add_switch(
            region,
            vec![ValueInput::output(TY_PREDICATE, case_to_pred_node, 0)],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);

        let value_0 = rvsdg.add_const_u32(branch_0, 0);

        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: value_0,
                output: 0,
            },
        );

        let branch_1 = rvsdg.add_switch_branch(switch_node);

        let value_1 = rvsdg.add_const_u32(branch_1, 1);

        rvsdg.reconnect_region_result(
            branch_1,
            0,
            ValueOrigin::Output {
                producer: value_1,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: switch_node,
                output: 0,
            },
        );

        let mut merger = PredToCaseToPredMerger::new();

        merger.merge_in_fn(&mut rvsdg, function);

        assert!(
            !rvsdg.is_live_node(pred_to_case_node),
            "the pred-to-case node was removed"
        );
        assert!(
            !rvsdg.is_live_node(case_to_pred_node),
            "the case-to-pred node was removed"
        );

        let switch_data = rvsdg[switch_node].expect_switch();

        assert_eq!(
            switch_data.branch_selector().origin,
            ValueOrigin::Output {
                producer: bool_pred_node,
                output: 0
            },
            "the switch node's predicate input now connects directly to the bool-to-pred node"
        );
        assert_eq!(
            switch_data.branches(),
            &[branch_1, branch_0],
            "the switch node's branches were reordered"
        );
    }

    #[test]
    fn test_propagate_merge_into_switch_branches() {
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
        let inner_selector = rvsdg.add_const_predicate(region, 0);
        let pred_to_case = rvsdg.add_op_branch_selector_to_case(
            region,
            ValueInput::output(TY_PREDICATE, inner_selector, 0),
            Int::U32,
            [BranchCase::from(1u32), BranchCase::from(0u32)],
        );
        let case_to_pred = rvsdg.add_op_case_to_branch_selector(
            region,
            ValueInput::output(TY_U32, pred_to_case, 0),
            Int::U32,
            [BranchCase::from(0u32)],
        );
        let outer_selector = rvsdg.add_const_predicate(region, 0);
        let switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::output(TY_PREDICATE, outer_selector, 0),
                ValueInput::output(TY_PREDICATE, case_to_pred, 0),
            ],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);
        let consumer_0 = rvsdg.add_op_branch_selector_to_case(
            branch_0,
            ValueInput::argument(TY_PREDICATE, 0),
            Int::U32,
            [BranchCase::from(111u32), BranchCase::from(222u32)],
        );

        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: consumer_0,
                output: 0,
            },
        );

        let branch_1 = rvsdg.add_switch_branch(switch_node);
        let consumer_1 = rvsdg.add_op_branch_selector_to_case(
            branch_1,
            ValueInput::argument(TY_PREDICATE, 0),
            Int::U32,
            [BranchCase::from(333u32), BranchCase::from(444u32)],
        );

        rvsdg.reconnect_region_result(
            branch_1,
            0,
            ValueOrigin::Output {
                producer: consumer_1,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: switch_node,
                output: 0,
            },
        );

        PredToCaseToPredMerger::new().merge_in_fn(&mut rvsdg, function);

        assert!(!rvsdg.is_live_node(pred_to_case));
        assert!(!rvsdg.is_live_node(case_to_pred));
        assert_eq!(
            rvsdg[switch_node].value_inputs()[1].origin,
            ValueOrigin::Output {
                producer: inner_selector,
                output: 0,
            },
            "the ordinary switch input now receives the original selector"
        );
        assert_eq!(
            rvsdg[consumer_0]
                .expect_op_branch_selector_to_case()
                .cases(),
            &[BranchCase::from(222u32), BranchCase::from(111u32)],
            "the consumer in the first branch was adjusted"
        );
        assert_eq!(
            rvsdg[consumer_1]
                .expect_op_branch_selector_to_case()
                .cases(),
            &[BranchCase::from(444u32), BranchCase::from(333u32)],
            "the consumer in the second branch was adjusted"
        );
    }

    #[test]
    fn test_propagate_merge_into_loop_when_loop_value_is_loop_invariant() {
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

        let selector = rvsdg.add_const_predicate(region, 0);
        let pred_to_case = rvsdg.add_op_branch_selector_to_case(
            region,
            ValueInput::output(TY_PREDICATE, selector, 0),
            Int::U32,
            [BranchCase::from(1u32), BranchCase::from(0u32)],
        );
        let case_to_pred = rvsdg.add_op_case_to_branch_selector(
            region,
            ValueInput::output(TY_U32, pred_to_case, 0),
            Int::U32,
            [BranchCase::from(0u32)],
        );
        let initial_value = rvsdg.add_const_u32(region, 0);

        let (loop_node, loop_region) = rvsdg.add_loop(
            region,
            vec![
                ValueInput::output(TY_PREDICATE, case_to_pred, 0),
                ValueInput::output(TY_U32, initial_value, 0),
            ],
            None,
        );
        let consumer = rvsdg.add_op_branch_selector_to_case(
            loop_region,
            ValueInput::argument(TY_PREDICATE, 0),
            Int::U32,
            [BranchCase::from(111u32), BranchCase::from(222u32)],
        );
        let reentry = rvsdg.add_const_bool(loop_region, false);

        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: reentry,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(loop_region, 1, ValueOrigin::Argument(0));
        rvsdg.reconnect_region_result(
            loop_region,
            2,
            ValueOrigin::Output {
                producer: consumer,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: loop_node,
                output: 1,
            },
        );

        PredToCaseToPredMerger::new().merge_in_fn(&mut rvsdg, function);

        assert!(!rvsdg.is_live_node(pred_to_case));
        assert!(!rvsdg.is_live_node(case_to_pred));
        assert_eq!(
            rvsdg[loop_node].value_inputs()[0].origin,
            ValueOrigin::Output {
                producer: selector,
                output: 0,
            },
            "the loop input now receives the original selector"
        );
        assert_eq!(
            rvsdg[consumer].expect_op_branch_selector_to_case().cases(),
            &[BranchCase::from(222u32), BranchCase::from(111u32)],
            "the consumer inside the loop was adjusted"
        );
    }

    #[test]
    fn test_reject_merge_into_loop_when_loop_value_is_not_loop_invariant() {
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
        let selector = rvsdg.add_const_predicate(region, 0);
        let pred_to_case = rvsdg.add_op_branch_selector_to_case(
            region,
            ValueInput::output(TY_PREDICATE, selector, 0),
            Int::U32,
            [BranchCase::from(1u32), BranchCase::from(0u32)],
        );
        let case_to_pred = rvsdg.add_op_case_to_branch_selector(
            region,
            ValueInput::output(TY_U32, pred_to_case, 0),
            Int::U32,
            [BranchCase::from(0u32)],
        );
        let initial_value = rvsdg.add_const_u32(region, 0);
        let (loop_node, loop_region) = rvsdg.add_loop(
            region,
            vec![
                ValueInput::output(TY_PREDICATE, case_to_pred, 0),
                ValueInput::output(TY_U32, initial_value, 0),
            ],
            None,
        );
        let consumer = rvsdg.add_op_branch_selector_to_case(
            loop_region,
            ValueInput::argument(TY_PREDICATE, 0),
            Int::U32,
            [BranchCase::from(111u32), BranchCase::from(222u32)],
        );
        let feedback_selector = rvsdg.add_const_predicate(loop_region, 0);
        let reentry = rvsdg.add_const_bool(loop_region, false);

        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: reentry,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            loop_region,
            1,
            ValueOrigin::Output {
                producer: feedback_selector,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            loop_region,
            2,
            ValueOrigin::Output {
                producer: consumer,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: loop_node,
                output: 1,
            },
        );

        PredToCaseToPredMerger::new().merge_in_fn(&mut rvsdg, function);

        assert!(rvsdg.is_live_node(pred_to_case));
        assert!(rvsdg.is_live_node(case_to_pred));
        assert_eq!(
            rvsdg[loop_node].value_inputs()[0].origin,
            ValueOrigin::Output {
                producer: case_to_pred,
                output: 0,
            },
            "the loop-input is unchanged"
        );
        assert_eq!(
            rvsdg[consumer].expect_op_branch_selector_to_case().cases(),
            &[BranchCase::from(111u32), BranchCase::from(222u32)],
            "the consumer inside the loop is unchanged"
        );
    }

    #[test]
    fn test_reject_merge_if_value_flow_exits_switch() {
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
                ret_ty: Some(TY_PREDICATE),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, region) = rvsdg.register_function(&module, function, iter::empty());
        let selector = rvsdg.add_const_predicate(region, 0);
        let switch_node = rvsdg.add_switch(
            region,
            vec![ValueInput::output(TY_PREDICATE, selector, 0)],
            vec![ValueOutput::new(TY_PREDICATE)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);
        let branch_selector = rvsdg.add_const_predicate(branch_0, 0);
        let pred_to_case = rvsdg.add_op_branch_selector_to_case(
            branch_0,
            ValueInput::output(TY_PREDICATE, branch_selector, 0),
            Int::U32,
            [BranchCase::from(1u32), BranchCase::from(0u32)],
        );
        let case_to_pred = rvsdg.add_op_case_to_branch_selector(
            branch_0,
            ValueInput::output(TY_U32, pred_to_case, 0),
            Int::U32,
            [BranchCase::from(0u32)],
        );

        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: case_to_pred,
                output: 0,
            },
        );

        let branch_1 = rvsdg.add_switch_branch(switch_node);
        let other_selector = rvsdg.add_const_predicate(branch_1, 0);

        rvsdg.reconnect_region_result(
            branch_1,
            0,
            ValueOrigin::Output {
                producer: other_selector,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: switch_node,
                output: 0,
            },
        );

        PredToCaseToPredMerger::new().merge_in_fn(&mut rvsdg, function);

        assert!(rvsdg.is_live_node(pred_to_case));
        assert!(rvsdg.is_live_node(case_to_pred));
        assert_eq!(
            rvsdg[branch_0].value_results()[0].origin,
            ValueOrigin::Output {
                producer: case_to_pred,
                output: 0,
            },
        );
    }
}
