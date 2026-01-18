use rustc_hash::FxHashSet;

use crate::rvsdg::visit::region_nodes::RegionNodesVisitor;
use crate::rvsdg::visit::value_flow::ValueFlowVisitor;
use crate::rvsdg::{Node, NodeKind, Region, Rvsdg, SimpleNode, ValueOrigin, ValueUser, visit};
use crate::{Function, Module};

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
    Exact,
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

    fn match_cases(&mut self, cases_0: &[u32], cases_1: &[u32]) -> CasesMatch {
        // Note: this assumes that the cases in `cases_0` are all unique (no duplicates).

        if cases_0.len() != cases_1.len() {
            return CasesMatch::NoMatch;
        }

        self.permutation.clear();

        let mut is_exact_match = true;

        for i in 0..cases_0.len() {
            if let Some(j) = cases_1.iter().position(|&x| x == cases_0[i]) {
                self.permutation.push(j);

                if j != i {
                    is_exact_match = false;
                }
            } else {
                return CasesMatch::NoMatch;
            }
        }

        if is_exact_match {
            CasesMatch::Exact
        } else {
            CasesMatch::Permutation(&self.permutation)
        }
    }
}

struct DependentFinder {
    dependents: Vec<Node>,
    visited: FxHashSet<(Region, ValueUser)>,
}

impl DependentFinder {
    fn new() -> Self {
        Self {
            dependents: Vec::new(),
            visited: FxHashSet::default(),
        }
    }

    fn find_dependents(&mut self, rvsdg: &Rvsdg, node: Node) -> &[Node] {
        self.dependents.clear();
        self.visited.clear();

        self.visit_value_output(rvsdg, node, 0);

        &self.dependents
    }
}

impl ValueFlowVisitor for DependentFinder {
    fn should_visit(&mut self, region: Region, user: ValueUser) -> bool {
        self.visited.insert((region, user))
    }

    fn visit_value_input(&mut self, rvsdg: &Rvsdg, node: Node, input: u32) {
        use NodeKind::*;
        use SimpleNode::*;

        match rvsdg[node].kind() {
            Simple(OpCaseToBranchSelector(_)) => self.dependents.push(node),
            Switch(_) if input == 0 => self.dependents.push(node),
            Switch(_) | Loop(_) => visit::value_flow::visit_value_input(self, rvsdg, node, input),
            _ => unreachable!("node kind cannot take a predicate type value as input"),
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
            {
                let case_match = self.case_matcher.match_cases(data.cases(), n.cases());

                if matches!(case_match, CasesMatch::NoMatch) {
                    continue;
                }

                if let CasesMatch::Permutation(permutation) = case_match {
                    for dependent in self
                        .dependent_finder
                        .find_dependents(rvsdg, case_to_pred)
                        .iter()
                        .copied()
                    {
                        match rvsdg[dependent].kind() {
                            Switch(_) => rvsdg.permute_switch_branches(dependent, permutation),
                            Simple(OpCaseToBranchSelector(_)) => rvsdg
                                .permute_op_case_to_branch_selector_cases(dependent, permutation),
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
            self.merger.try_merge_pred_to_case_to_pred(rvsdg, node);
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
    use crate::ty::{TY_BOOL, TY_DUMMY, TY_PREDICATE, TY_U32};
    use crate::{FnSig, Symbol};

    #[test]
    fn test_case_match() {
        let mut matcher = CasesMatcher::new();

        let cases_0 = [1, 2, 3];
        let cases_1 = [1, 2, 3];

        assert_eq!(matcher.match_cases(&cases_0, &cases_1), CasesMatch::Exact);

        let cases_0 = [1, 2, 3];
        let cases_1 = [3, 1, 2];

        assert_eq!(
            matcher.match_cases(&cases_0, &cases_1),
            CasesMatch::Permutation(&[1, 2, 0])
        );

        let cases_0 = [1, 2, 3];
        let cases_1 = [1, 2, 4];

        assert_eq!(matcher.match_cases(&cases_0, &cases_1), CasesMatch::NoMatch);

        let cases_0 = [1, 2, 3];
        let cases_1 = [1, 2];

        assert_eq!(matcher.match_cases(&cases_0, &cases_1), CasesMatch::NoMatch);

        let cases_0 = [1, 2];
        let cases_1 = [1, 2, 3];

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
            [0, 1],
        );
        let case_to_pred_node = rvsdg.add_op_case_to_branch_selector(
            region,
            ValueInput::output(TY_U32, pred_to_case_node, 0),
            [0, 1],
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
            rvsdg[switch_node].expect_switch().predicate().origin,
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
            [0, 1],
        );
        let case_to_pred_node = rvsdg.add_op_case_to_branch_selector(
            region,
            ValueInput::output(TY_U32, pred_to_case_node, 0),
            [1, 0],
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
            switch_data.predicate().origin,
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
}
