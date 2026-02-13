//! Lowers any [OpBranchSelectorToCase] nodes not merged by the pred-to-case-to-pred-merging pass
//! into standard [Switch] nodes.
//!
//! All [OpBranchSelectorToCase] result from the pred-to-case-extraction pass, the purpose of which
//! is to discover merging opportunities for the pred-to-case-to-pred merging pass. However, any
//! such [OpBranchSelectorToCase] that did not end up being merged must be transformed back to
//! [Switch] nodes before we convert to the SCF representation.

use crate::rvsdg::visit::region_nodes::RegionNodesVisitor;
use crate::rvsdg::{
    Connectivity, Node, NodeKind, Region, Rvsdg, SimpleNode, ValueInput, ValueOrigin, ValueOutput,
    visit,
};
use crate::ty::TY_U32;
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

fn try_switchify_pred_to_case(rvsdg: &mut Rvsdg, node: Node) {
    use NodeKind::*;
    use SimpleNode::*;

    let region = rvsdg[node].region();
    let cases = {
        let Simple(OpBranchSelectorToCase(data)) = rvsdg[node].kind() else {
            return;
        };
        data.cases().to_vec()
    };
    let predicate_input = rvsdg[node].value_inputs()[0];

    let switch_node = rvsdg.add_switch(
        region,
        vec![predicate_input],
        vec![ValueOutput::new(TY_U32)],
        None,
    );

    for case_value in cases {
        let branch_region = rvsdg.add_switch_branch(switch_node);
        let const_node = rvsdg.add_const_u32(branch_region, case_value);

        rvsdg.reconnect_region_result(
            branch_region,
            0,
            ValueOrigin::Output {
                producer: const_node,
                output: 0,
            },
        );
    }

    rvsdg.reconnect_value_users(
        region,
        ValueOrigin::Output {
            producer: node,
            output: 0,
        },
        ValueOrigin::Output {
            producer: switch_node,
            output: 0,
        },
    );

    rvsdg.remove_node(node);
}

pub struct SwitchifyPredToCase {
    candidates: Vec<Node>,
}

impl SwitchifyPredToCase {
    pub fn new() -> Self {
        Self {
            candidates: Vec::new(),
        }
    }

    pub fn switchify_in_fn(&mut self, rvsdg: &mut Rvsdg, function: Function) {
        let fn_node = rvsdg
            .get_function_node(function)
            .expect("function not registered");
        let mut collector = NodeCollector {
            candidates: &mut self.candidates,
        };

        collector.visit_node(rvsdg, fn_node);

        for node in self.candidates.drain(..) {
            try_switchify_pred_to_case(rvsdg, node);
        }
    }
}

pub fn transform_entry_points(module: &Module, rvsdg: &mut Rvsdg) {
    let mut switchifier = SwitchifyPredToCase::new();

    for (function, _) in module.entry_points.iter() {
        switchifier.switchify_in_fn(rvsdg, function);
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::rvsdg::{ValueInput, ValueOrigin};
    use crate::ty::{TY_BOOL, TY_DUMMY, TY_PREDICATE, TY_U32};
    use crate::{FnSig, Symbol};

    #[test]
    fn test_switchify_pred_to_case() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref("test"),
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

        let cases = [10, 20, 30];
        let pred_to_case_node = rvsdg.add_op_branch_selector_to_case(
            region,
            ValueInput::output(TY_PREDICATE, bool_pred_node, 0),
            cases,
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: pred_to_case_node,
                output: 0,
            },
        );

        let mut switchifier = SwitchifyPredToCase::new();
        switchifier.switchify_in_fn(&mut rvsdg, function);

        let result_origin = rvsdg[region].value_results()[0].origin;
        let ValueOrigin::Output {
            producer: switch_node,
            output: 0,
        } = result_origin
        else {
            panic!("region result should be connected to a switch node");
        };

        let switch_data = rvsdg[switch_node].expect_switch();

        assert_eq!(switch_data.branches().len(), 3);

        for (i, &expected_case) in cases.iter().enumerate() {
            let branch_region = switch_data.branches()[i];
            let branch_result = rvsdg[branch_region].value_results()[0].origin;

            let ValueOrigin::Output {
                producer: const_node,
                output: 0,
            } = branch_result
            else {
                panic!("branch result should be connected to a const node");
            };

            let const_data = rvsdg[const_node].expect_const_u32();

            assert_eq!(const_data.value(), expected_case);
        }

        assert!(
            !rvsdg.is_live_node(pred_to_case_node),
            "the pred-to-case node should be removed"
        );
    }
}
