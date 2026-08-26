use crate::rvsdg::{Connectivity, Node, NodeKind, Rvsdg, SimpleNode, ValueOrigin};

/// Retains an ordered subset of a switch node's branches and adjust its case selector.
///
/// `retained_branches` must contain at least two strictly increasing indices from the switch's
/// current branch list. The branch-selector must originate directly from an
/// [OpCaseToBranchSelector] node (the RVSDG must be in predicate-continuation form, see
/// [branch_selector_normalization). If only a single branch is to remain, use
/// [inline_switch_branch] instead.
///
/// [inline_switch_branch]: crate::rvsdg::transform::region_replication::inline_switch_branch
pub fn retain_switch_branches(rvsdg: &mut Rvsdg, switch_node: Node, retained_branches: &[usize]) {
    assert!(retained_branches.len() > 1);
    assert!(retained_branches.windows(2).all(|pair| pair[0] < pair[1]));

    let region = rvsdg[switch_node].region();
    let branch_count = rvsdg[switch_node].expect_switch().branches().len();

    assert!(
        retained_branches
            .iter()
            .all(|&branch| branch < branch_count)
    );

    let ValueOrigin::Output {
        producer,
        output: 0,
    } = rvsdg[switch_node].expect_switch().branch_selector().origin
    else {
        panic!("the RVSDG should be in predicate-continuation form");
    };

    let (input, encoding, cases) = match rvsdg[producer].kind() {
        NodeKind::Simple(SimpleNode::OpCaseToBranchSelector(selector)) => (
            rvsdg[producer].value_inputs()[0],
            selector.encoding(),
            selector.cases().to_vec(),
        ),
        NodeKind::Simple(SimpleNode::OpBoolToBranchSelector(_)) => {
            panic!("a boolean switch with multiple retained branches should be unchanged")
        }
        _ => panic!("the RVSDG should be in predicate-continuation form"),
    };

    let mut retained_cases = retained_branches
        .iter()
        .filter_map(|&branch| cases.get(branch).copied())
        .collect::<Vec<_>>();

    // The final branch acts as the "default" branch, so we should have any fewer case than
    // branches. If necessary, we can remove the final case to achieve this; if the case occurs, it
    // will still map correctly to the final branch.
    if retained_cases.len() == retained_branches.len() {
        retained_cases.pop();
    }

    let selector = rvsdg.add_op_case_to_branch_selector(region, input, encoding, retained_cases);

    rvsdg.permute_switch_branches(switch_node, retained_branches);
    rvsdg.reconnect_value_input(
        switch_node,
        0,
        ValueOrigin::Output {
            producer: selector,
            output: 0,
        },
    );
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::rvsdg::{ValueInput, ValueOutput};
    use crate::ty::{Int, TY_DUMMY, TY_PREDICATE, TY_U32};
    use crate::{FnSig, Function, Module, Symbol};

    #[test]
    fn retained_branches_do_not_include_default() {
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

        let selector_value = rvsdg.add_const_u32(body, 0);
        let selector = rvsdg.add_op_case_to_branch_selector(
            body,
            ValueInput::output(TY_U32, selector_value, 0),
            Int::U32,
            [0, 1, 2],
        );
        let switch = rvsdg.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, selector, 0)],
            Vec::<ValueOutput>::new(),
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch);
        let branch_1 = rvsdg.add_switch_branch(switch);
        let branch_2 = rvsdg.add_switch_branch(switch);
        let branch_3 = rvsdg.add_switch_branch(switch);

        retain_switch_branches(&mut rvsdg, switch, &[0, 2]);

        let switch_node = rvsdg[switch].expect_switch();

        assert_eq!(switch_node.branches(), &[branch_0, branch_2]);

        assert!(!rvsdg.is_live_region(branch_1));
        assert!(!rvsdg.is_live_region(branch_3));

        let ValueOrigin::Output {
            producer: selector,
            output: 0,
        } = switch_node.branch_selector().origin
        else {
            panic!("expected the branch selector to be an output");
        };

        assert_eq!(
            rvsdg[selector].expect_op_case_to_branch_selector().cases(),
            &[0]
        );
    }

    #[test]
    fn retained_branches_include_default() {
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

        let selector_value = rvsdg.add_const_u32(body, 0);
        let selector = rvsdg.add_op_case_to_branch_selector(
            body,
            ValueInput::output(TY_U32, selector_value, 0),
            Int::U32,
            [0, 1, 2],
        );
        let switch = rvsdg.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, selector, 0)],
            Vec::<ValueOutput>::new(),
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch);
        let branch_1 = rvsdg.add_switch_branch(switch);
        let branch_2 = rvsdg.add_switch_branch(switch);
        let branch_3 = rvsdg.add_switch_branch(switch);

        retain_switch_branches(&mut rvsdg, switch, &[1, 2, 3]);

        let switch_node = rvsdg[switch].expect_switch();

        assert_eq!(switch_node.branches(), &[branch_1, branch_2, branch_3]);

        assert!(!rvsdg.is_live_region(branch_0));

        let ValueOrigin::Output {
            producer: selector,
            output: 0,
        } = switch_node.branch_selector().origin
        else {
            panic!("expected the branch selector to be an output");
        };

        assert_eq!(
            rvsdg[selector].expect_op_case_to_branch_selector().cases(),
            &[1, 2]
        );
    }
}
