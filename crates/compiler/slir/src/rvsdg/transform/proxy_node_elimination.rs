use crate::Module;
use crate::rvsdg::{Node, NodeKind, Region, Rvsdg, SimpleNode};

fn process_node(rvsdg: &mut Rvsdg, node: Node) {
    match rvsdg[node].kind() {
        NodeKind::Switch(_) => process_switch_node(rvsdg, node),
        NodeKind::Loop(_) => process_loop_node(rvsdg, node),
        NodeKind::Simple(SimpleNode::ValueProxy(_)) => eliminate_proxy_node(rvsdg, node),
        _ => {}
    }
}

fn process_switch_node(rvsdg: &mut Rvsdg, switch_node: Node) {
    let branch_count = rvsdg[switch_node].expect_switch().branches().len();

    for i in (0..branch_count).rev() {
        let branch = rvsdg[switch_node].expect_switch().branches()[i];

        region_eliminate_proxy_nodes(rvsdg, branch);
    }
}

fn process_loop_node(rvsdg: &mut Rvsdg, loop_node: Node) {
    let loop_region = rvsdg[loop_node].expect_loop().loop_region();

    region_eliminate_proxy_nodes(rvsdg, loop_region);
}

fn eliminate_proxy_node(rvsdg: &mut Rvsdg, node: Node) {
    let node_data = &rvsdg[node];
    let region = node_data.region();
    let node_data = node_data.expect_value_proxy();
    let origin = node_data.input().origin;
    let user_count = node_data.output().users.len();

    for i in (0..user_count).rev() {
        let user = rvsdg[node].expect_value_proxy().output().users[i];

        rvsdg.reconnect_value_user(region, user, origin);
    }

    rvsdg.remove_node(node);
}

pub fn region_eliminate_proxy_nodes(rvsdg: &mut Rvsdg, region: Region) {
    let node_count = rvsdg[region].nodes().len();

    for i in (0..node_count).rev() {
        let node = rvsdg[region].nodes()[i];

        process_node(rvsdg, node);
    }
}

pub fn entry_points_eliminate_proxy_nodes(module: &Module, rvsdg: &mut Rvsdg) {
    let entry_points = module
        .entry_points
        .iter()
        .map(|(f, _)| rvsdg.get_function_node(f).unwrap())
        .collect::<Vec<_>>();

    for entry_point in entry_points {
        let body_region = rvsdg[entry_point].expect_function().body_region();

        region_eliminate_proxy_nodes(rvsdg, body_region);
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::rvsdg::{ValueInput, ValueOrigin, ValueOutput, ValueUser};
    use crate::ty::{TY_DUMMY, TY_PREDICATE, TY_U32};
    use crate::{BinaryOperator, FnArg, FnSig, Function, Symbol, thin_set};

    #[test]
    fn test_eliminate_proxy_node() {
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

        let node_0 = rvsdg.add_const_u32(region, 0);
        let node_1 = rvsdg.add_const_u32(region, 1);
        let proxy = rvsdg.add_value_proxy(region, ValueInput::output(TY_U32, node_0, 0));
        let add = rvsdg.add_op_binary(
            region,
            BinaryOperator::Add,
            ValueInput::output(TY_U32, proxy, 0),
            ValueInput::output(TY_U32, node_1, 0),
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: add,
                output: 0,
            },
        );

        region_eliminate_proxy_nodes(&mut rvsdg, region);

        assert!(!rvsdg.is_live_node(proxy));
        assert_eq!(
            &rvsdg[node_0].expect_const_u32().output().users,
            &thin_set![ValueUser::Input {
                consumer: add,
                input: 0,
            }]
        );
        assert_eq!(
            rvsdg[add].expect_op_binary().lhs_input().origin,
            ValueOrigin::Output {
                producer: node_0,
                output: 0
            }
        );
    }

    #[test]
    fn test_eliminate_proxy_node_inside_switch() {
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
                args: vec![FnArg {
                    ty: TY_PREDICATE,
                    shader_io_binding: None,
                }],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let switch_node = rvsdg.add_switch(
            region,
            vec![ValueInput::argument(TY_PREDICATE, 0)],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);

        let branch_0_node_0 = rvsdg.add_const_u32(branch_0, 0);
        let proxy = rvsdg.add_value_proxy(branch_0, ValueInput::output(TY_U32, branch_0_node_0, 0));

        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: proxy,
                output: 0,
            },
        );

        let branch_1 = rvsdg.add_switch_branch(switch_node);

        let branch_1_node_0 = rvsdg.add_const_u32(branch_1, 0);

        rvsdg.reconnect_region_result(
            branch_1,
            0,
            ValueOrigin::Output {
                producer: branch_1_node_0,
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

        region_eliminate_proxy_nodes(&mut rvsdg, region);

        assert!(!rvsdg.is_live_node(proxy));
        assert_eq!(
            &rvsdg[branch_0_node_0].expect_const_u32().output().users,
            &thin_set![ValueUser::Result(0)]
        );
        assert_eq!(
            rvsdg[branch_0].value_results()[0].origin,
            ValueOrigin::Output {
                producer: branch_0_node_0,
                output: 0
            }
        );
    }
}
