//! Replaces switch node outputs with scalar constants when every branch produces the same constant.
//!
//! Branch results are resolved through transparent value proxies, switch branch arguments, and
//! loop-invariant loop-region arguments. Note that this pass does not remove the switch output that
//! is replaced by the constant node; this is left to [dead_value_elimination].
//!
//! [dead_value_elimination]: crate::rvsdg::transform::dead_value_elimination

use rustc_hash::FxHashSet;

use crate::rvsdg::analyse::scalar_constant::ScalarConstant;
use crate::rvsdg::visit::region_nodes::RegionNodesVisitor;
use crate::rvsdg::{Connectivity, Node, NodeKind, Region, Rvsdg, SimpleNode, ValueOrigin, visit};

struct SwitchNodeCollector<'a> {
    queue: &'a mut Vec<Node>,
}

impl RegionNodesVisitor for SwitchNodeCollector<'_> {
    fn visit_node(&mut self, rvsdg: &Rvsdg, node: Node) {
        if rvsdg[node].is_switch() {
            self.queue.push(node);
        }

        visit::region_nodes::visit_node(self, rvsdg, node);
    }
}

struct ConstantResolver {
    visited: FxHashSet<(Region, ValueOrigin)>,
}

impl ConstantResolver {
    fn new() -> Self {
        Self {
            visited: FxHashSet::default(),
        }
    }

    fn resolve(
        &mut self,
        rvsdg: &Rvsdg,
        mut region: Region,
        mut origin: ValueOrigin,
    ) -> Option<ScalarConstant> {
        self.visited.clear();

        loop {
            if !self.visited.insert((region, origin)) {
                return None;
            }

            match origin {
                ValueOrigin::Output { producer, output } => {
                    if let Some(constant) = ScalarConstant::from_node(rvsdg, producer, output) {
                        return Some(constant);
                    }

                    if output == 0
                        && let NodeKind::Simple(SimpleNode::ValueProxy(proxy)) =
                            rvsdg[producer].kind()
                    {
                        region = rvsdg[producer].region();
                        origin = proxy.value_inputs()[0].origin;

                        continue;
                    }

                    return None;
                }
                ValueOrigin::Argument(argument) if region != rvsdg.global_region() => {
                    let owner = rvsdg[region].owner();
                    let outer_region = rvsdg[owner].region();

                    match rvsdg[owner].kind() {
                        NodeKind::Switch(switch) => {
                            origin = switch.value_inputs()[argument as usize + 1].origin;
                            region = outer_region;
                        }
                        NodeKind::Loop(loop_node) => {
                            let loop_region = loop_node.loop_region();
                            let result = argument as usize + 1;

                            if rvsdg[loop_region].value_results()[result].origin
                                != ValueOrigin::Argument(argument)
                            {
                                return None;
                            }

                            origin = loop_node.value_inputs()[argument as usize].origin;
                            region = outer_region;
                        }
                        _ => return None,
                    }
                }
                _ => return None,
            }
        }
    }
}

pub struct ConstSwitchOutputExtractor {
    switch_node_queue: Vec<Node>,
    resolver: ConstantResolver,
}

impl ConstSwitchOutputExtractor {
    pub fn new() -> Self {
        Self {
            switch_node_queue: Vec::new(),
            resolver: ConstantResolver::new(),
        }
    }

    pub fn process_region(&mut self, rvsdg: &mut Rvsdg, region: Region) -> bool {
        self.switch_node_queue.clear();

        SwitchNodeCollector {
            queue: &mut self.switch_node_queue,
        }
        .visit_region(rvsdg, region);

        let mut changed = false;

        while let Some(switch) = self.switch_node_queue.pop() {
            changed |= self.process_switch(rvsdg, switch);
        }

        changed
    }

    fn process_switch(&mut self, rvsdg: &mut Rvsdg, switch: Node) -> bool {
        let output_count = rvsdg[switch].value_outputs().len();
        let region = rvsdg[switch].region();

        let mut changed = false;

        for output in 0..output_count {
            if rvsdg[switch].value_outputs()[output].users.is_empty() {
                continue;
            }

            let branches = rvsdg[switch].expect_switch().branches().to_vec();

            let mut common = None;

            for branch in branches {
                let origin = rvsdg[branch].value_results()[output].origin;

                let Some(constant) = self.resolver.resolve(rvsdg, branch, origin) else {
                    common = None;

                    break;
                };

                if common.is_some_and(|common| common != constant) {
                    common = None;

                    break;
                }

                common = Some(constant);
            }

            if let Some(constant) = common {
                let constant_node = constant.add_to_region(rvsdg, region);

                rvsdg.reconnect_value_users(
                    region,
                    ValueOrigin::Output {
                        producer: switch,
                        output: output as u32,
                    },
                    ValueOrigin::Output {
                        producer: constant_node,
                        output: 0,
                    },
                );

                changed = true;
            }
        }

        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rvsdg::{StateOrigin, ValueInput, ValueOutput};
    use crate::ty::{TY_DUMMY, TY_PREDICATE, TY_U32};
    use crate::{FnSig, Function, Module, Symbol};

    #[test]
    fn extracts_uniform_constant_through_proxy_and_switch_argument() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref("f"),
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
        let (_, region) = rvsdg.register_function(&module, function, vec![]);

        let outer_constant = rvsdg.add_const_u32(region, 7);
        let selector = rvsdg.add_const_predicate(region, 0);
        let switch = rvsdg.add_switch(
            region,
            vec![
                ValueInput::output(TY_PREDICATE, selector, 0),
                ValueInput::output(TY_U32, outer_constant, 0),
            ],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch);
        let proxy = rvsdg.add_value_proxy(branch_0, ValueInput::argument(TY_U32, 0));

        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: proxy,
                output: 0,
            },
        );

        let branch_1 = rvsdg.add_switch_branch(switch);
        let branch_constant = rvsdg.add_const_u32(branch_1, 7);

        rvsdg.reconnect_region_result(
            branch_1,
            0,
            ValueOrigin::Output {
                producer: branch_constant,
                output: 0,
            },
        );

        let user = rvsdg.add_value_proxy(region, ValueInput::output(TY_U32, switch, 0));

        let changed = ConstSwitchOutputExtractor::new().process_region(&mut rvsdg, region);

        assert!(changed);

        let ValueOrigin::Output { producer, output } = rvsdg[user].value_inputs()[0].origin else {
            panic!("expected constant node output")
        };
        assert_eq!(
            ScalarConstant::from_node(&rvsdg, producer, output),
            Some(ScalarConstant::U32(7))
        );
    }

    #[test]
    fn extracts_constant_through_loop_invariant_argument() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref("f"),
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
        let (_, region) = rvsdg.register_function(&module, function, vec![]);

        let initial = rvsdg.add_const_u32(region, 7);
        let (loop_node, loop_region) =
            rvsdg.add_loop(region, vec![ValueInput::output(TY_U32, initial, 0)], None);

        let selector = rvsdg.add_const_predicate(loop_region, 0);
        let switch = rvsdg.add_switch(
            loop_region,
            vec![
                ValueInput::output(TY_PREDICATE, selector, 0),
                ValueInput::argument(TY_U32, 0),
            ],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch);

        rvsdg.reconnect_region_result(branch_0, 0, ValueOrigin::Argument(0));

        let branch_1 = rvsdg.add_switch_branch(switch);
        let branch_constant = rvsdg.add_const_u32(branch_1, 7);

        rvsdg.reconnect_region_result(
            branch_1,
            0,
            ValueOrigin::Output {
                producer: branch_constant,
                output: 0,
            },
        );

        let user = rvsdg.add_value_proxy(loop_region, ValueInput::output(TY_U32, switch, 0));

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

        let changed = ConstSwitchOutputExtractor::new().process_region(&mut rvsdg, region);

        assert!(changed);

        let ValueOrigin::Output { producer, output } = rvsdg[user].value_inputs()[0].origin else {
            panic!("expected constant node output")
        };
        assert_eq!(
            ScalarConstant::from_node(&rvsdg, producer, output),
            Some(ScalarConstant::U32(7))
        );
    }
}
