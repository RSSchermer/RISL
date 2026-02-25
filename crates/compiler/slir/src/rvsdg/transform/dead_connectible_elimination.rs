use std::mem;

use rustc_hash::FxHashSet;

use crate::Module;
use crate::rvsdg::{Connectivity, Node, NodeKind, Region, Rvsdg, StateUser, ValueOrigin};

pub struct DeadConnectibleEliminator {
    current_candidates: FxHashSet<Node>,
    next_candidates: FxHashSet<Node>,
}

impl DeadConnectibleEliminator {
    pub fn new() -> Self {
        Self {
            current_candidates: FxHashSet::default(),
            next_candidates: FxHashSet::default(),
        }
    }

    pub fn process_region(&mut self, rvsdg: &mut Rvsdg, region: Region) {
        // We reuse these candidate buffers to avoid allocating new ones for each region. This means
        // that their capacity is not necessarily equal. The first pass over the region should
        // find the most candidates, so we might avoid a reallocation if we make sure to use the
        // bigger buffer for this pass.
        if self.current_candidates.capacity() > self.next_candidates.capacity() {
            mem::swap(&mut self.current_candidates, &mut self.next_candidates);
        }

        // The initial pass checks all nodes in the region, and all sub-regions, removing any nodes
        // that are found to be "dead". Removing these nodes may render any origin nodes "dead" as
        // well. To avoid having to check all nodes again, we store only origin nodes from nodes
        // that are removed as "candidates" for further processing.
        process_region(rvsdg, region, &mut self.next_candidates);

        // We now process all candidates identified by the initial pass and remove any nodes that
        // are now dead. This may again find additional candidates, so we repeat this process until
        // there are no more candidates.
        while !self.next_candidates.is_empty() {
            mem::swap(&mut self.current_candidates, &mut self.next_candidates);

            for node in self.current_candidates.drain() {
                process_node(rvsdg, node, &mut self.next_candidates, false);
            }
        }
    }
}

fn candidate_from_origin(rvsdg: &Rvsdg, region: Region, origin: ValueOrigin) -> Node {
    match origin {
        ValueOrigin::Argument(_) => rvsdg[region].owner(),
        ValueOrigin::Output { producer, .. } => producer,
    }
}

fn argument_is_dead(rvsdg: &Rvsdg, region: Region, argument: u32) -> bool {
    rvsdg[region].value_arguments()[argument as usize]
        .users
        .is_empty()
}

fn output_is_dead(rvsdg: &Rvsdg, node: Node, output: u32) -> bool {
    rvsdg[node].value_outputs()[output as usize]
        .users
        .is_empty()
}

fn remove_node(rvsdg: &mut Rvsdg, node: Node, candidates: &mut FxHashSet<Node>) {
    let region = rvsdg[node].region();

    for input in rvsdg[node].value_inputs() {
        candidates.insert(candidate_from_origin(rvsdg, region, input.origin));
    }

    rvsdg.remove_node(node);
}

fn process_region(rvsdg: &mut Rvsdg, region: Region, candidates: &mut FxHashSet<Node>) {
    let node_count = rvsdg[region].nodes().len();

    for i in (0..node_count).rev() {
        let node = rvsdg[region].nodes()[i];

        process_node(rvsdg, node, candidates, true)
    }
}

fn process_node(
    rvsdg: &mut Rvsdg,
    node: Node,
    candidates: &mut FxHashSet<Node>,
    process_subregions: bool,
) {
    if rvsdg.is_live_node(node) {
        match rvsdg[node].kind() {
            NodeKind::Switch(_) => process_switch_node(rvsdg, node, candidates, process_subregions),
            NodeKind::Loop(_) => process_loop_node(rvsdg, node, candidates, process_subregions),
            NodeKind::Simple(_) => process_simple_node(rvsdg, node, candidates),
            _ => {}
        }
    }
}

fn process_switch_node(
    rvsdg: &mut Rvsdg,
    node: Node,
    candidates: &mut FxHashSet<Node>,
    process_branches: bool,
) {
    let switch_node = rvsdg[node].expect_switch();
    let argument_count = switch_node.value_inputs().len() - 1;
    let output_count = switch_node.value_outputs().len();
    let branch_count = switch_node.branches().len();

    // First check if the node is dead (all value outputs are dead and all branches are stateless),
    // because if it is, we can simply remove the node without processing individual arguments and
    // outputs.
    let all_outputs_dead = switch_node
        .value_outputs()
        .iter()
        .all(|o| o.users.is_empty());
    let is_stateless = switch_node
        .branches()
        .iter()
        .all(|b| *rvsdg[*b].state_argument() == StateUser::Result);

    if all_outputs_dead && is_stateless {
        remove_node(rvsdg, node, candidates);

        return;
    }

    // The switch node is not dead, so now we need to process individual arguments and outputs.

    for argument in (0..argument_count).rev() {
        try_remove_switch_argument(rvsdg, node, argument as u32, candidates);
    }

    for output in (0..output_count).rev() {
        try_remove_switch_output(rvsdg, node, output as u32, candidates);
    }

    // If this function was called as part of the initial recursive region processing, we also have
    // to process the branch regions
    if process_branches {
        for i in 0..branch_count {
            let branch = rvsdg[node].expect_switch().branches()[i];

            process_region(rvsdg, branch, candidates);
        }
    }
}

fn try_remove_switch_output(
    rvsdg: &mut Rvsdg,
    node: Node,
    output: u32,
    candidates: &mut FxHashSet<Node>,
) {
    if output_is_dead(rvsdg, node, output) {
        for branch in rvsdg[node].expect_switch().branches() {
            let result_origin = rvsdg[*branch].value_results()[output as usize].origin;

            candidates.insert(candidate_from_origin(rvsdg, *branch, result_origin));
        }

        rvsdg.remove_switch_output(node, output);
    }
}

fn try_remove_switch_argument(
    rvsdg: &mut Rvsdg,
    node: Node,
    argument: u32,
    candidates: &mut FxHashSet<Node>,
) {
    let region = rvsdg[node].region();
    let switch_node = rvsdg[node].expect_switch();

    let is_dead = switch_node
        .branches()
        .iter()
        .all(|b| argument_is_dead(rvsdg, *b, argument));

    if is_dead {
        let input = argument + 1;
        let origin = switch_node.value_inputs()[input as usize].origin;

        candidates.insert(candidate_from_origin(rvsdg, region, origin));

        rvsdg.remove_switch_input(node, input);
    }
}

fn process_loop_node(
    rvsdg: &mut Rvsdg,
    node: Node,
    candidates: &mut FxHashSet<Node>,
    process_loop_region: bool,
) {
    let loop_node = rvsdg[node].expect_loop();
    let loop_region = loop_node.loop_region();
    let output_count = loop_node.value_outputs().len();

    // First check if the node is dead (all value outputs are dead and all branches are stateless),
    // because if it is, we can simply remove the node without processing individual arguments and
    // outputs.
    let all_outputs_dead = loop_node.value_outputs().iter().all(|o| o.users.is_empty());
    let is_stateless = *rvsdg[loop_region].state_argument() == StateUser::Result;

    if all_outputs_dead && is_stateless {
        remove_node(rvsdg, node, candidates);

        return;
    }

    // The loop node is not dead, so now we need to process individual arguments and outputs.
    for i in (0..output_count).rev() {
        try_remove_loop_value(rvsdg, node, i as u32, candidates);
    }

    // If this function was called as part of the initial recursive region processing, we also have
    // to process the loop region.
    if process_loop_region {
        process_region(rvsdg, loop_region, candidates);
    }
}

fn try_remove_loop_value(
    rvsdg: &mut Rvsdg,
    node: Node,
    value: u32,
    candidates: &mut FxHashSet<Node>,
) {
    let result_index = value as usize + 1;

    let region = rvsdg[node].region();
    let loop_node = rvsdg[node].expect_loop();
    let loop_region = loop_node.loop_region();

    let arg_users = &rvsdg[loop_region].value_arguments()[value as usize].users;

    // TODO: more advanced analyses for arguments. Arguments that cannot reach outputs that have
    // users (including the state output), cannot affect program output and are therefore dead.
    let arg_is_dead = arg_users.is_empty();
    let output_is_dead = output_is_dead(rvsdg, node, value);

    if arg_is_dead && output_is_dead {
        let input_origin = loop_node.value_inputs()[value as usize].origin;

        candidates.insert(candidate_from_origin(rvsdg, region, input_origin));

        let result_origin = rvsdg[loop_region].value_results()[result_index].origin;

        if result_origin != ValueOrigin::Argument(value) {
            candidates.insert(candidate_from_origin(rvsdg, loop_region, result_origin));
        }

        rvsdg.remove_loop_input(node, value);
    }
}

fn process_simple_node(rvsdg: &mut Rvsdg, node: Node, candidates: &mut FxHashSet<Node>) {
    // We generally consider simple nodes to be alive if they are part of the state chain. The
    // exception is OpLoad, which we do consider dead if it value output has no users.
    //
    // TODO: make exception for OpAtomicLoad as well once we add that node type.
    if rvsdg[node].state().is_some() && !rvsdg[node].is_op_load() {
        return;
    }

    for output in rvsdg[node].value_outputs() {
        if !output.users.is_empty() {
            return;
        }
    }

    remove_node(rvsdg, node, candidates);
}

pub fn transform_entry_points(module: &mut Module, rvsdg: &mut Rvsdg) {
    let mut dce = DeadConnectibleEliminator::new();

    let entry_points = module
        .entry_points
        .iter()
        .map(|(f, _)| rvsdg.get_function_node(f).unwrap())
        .collect::<Vec<_>>();

    for entry_point in entry_points {
        let body_region = rvsdg[entry_point].expect_function().body_region();

        dce.process_region(rvsdg, body_region);
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::rvsdg::{ValueInput, ValueOutput};
    use crate::ty::{TY_DUMMY, TY_PREDICATE, TY_U32};
    use crate::{BinaryOperator, FnArg, FnSig, Function, Module, Symbol};

    #[test]
    fn test_eliminate_dead_node() {
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

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: node_1,
                output: 0,
            },
        );

        let mut dce = DeadConnectibleEliminator::new();

        dce.process_region(&mut rvsdg, region);

        assert!(!rvsdg.is_live_node(node_0));
        assert!(rvsdg.is_live_node(node_1));
    }

    #[test]
    fn test_eliminate_dependent_dead_node() {
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
                    ty: TY_U32,
                    shader_io_binding: None,
                }],
                ret_ty: None,
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let node_0 = rvsdg.add_const_u32(region, 1);
        let node_1 = rvsdg.add_op_binary(
            region,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 0),
            ValueInput::output(TY_U32, node_0, 0),
        );

        let mut dce = DeadConnectibleEliminator::new();

        dce.process_region(&mut rvsdg, region);

        assert!(!rvsdg.is_live_node(node_0));
        assert!(!rvsdg.is_live_node(node_1));
    }

    #[test]
    fn test_eliminate_dead_node_inside_switch() {
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
        let branch_0_node_1 = rvsdg.add_const_u32(branch_0, 1);

        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: branch_0_node_1,
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

        let mut dce = DeadConnectibleEliminator::new();

        dce.process_region(&mut rvsdg, region);

        assert!(!rvsdg.is_live_node(branch_0_node_0));
        assert!(rvsdg.is_live_node(branch_0_node_1));
        assert!(rvsdg.is_live_node(branch_1_node_0));
    }

    #[test]
    fn test_eliminate_dead_switch_node() {
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
                ret_ty: None,
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

        let branch_0_node_0 = rvsdg.add_const_u32(branch_0, 1);

        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: branch_0_node_0,
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

        let mut dce = DeadConnectibleEliminator::new();

        dce.process_region(&mut rvsdg, region);

        assert!(!rvsdg.is_live_node(switch_node));
        assert!(!rvsdg.is_live_node(branch_0_node_0));
        assert!(!rvsdg.is_live_node(branch_1_node_0));
    }

    #[test]
    fn test_eliminate_dead_switch_output() {
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
            vec![ValueOutput::new(TY_U32), ValueOutput::new(TY_U32)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);

        let branch_0_node_0 = rvsdg.add_const_u32(branch_0, 1);
        let branch_0_node_1 = rvsdg.add_const_u32(branch_0, 1);

        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: branch_0_node_0,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            branch_0,
            1,
            ValueOrigin::Output {
                producer: branch_0_node_1,
                output: 0,
            },
        );

        let branch_1 = rvsdg.add_switch_branch(switch_node);

        let branch_1_node_0 = rvsdg.add_const_u32(branch_1, 0);
        let branch_1_node_1 = rvsdg.add_const_u32(branch_1, 0);

        rvsdg.reconnect_region_result(
            branch_1,
            0,
            ValueOrigin::Output {
                producer: branch_1_node_0,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            branch_1,
            1,
            ValueOrigin::Output {
                producer: branch_1_node_1,
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

        let mut dce = DeadConnectibleEliminator::new();

        dce.process_region(&mut rvsdg, region);

        assert!(rvsdg.is_live_node(switch_node));
        assert!(rvsdg.is_live_node(branch_0_node_0));
        assert!(!rvsdg.is_live_node(branch_0_node_1));
        assert!(rvsdg.is_live_node(branch_1_node_0));
        assert!(!rvsdg.is_live_node(branch_1_node_1));

        assert_eq!(rvsdg[switch_node].value_outputs().len(), 1);
    }

    #[test]
    fn test_eliminate_dead_switch_argument() {
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

        let node_0 = rvsdg.add_const_u32(region, 0);

        let switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(TY_U32, node_0, 0),
            ],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);
        let branch_0_node_0 = rvsdg.add_const_u32(branch_0, 0);

        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: branch_0_node_0,
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

        let mut dce = DeadConnectibleEliminator::new();

        dce.process_region(&mut rvsdg, region);

        assert!(rvsdg.is_live_node(switch_node));
        assert!(!rvsdg.is_live_node(node_0));

        assert_eq!(rvsdg[switch_node].value_inputs().len(), 1);
    }

    #[test]
    fn test_eliminate_dead_loop_value() {
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
                    ty: TY_U32,
                    shader_io_binding: None,
                }],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let counter = rvsdg.add_const_u32(region, 0);
        let dead_input = rvsdg.add_const_u32(region, 0);

        let (loop_node, loop_region) = rvsdg.add_loop(
            region,
            vec![
                ValueInput::output(TY_U32, counter, 0),
                ValueInput::argument(TY_U32, 0),
                ValueInput::output(TY_U32, dead_input, 0),
            ],
            None,
        );

        let increment_node = rvsdg.add_const_u32(loop_region, 1);
        let counter_increment_node = rvsdg.add_op_binary(
            loop_region,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 0),
            ValueInput::output(TY_U32, increment_node, 0),
        );
        let reentry_test_node = rvsdg.add_op_binary(
            loop_region,
            BinaryOperator::Lt,
            ValueInput::output(TY_U32, counter_increment_node, 0),
            ValueInput::argument(TY_U32, 1),
        );

        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: reentry_test_node,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            loop_region,
            1,
            ValueOrigin::Output {
                producer: counter_increment_node,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: loop_node,
                output: 0,
            },
        );

        let mut dce = DeadConnectibleEliminator::new();

        dce.process_region(&mut rvsdg, region);

        assert!(rvsdg.is_live_node(loop_node));
        assert!(!rvsdg.is_live_node(dead_input));

        assert_eq!(rvsdg[loop_node].value_inputs().len(), 2);
        assert_eq!(rvsdg[loop_node].value_outputs().len(), 2);
    }
}
