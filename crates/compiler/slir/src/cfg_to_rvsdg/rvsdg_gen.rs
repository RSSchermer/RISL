use std::ops::Index;

use index_vec::IndexVec;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::cfg::analyze::item_dependencies::{Item, ItemDependencies, item_dependencies};
use crate::cfg::{
    Assign, BasicBlock, Bind, Cfg, InlineConst, IntrinsicOp, LocalBinding, OpCall, RootIdentifier,
    StatementData, Terminator, Uninitialized, Value,
};
use crate::cfg_to_rvsdg::control_flow_restructuring::{
    Graph, restructure_branches, restructure_loops,
};
use crate::cfg_to_rvsdg::control_tree::{
    BranchingNode, ControlTree, ControlTreeNode, ControlTreeNodeKind, LinearNode, LoopNode,
    SliceAnnotation, annotate_demand, annotate_item_dependencies, annotate_read_write,
    annotate_state_use,
};
use crate::intrinsic::Intrinsic;
use crate::rvsdg::{Node, Region, Rvsdg, StateOrigin, ValueInput, ValueOrigin, ValueOutput};
use crate::ty::{TY_BOOL, TY_F32, TY_I32, TY_U32, Type, TypeKind};
use crate::{Function, Module, rvsdg};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum InputState {
    Value(LocalBinding),
    Item(Item),
}

/// Keeps track of the most recent source of a [Value] or [Item] data dependency in a region's
/// RVSDG during construction.
///
/// When translating CFG construct to RVSDG constructs we need to map the [Value] and [Item]
/// operands of the CFG construct, to region arguments or sibling node outputs that represent the
/// corresponding data origin in the RVSDG. Whenever we begin a new region, we insert such mappings
/// into this tracker for each of the regions arguments. Subsequently, whenever a node is added to
/// the region that outputs a value, we insert another mapping or - if a mapping was already present
/// for the value/item (this can happen; we don't require the CFG be in SSA form) - we update the
/// existing mapping.
///
/// Whenever we add a node that requires inputs, we resolve them from the tracker. Walking the
/// [ControlTree] (that we use to construct the RVSDG) in post-order, should ensure that any input
/// requirements that a node may have, should have already been added to the tracker when visiting
/// prior nodes, thus such lookups should never fail.
#[derive(Clone, Debug)]
struct InputStateTracker {
    state: FxHashMap<InputState, ValueInput>,
    current_arg_index: u32,
}

impl InputStateTracker {
    fn new() -> Self {
        InputStateTracker {
            state: Default::default(),
            current_arg_index: 0,
        }
    }

    fn insert_value(&mut self, value: LocalBinding, input: ValueInput) {
        self.state.insert(InputState::Value(value), input);
    }

    fn insert_value_arg(&mut self, cfg: &Cfg, value: LocalBinding) {
        let ty = cfg[value].ty();
        let input = ValueInput {
            ty,
            origin: ValueOrigin::Argument(self.current_arg_index),
        };

        self.state.insert(InputState::Value(value), input);
        self.current_arg_index += 1;
    }

    fn insert_item_arg(&mut self, item: Item, ty: Type) {
        let input = ValueInput {
            ty,
            origin: ValueOrigin::Argument(self.current_arg_index),
        };

        self.state.insert(InputState::Item(item), input);
        self.current_arg_index += 1;
    }

    fn insert_value_node(&mut self, cfg: &Cfg, value: LocalBinding, producer: Node, output: u32) {
        let ty = cfg[value].ty();
        let input = ValueInput {
            ty,
            origin: ValueOrigin::Output { producer, output },
        };

        self.state.insert(InputState::Value(value), input);
    }
}

impl Index<LocalBinding> for InputStateTracker {
    type Output = ValueInput;

    fn index(&self, value: LocalBinding) -> &Self::Output {
        self.state
            .get(&InputState::Value(value))
            .expect("no input found for value")
    }
}

impl Index<Item> for InputStateTracker {
    type Output = ValueInput;

    fn index(&self, item: Item) -> &Self::Output {
        self.state
            .get(&InputState::Item(item))
            .expect("no input found for item")
    }
}

struct RegionBuilder<'a> {
    region: Region,
    module: &'a mut Module,
    control_tree: &'a ControlTree,
    cfg: &'a Cfg,
    item_dependencies: &'a SliceAnnotation<Item>,
    demand: &'a SliceAnnotation<LocalBinding>,
    state_use: &'a IndexVec<ControlTreeNode, bool>,
    rvsdg: &'a mut Rvsdg,
    input_state_tracker: InputStateTracker,
    state_origin: StateOrigin,
}

impl<'a> RegionBuilder<'a> {
    fn visit_node_expect_linear(&mut self, node: ControlTreeNode) {
        let data = self.control_tree[node].expect_linear();

        self.visit_linear_node(data);
    }

    fn visit_linear_node(&mut self, data: &LinearNode) {
        for i in 0..data.children.len() {
            let child = data.children[i];

            match &self.control_tree[child] {
                ControlTreeNodeKind::BasicBlock(bb) => self.visit_basic_block(*bb),
                ControlTreeNodeKind::Linear(child_data) => self.visit_linear_node(child_data),
                ControlTreeNodeKind::Branching(child_data) => {
                    let next_sibling = data.children.get(i + 1).copied();

                    self.visit_branching_node((child, child_data), next_sibling);
                }
                ControlTreeNodeKind::Loop(child_data) => self.visit_loop_node((child, child_data)),
            }
        }
    }

    fn visit_branching_node(
        &mut self,
        (node, data): (ControlTreeNode, &BranchingNode),
        next_sibling: Option<ControlTreeNode>,
    ) {
        let item_deps = self.item_dependencies.get(node);
        let demand = self.demand.get(node);
        let uses_state = self.state_use[node];

        let mut value_inputs = Vec::with_capacity(item_deps.len() + demand.len() + 1);

        value_inputs.push(self.input_state_tracker[data.selector]);

        // We need to construct the input state for the branch regions, based on the inputs to the
        // switch node. Each region builder for a branch's sub-region will start with a copy of this
        // tracker.
        let mut branch_input_state = InputStateTracker::new();

        for value in demand {
            value_inputs.push(self.input_state_tracker[*value]);
            branch_input_state.insert_value_arg(self.cfg, *value);
        }

        for dep in item_deps {
            value_inputs.push(self.input_state_tracker[*dep]);
            branch_input_state.insert_item_arg(*dep, dep.ty(self.module));
        }

        // The branching node needs to output the values that its next sibling demands.
        //
        // Note that we don't have a node handle for the switch node yet, so we'll have to update
        // the input state tracker later.
        let value_outputs = if let Some(next_sibling) = next_sibling {
            let next_sibling_demand = self.demand.get(next_sibling);

            next_sibling_demand
                .iter()
                .map(|value| ValueOutput::new(self.cfg[*value].ty()))
                .collect()
        } else {
            Vec::new()
        };

        let state_origin = uses_state.then(|| self.state_origin);

        // Add the switch node itself
        let node = self
            .rvsdg
            .add_switch(self.region, value_inputs, value_outputs, state_origin);

        // Add each of the branches
        for branch in &data.branches {
            let region = self.rvsdg.add_switch_branch(node);
            let mut branch_builder = self.subregion_builder(region, branch_input_state.clone());

            branch_builder.visit_node_expect_linear(*branch);

            // If the switch node has output demand, then connect the results of the branch region
            if let Some(next_sibling) = next_sibling {
                let next_sibling_demand = branch_builder.demand.get(next_sibling);

                for (i, value) in next_sibling_demand.iter().enumerate() {
                    branch_builder.connect_result(i as u32, value.into());
                }
            }
        }

        // Now that we have a node handle, update the input state tracker with the switch node's
        // outputs
        if let Some(next_sibling) = next_sibling {
            let next_sibling_demand = self.demand.get(next_sibling);

            for (i, value) in next_sibling_demand.iter().enumerate() {
                self.input_state_tracker
                    .insert_value_node(self.cfg, *value, node, i as u32);
            }
        }

        // Keep track of the state tail
        if uses_state {
            self.state_origin = StateOrigin::Node(node);
        }
    }

    fn visit_loop_node(&mut self, (node, data): (ControlTreeNode, &LoopNode)) {
        let item_deps = self.item_dependencies.get(node);
        let demand = self.demand.get(node);
        let uses_state = self.state_use[node];

        let mut value_inputs = Vec::with_capacity(item_deps.len() + demand.len());
        let mut inner_input_state = InputStateTracker::new();

        for value in demand {
            value_inputs.push(self.input_state_tracker[*value]);
            inner_input_state.insert_value_arg(self.cfg, *value);
        }

        for dep in item_deps {
            value_inputs.push(self.input_state_tracker[*dep]);
            inner_input_state.insert_item_arg(*dep, dep.ty(self.module));
        }

        let state_origin = uses_state.then(|| self.state_origin);

        let (node, region) = self.rvsdg.add_loop(self.region, value_inputs, state_origin);

        let mut inner_builder = self.subregion_builder(region, inner_input_state);

        inner_builder.visit_node_expect_linear(data.inner);

        // Connect the re-entry predicate result
        inner_builder.connect_result(0, data.reentry_predicate.into());

        // Connect the other results based on the demand.
        for (i, value) in demand.iter().enumerate() {
            // The first result is the re-entry predicate, the demand-based results follow, so shift
            // the index by 1.
            let result_index = i as u32 + 1;

            inner_builder.connect_result(result_index, value.into());
        }

        // Now that we have a node handle, update the input state tracker with the switch node's
        // outputs
        for (i, value) in demand.iter().enumerate() {
            self.input_state_tracker
                .insert_value_node(self.cfg, *value, node, i as u32);
        }

        // Keep track of the state tail
        if uses_state {
            self.state_origin = StateOrigin::Node(node);
        }
    }

    fn visit_basic_block(&mut self, bb: BasicBlock) {
        let data = &self.cfg[bb];

        for statement in data.statements() {
            self.visit_statement(&self.cfg[*statement]);
        }

        if let Terminator::Return(Some(value)) = data.terminator() {
            // Restructuring should have left only a single return terminator (if any), and it
            // should belong to the last child of the control tree's root linear node, so we know we
            // should currently be in a function's top-level region. We can therefor simply connect
            // the return value to result `0` of the current region.
            self.connect_result(0, *value);
        }
    }

    fn visit_statement(&mut self, statement: &StatementData) {
        match statement {
            StatementData::Assign(op) => self.visit_assign(op),
            StatementData::Bind(op) => self.visit_bind(op),
            StatementData::Uninitialized(op) => self.visit_uninitialized(op),
            StatementData::OpAlloca(op) => self.visit_intrinsic_op(op),
            StatementData::OpLoad(op) => self.visit_intrinsic_op(op),
            StatementData::OpStore(op) => self.visit_intrinsic_op(op),
            StatementData::OpExtractElement(op) => self.visit_intrinsic_op(op),
            StatementData::OpExtractField(op) => self.visit_intrinsic_op(op),
            StatementData::OpElementPtr(op) => self.visit_intrinsic_op(op),
            StatementData::OpFieldPtr(op) => self.visit_intrinsic_op(op),
            StatementData::OpVariantPtr(op) => self.visit_intrinsic_op(op),
            StatementData::OpGetDiscriminant(op) => self.visit_intrinsic_op(op),
            StatementData::OpSetDiscriminant(op) => self.visit_intrinsic_op(op),
            StatementData::OpOffsetSlice(op) => self.visit_intrinsic_op(op),
            StatementData::OpUnary(op) => self.visit_intrinsic_op(op),
            StatementData::OpBinary(op) => self.visit_intrinsic_op(op),
            StatementData::OpCaseToBranchSelector(op) => self.visit_intrinsic_op(op),
            StatementData::OpBoolToBranchSelector(op) => self.visit_intrinsic_op(op),
            StatementData::OpConvertToU32(op) => self.visit_intrinsic_op(op),
            StatementData::OpConvertToI32(op) => self.visit_intrinsic_op(op),
            StatementData::OpConvertToF32(op) => self.visit_intrinsic_op(op),
            StatementData::OpConvertToBool(op) => self.visit_intrinsic_op(op),
            StatementData::OpArrayLength(op) => self.visit_intrinsic_op(op),
            StatementData::OpCall(op) => self.visit_op_call(op),
        }
    }

    fn visit_assign(&mut self, op: &Assign) {
        // Assignment statements are not represented in the RVSDG, they are implicit in the data
        // flow. We instead redirect the tracker to the origin of the data that is being assigned.

        let input = self.resolve_value(op.value());

        self.input_state_tracker
            .insert_value(op.local_binding(), input);
    }

    fn visit_bind(&mut self, op: &Bind) {
        // Bind statements are not represented in the RVSDG, they are implicit in the data
        // flow. We instead redirect the tracker to the origin of the data that is being assigned.

        let input = self.resolve_value(op.value());

        self.input_state_tracker
            .insert_value(op.local_binding(), input);
    }

    fn visit_uninitialized(&mut self, _: &Uninitialized) {
        // Like assignment and bind statements, uninitialized statements are not represented in the
        // RVSDG. They also do not have an associated value, so we do not have to do anything here.
    }

    fn visit_intrinsic_op<T>(&mut self, op: &IntrinsicOp<T>)
    where
        T: Intrinsic + Clone,
        rvsdg::SimpleNode: From<rvsdg::IntrinsicNode<T>>,
    {
        let value_inputs: SmallVec<[ValueInput; 6]> = op
            .arguments()
            .iter()
            .map(|v| self.resolve_value(*v))
            .collect();

        let affects_state = op.intrinsic().affects_state();
        let state_origin = affects_state.then(|| self.state_origin);

        let node = self.rvsdg.add_intrinsic_op(
            self.region,
            op.intrinsic().clone(),
            value_inputs,
            state_origin,
        );

        if let Some(result) = op.maybe_result() {
            self.input_state_tracker
                .insert_value_node(self.cfg, result, node, 0);
        }

        if affects_state {
            self.state_origin = StateOrigin::Node(node);
        }
    }

    fn visit_op_call(&mut self, op: &OpCall) {
        let fn_input = self.input_state_tracker[Item::Function(op.callee())];
        let arg_inputs = op
            .arguments()
            .iter()
            .copied()
            .map(|v| self.resolve_value(v))
            .collect::<Vec<_>>();
        let node = self.rvsdg.add_op_call(
            self.module,
            self.region,
            fn_input,
            arg_inputs,
            self.state_origin,
        );

        if let Some(result) = op.maybe_result() {
            self.input_state_tracker
                .insert_value_node(self.cfg, result, node, 0);
        }

        self.state_origin = StateOrigin::Node(node);
    }

    fn connect_result(&mut self, result: u32, value: Value) {
        let input = self.resolve_value(value);

        self.rvsdg
            .reconnect_region_result(self.region, result, input.origin);
    }

    fn resolve_value(&mut self, value: Value) -> ValueInput {
        match value {
            Value::Local(v) => self.input_state_tracker[v],
            Value::InlineConst(c) => self.resolve_inline_const(c),
        }
    }

    fn resolve_inline_const(&mut self, c: InlineConst) -> ValueInput {
        let (ty, producer) = match c {
            InlineConst::U32(v) => (TY_U32, self.rvsdg.add_const_u32(self.region, v)),
            InlineConst::I32(v) => (TY_I32, self.rvsdg.add_const_i32(self.region, v)),
            InlineConst::F32(v) => (
                TY_F32,
                self.rvsdg.add_const_f32(self.region, v.into_inner()),
            ),
            InlineConst::Bool(v) => (TY_BOOL, self.rvsdg.add_const_bool(self.region, v)),
            InlineConst::Ptr(ptr) => {
                let base = self.resolve_root_identifier(ptr.root_identifier());
                let ty = ptr.ty();
                let pointee_ty = self.module.ty.kind(ty).expect_ptr();
                let node = self.rvsdg.add_const_ptr(self.region, pointee_ty, base);

                (ty, node)
            }
        };

        ValueInput {
            ty,
            origin: ValueOrigin::Output {
                producer,
                output: 0,
            },
        }
    }

    fn resolve_root_identifier(&self, root_identifier: RootIdentifier) -> ValueInput {
        match root_identifier {
            RootIdentifier::Local(v) => self.input_state_tracker[v],
            RootIdentifier::Uniform(b) => self.input_state_tracker[Item::UniformBinding(b)],
            RootIdentifier::Storage(b) => self.input_state_tracker[Item::StorageBinding(b)],
            RootIdentifier::Workgroup(b) => self.input_state_tracker[Item::WorkgroupBinding(b)],
            RootIdentifier::Constant(c) => self.input_state_tracker[Item::Constant(c)],
        }
    }

    fn subregion_builder(
        &mut self,
        region: Region,
        input_state_tracker: InputStateTracker,
    ) -> RegionBuilder {
        RegionBuilder {
            region,
            module: self.module,
            control_tree: self.control_tree,
            cfg: self.cfg,
            item_dependencies: self.item_dependencies,
            demand: self.demand,
            state_use: self.state_use,
            rvsdg: &mut self.rvsdg,
            input_state_tracker,
            state_origin: StateOrigin::Argument,
        }
    }
}

fn build_body(
    into: Region,
    function: Function,
    module: &mut Module,
    cfg: &mut Cfg,
    rvsdg: &mut Rvsdg,
    input_state_tracker: InputStateTracker,
) {
    let mut graph = Graph::init(cfg, function);

    let reentry_edges = restructure_loops(&mut graph);
    let branch_info = restructure_branches(&mut graph, &reentry_edges);

    let control_tree = ControlTree::generate(&graph, &reentry_edges, &branch_info);

    let item_dependencies = annotate_item_dependencies(&control_tree, &cfg);
    let (read, write) = annotate_read_write(&control_tree, &cfg);
    let demand = annotate_demand(&control_tree, &read, &write);
    let state_use = annotate_state_use(&control_tree, &cfg);

    let mut region_builder = RegionBuilder {
        region: into,
        module,
        control_tree: &control_tree,
        cfg: &cfg,
        item_dependencies: &item_dependencies,
        demand: &demand,
        state_use: &state_use,
        rvsdg,
        input_state_tracker,
        state_origin: StateOrigin::Argument,
    };

    region_builder.visit_node_expect_linear(control_tree.root());
}

fn add_item(
    item: Item,
    module: &mut Module,
    cfg: &mut Cfg,
    item_dependencies: &ItemDependencies,
    rvsdg: &mut Rvsdg,
    visited: &mut FxHashSet<Item>,
    item_node: &mut FxHashMap<Item, Node>,
) {
    if visited.insert(item) {
        let node = match item {
            Item::UniformBinding(binding) => rvsdg.register_uniform_binding(module, binding),
            Item::StorageBinding(binding) => rvsdg.register_storage_binding(module, binding),
            Item::WorkgroupBinding(binding) => rvsdg.register_workgroup_binding(module, binding),
            Item::Constant(v) => rvsdg.register_constant(module, v),
            Item::Function(function) => {
                let mut input_state_tracker = InputStateTracker::new();

                let (node, region) = if let Some(deps) = item_dependencies.get(&item) {
                    for dep in deps {
                        add_item(
                            *dep,
                            module,
                            cfg,
                            item_dependencies,
                            rvsdg,
                            visited,
                            item_node,
                        );

                        input_state_tracker.insert_item_arg(*dep, dep.ty(module));
                    }

                    let deps = deps.iter().map(|dep| item_node.get(dep).unwrap()).copied();

                    rvsdg.register_function(module, function, deps)
                } else {
                    rvsdg.register_function(module, function, [])
                };

                for param in cfg[function].argument_values() {
                    input_state_tracker.insert_value_arg(cfg, *param);
                }

                build_body(region, function, module, cfg, rvsdg, input_state_tracker);

                node
            }
        };

        item_node.insert(item, node);
    }
}

pub fn cfg_to_rvsdg(module: &mut Module, cfg: &mut Cfg) -> Rvsdg {
    let mut rvsdg = Rvsdg::new(module.ty.clone());
    let mut visited = FxHashSet::default();
    let mut item_node = FxHashMap::default();

    let item_dependencies = item_dependencies(cfg);

    for item in item_dependencies.keys() {
        add_item(
            *item,
            module,
            cfg,
            &item_dependencies,
            &mut rvsdg,
            &mut visited,
            &mut item_node,
        );
    }

    rvsdg
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::cfg::BlockPosition;
    use crate::ty::{TY_DUMMY, TY_PREDICATE};
    use crate::{BinaryOperator, FnArg, FnSig, Function, Symbol};

    #[test]
    fn test_single_bb() {
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
                args: vec![
                    FnArg {
                        ty: TY_U32,
                        shader_io_binding: None,
                    },
                    FnArg {
                        ty: TY_U32,
                        shader_io_binding: None,
                    },
                ],
                ret_ty: Some(TY_U32),
            },
        );

        let mut cfg = Cfg::new(module.ty.clone());

        let body = cfg.register_function(&module, function);

        let bb = body.entry_block();

        let a0 = body.argument_values()[0];
        let a1 = body.argument_values()[1];

        let (_, l0) = cfg.add_stmt_op_binary(
            bb,
            BlockPosition::Append,
            BinaryOperator::Add,
            a0.into(),
            5u32.into(),
        );
        let (_, l1) = cfg.add_stmt_op_binary(
            bb,
            BlockPosition::Append,
            BinaryOperator::Add,
            l0.into(),
            a1.into(),
        );

        cfg.set_terminator(bb, Terminator::return_value(l1.into()));

        let actual = cfg_to_rvsdg(&mut module, &mut cfg);

        let mut expected = Rvsdg::new(module.ty.clone());

        let (_, region) = expected.register_function(&module, function, iter::empty());

        let node_0 = expected.add_const_u32(region, 5);

        let node_1 = expected.add_op_binary(
            region,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 0),
            ValueInput::output(TY_U32, node_0, 0),
        );

        let node_2 = expected.add_op_binary(
            region,
            BinaryOperator::Add,
            ValueInput::output(TY_U32, node_1, 0),
            ValueInput::argument(TY_U32, 1),
        );

        expected.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: node_2,
                output: 0,
            },
        );

        dbg!(&actual);
        dbg!(&expected);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_simple_branch() {
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
                args: vec![
                    FnArg {
                        ty: TY_U32,
                        shader_io_binding: None,
                    },
                    FnArg {
                        ty: TY_U32,
                        shader_io_binding: None,
                    },
                ],
                ret_ty: Some(TY_U32),
            },
        );

        let mut cfg = Cfg::new(module.ty.clone());

        let body = cfg.register_function(&module, function);

        let a0 = body.argument_values()[0];
        let a1 = body.argument_values()[1];

        let bb0 = body.entry_block();
        let bb1 = cfg.add_basic_block(function);
        let bb2 = cfg.add_basic_block(function);
        let bb3 = cfg.add_basic_block(function);

        // BB0
        let (_, res) = cfg.add_stmt_uninitialized(bb0, BlockPosition::Append, TY_U32);
        let (_, predicate) =
            cfg.add_stmt_op_case_to_branch_selector(bb0, BlockPosition::Append, a0.into(), [0, 1]);
        cfg.set_terminator(bb0, Terminator::branch_multiple(predicate, [bb1, bb2]));

        // BB1
        let (_, add_res) = cfg.add_stmt_op_binary(
            bb1,
            BlockPosition::Append,
            BinaryOperator::Add,
            a1.into(),
            1u32.into(),
        );
        cfg.add_stmt_assign(bb1, BlockPosition::Append, res, add_res.into());
        cfg.set_terminator(bb1, Terminator::branch_single(bb3));

        // BB2
        cfg.add_stmt_assign(bb2, BlockPosition::Append, res, 0u32.into());
        cfg.set_terminator(bb2, Terminator::branch_single(bb3));

        // BB3
        cfg.set_terminator(bb3, Terminator::return_value(res.into()));

        let actual = cfg_to_rvsdg(&mut module, &mut cfg);

        let mut expected = Rvsdg::new(module.ty.clone());

        let (_, region) = expected.register_function(&module, function, iter::empty());

        let predicate_node = expected.add_op_case_to_branch_selector(
            region,
            ValueInput::argument(TY_U32, 0),
            [0, 1],
        );
        let switch_node = expected.add_switch(
            region,
            vec![
                ValueInput::output(TY_PREDICATE, predicate_node, 0),
                ValueInput::argument(TY_U32, 1),
            ],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        expected.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: switch_node,
                output: 0,
            },
        );

        let branch_0 = expected.add_switch_branch(switch_node);

        let branch_0_added_value_node = expected.add_const_u32(branch_0, 1);
        let branch_0_add_node = expected.add_op_binary(
            branch_0,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 0),
            ValueInput::output(TY_U32, branch_0_added_value_node, 0),
        );

        expected.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: branch_0_add_node,
                output: 0,
            },
        );

        let branch_1 = expected.add_switch_branch(switch_node);

        let branch_1_const_node = expected.add_const_u32(branch_1, 0);

        expected.reconnect_region_result(
            branch_1,
            0,
            ValueOrigin::Output {
                producer: branch_1_const_node,
                output: 0,
            },
        );

        dbg!(&actual);
        dbg!(&expected);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_simple_loop() {
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

        let mut cfg = Cfg::new(module.ty.clone());

        let body = cfg.register_function(&module, function);

        let a0 = body.argument_values()[0];

        let bb0 = body.entry_block();
        let bb1 = cfg.add_basic_block(function);

        // BB0
        let (_, add_res) = cfg.add_stmt_op_binary(
            bb0,
            BlockPosition::Append,
            BinaryOperator::Add,
            a0.into(),
            1u32.into(),
        );
        cfg.add_stmt_assign(bb0, BlockPosition::Append, a0, add_res.into());
        let (_, cmp) = cfg.add_stmt_op_binary(
            bb0,
            BlockPosition::Append,
            BinaryOperator::Gt,
            a0.into(),
            10u32.into(),
        );
        let (_, predicate) =
            cfg.add_stmt_op_bool_to_branch_selector(bb0, BlockPosition::Append, cmp.into());
        cfg.set_terminator(bb0, Terminator::branch_multiple(predicate, [bb1, bb0]));

        // BB1
        cfg.set_terminator(bb1, Terminator::return_value(a0.into()));

        let actual = cfg_to_rvsdg(&mut module, &mut cfg);

        let mut expected = Rvsdg::new(module.ty.clone());

        let (_, region) = expected.register_function(&module, function, iter::empty());

        let (loop_node, loop_region) =
            expected.add_loop(region, vec![ValueInput::argument(TY_U32, 0)], None);

        let added_value_node = expected.add_const_u32(loop_region, 1);
        let add_node = expected.add_op_binary(
            loop_region,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 0),
            ValueInput::output(TY_U32, added_value_node, 0),
        );
        let compare_value_node = expected.add_const_u32(loop_region, 10);
        let cmp_node = expected.add_op_binary(
            loop_region,
            BinaryOperator::Gt,
            ValueInput::output(TY_U32, add_node, 0),
            ValueInput::output(TY_U32, compare_value_node, 0),
        );
        let predicate_node = expected
            .add_op_bool_to_branch_selector(loop_region, ValueInput::output(TY_BOOL, cmp_node, 0));

        expected.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: predicate_node,
                output: 0,
            },
        );
        expected.reconnect_region_result(
            loop_region,
            1,
            ValueOrigin::Output {
                producer: add_node,
                output: 0,
            },
        );

        expected.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: loop_node,
                output: 0,
            },
        );

        dbg!(&actual);
        dbg!(&expected);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_stateful() {
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
                args: vec![
                    FnArg {
                        ty: module.ty.register(TypeKind::Ptr(TY_U32)),
                        shader_io_binding: None,
                    },
                    FnArg {
                        ty: TY_U32,
                        shader_io_binding: None,
                    },
                ],
                ret_ty: None,
            },
        );

        let mut cfg = Cfg::new(module.ty.clone());

        let body = cfg.register_function(&module, function);

        let a0 = body.argument_values()[0];
        let a1 = body.argument_values()[1];

        let bb = body.entry_block();

        let (_, loaded_value) = cfg.add_stmt_op_load(bb, BlockPosition::Append, a0.into());
        let (_, summed_value) = cfg.add_stmt_op_binary(
            bb,
            BlockPosition::Append,
            BinaryOperator::Add,
            loaded_value.into(),
            a1.into(),
        );
        cfg.add_stmt_op_store(bb, BlockPosition::Append, a0.into(), summed_value.into());
        cfg.set_terminator(bb, Terminator::return_void());

        let actual = cfg_to_rvsdg(&mut module, &mut cfg);

        let mut expected = Rvsdg::new(module.ty.clone());

        let (_, region) = expected.register_function(&module, function, iter::empty());

        let load_node = expected.add_op_load(
            region,
            ValueInput::argument(module.ty.register(TypeKind::Ptr(TY_U32)), 0),
            StateOrigin::Argument,
        );
        let add_node = expected.add_op_binary(
            region,
            BinaryOperator::Add,
            ValueInput::output(TY_U32, load_node, 0),
            ValueInput::argument(TY_U32, 1),
        );
        let store_node = expected.add_op_store(
            region,
            ValueInput::argument(module.ty.register(TypeKind::Ptr(TY_U32)), 0),
            ValueInput::output(TY_U32, add_node, 0),
            StateOrigin::Node(load_node),
        );

        dbg!(&actual);
        dbg!(&expected);

        assert_eq!(actual, expected);
    }
}
