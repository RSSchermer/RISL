use rustc_hash::FxHashMap;

use crate::Module;
use crate::rvsdg::NodeKind::{
    Function, Loop, Simple, StorageBinding, Switch, UniformBinding, WorkgroupBinding,
};
use crate::rvsdg::SimpleNode::{
    ConstBool, ConstF32, ConstFallback, ConstI32, ConstPredicate, ConstPtr, ConstU32,
    OpAddPtrOffset, OpAlloca, OpBinary, OpBoolToSwitchPredicate, OpCall, OpCaseToSwitchPredicate,
    OpExtractElement, OpGetDiscriminant, OpGetPtrOffset, OpLoad, OpPtrDiscriminantPtr,
    OpPtrElementPtr, OpPtrVariantPtr, OpSetDiscriminant, OpStore, OpSwitchPredicateToCase,
    OpU32ToSwitchPredicate, OpUnary, Reaggregation, ValueProxy,
};
use crate::rvsdg::{
    Connectivity, Node, OpMatrix, OpVector, Region, Rvsdg, StateOrigin, ValueInput, ValueOrigin,
    ValueOutput,
};

struct RegionReplicator<'a, 'b> {
    module: &'a mut Module,
    rvsdg: &'b mut Rvsdg,
    src_region: Region,
    dst_region: Region,

    /// A mapping from the inlined function's arguments to the new value origins at the call site.
    ///
    /// Since arguments are identified by a contiguous range of indices, we can use a vec rather
    /// than a hash map to record this mapping.
    value_argument_mapping: Vec<ValueOrigin>,

    state_argument_mapping: Option<StateOrigin>,
    node_mapping: FxHashMap<Node, Node>,
}

impl<'a, 'b> RegionReplicator<'a, 'b> {
    fn new(
        module: &'a mut Module,
        rvsdg: &'b mut Rvsdg,
        src_region: Region,
        dst_region: Region,
        value_argument_mapping: Vec<ValueOrigin>,
        state_argument_mapping: Option<StateOrigin>,
    ) -> Self {
        Self {
            module,
            rvsdg,
            src_region,
            dst_region,
            value_argument_mapping,
            state_argument_mapping,
            node_mapping: Default::default(),
        }
    }

    fn replicate_region(&mut self) -> Vec<ValueOrigin> {
        let result_count = self.rvsdg[self.src_region].value_results().len();
        let mut result_mapping = Vec::with_capacity(result_count);

        for i in 0..result_count {
            let origin = self.rvsdg[self.src_region].value_results()[i].origin;

            if origin.is_placeholder() {
                panic!("cannot inline a function that still contains placeholder result origins");
            }

            match origin {
                ValueOrigin::Argument(i) => {
                    result_mapping.push(self.value_argument_mapping[i as usize]);
                }
                ValueOrigin::Output { producer, output } => {
                    let replicate_node = self.visit_node(producer);

                    result_mapping.push(ValueOrigin::Output {
                        producer: replicate_node,
                        output,
                    });
                }
            }
        }

        if let StateOrigin::Node(node) = *self.rvsdg[self.src_region].state_result() {
            self.visit_node(node);
        }

        result_mapping
    }

    fn visit_node(&mut self, node: Node) -> Node {
        // We want to replicate the function body in a bottom-up post-order; that is, we want to
        // start from the results, and then do a depth-first search up the inputs, until we reach
        // nodes that are either input-less, or for which all inputs are region arguments. If we
        // create nodes in this order at the call-site, then we guarantee that for nodes that
        // depend on other nodes, all such dependencies will have been created before the dependent
        // node. We maintain a node mapping so that we can map function body origins to call-site
        // origins. Because outputs can have multiple users, this can result in us visiting the same
        // node multiple times. However, we can get dual use out of a node mapping by using it as a
        // "visited" set.

        if let Some(replicate_node) = self.node_mapping.get(&node) {
            // We've already visited this node
            return *replicate_node;
        }

        // Make sure all dependencies are replicated first
        for i in 0..self.rvsdg[node].value_inputs().len() {
            let origin = self.rvsdg[node].value_inputs()[i].origin;

            if !origin.is_placeholder()
                && let ValueOrigin::Output { producer, .. } = origin
            {
                self.visit_node(producer);
            }
        }

        if let Some(state) = self.rvsdg[node].state()
            && let StateOrigin::Node(dependency) = state.origin
        {
            self.visit_node(dependency);
        }

        let replicate_node = self.replicate_node(node);

        self.node_mapping.insert(node, replicate_node);

        replicate_node
    }

    fn replicate_node(&mut self, node: Node) -> Node {
        use crate::rvsdg::rvsdg::NodeKind::*;
        use crate::rvsdg::rvsdg::SimpleNode::*;

        match self.rvsdg[node].kind() {
            Switch(_) => self.replicate_switch_node(node),
            Loop(_) => self.replicate_loop_node(node),
            Simple(ConstU32(_)) => self.replicate_const_u32_node(node),
            Simple(ConstI32(_)) => self.replicate_const_i32_node(node),
            Simple(ConstF32(_)) => self.replicate_const_f32_node(node),
            Simple(ConstBool(_)) => self.replicate_const_bool_node(node),
            Simple(ConstPredicate(_)) => self.replicate_const_predicate_node(node),
            Simple(ConstPtr(_)) => self.replicate_const_ptr_node(node),
            Simple(ConstFallback(_)) => self.replicate_const_fallback_node(node),
            Simple(OpAlloca(_)) => self.replicate_op_alloca_node(node),
            Simple(OpLoad(_)) => self.replicate_op_load_node(node),
            Simple(OpStore(_)) => self.replicate_op_store_node(node),
            Simple(OpPtrElementPtr(_)) => self.replicate_op_ptr_element_ptr_node(node),
            Simple(OpPtrDiscriminantPtr(_)) => self.replicate_op_ptr_discriminant_ptr_node(node),
            Simple(OpPtrVariantPtr(_)) => self.replicate_op_ptr_variant_ptr_node(node),
            Simple(OpExtractElement(_)) => self.replicate_op_extract_element(node),
            Simple(OpGetDiscriminant(_)) => self.replicate_op_get_discriminant_node(node),
            Simple(OpSetDiscriminant(_)) => self.replicate_op_set_discriminant_node(node),
            Simple(OpAddPtrOffset(_)) => self.replicate_op_add_ptr_offset_node(node),
            Simple(OpGetPtrOffset(_)) => self.replicate_op_get_ptr_offset_node(node),
            Simple(OpCall(_)) => self.replicate_op_call_node(node),
            Simple(OpCallBuiltin(_)) => self.replicate_op_call_builtin_node(node),
            Simple(OpUnary(_)) => self.replicate_op_unary_node(node),
            Simple(OpBinary(_)) => self.replicate_op_binary_node(node),
            Simple(OpVector(_)) => self.replicate_op_vector_node(node),
            Simple(OpMatrix(_)) => self.replicate_op_matrix_node(node),
            Simple(OpCaseToSwitchPredicate(_)) => {
                self.replicate_op_case_to_switch_predicate_node(node)
            }
            Simple(OpBoolToSwitchPredicate(_)) => {
                self.replicate_op_bool_to_switch_predicate_node(node)
            }
            Simple(OpU32ToSwitchPredicate(_)) => {
                self.replicate_op_u32_to_switch_predicate_node(node)
            }
            Simple(OpSwitchPredicateToCase(_)) => {
                self.replicate_op_switch_predicate_to_case_node(node)
            }
            Simple(OpConvertToU32(_)) => self.replicate_op_convert_to_u32_node(node),
            Simple(OpConvertToI32(_)) => self.replicate_op_convert_to_i32_node(node),
            Simple(OpConvertToF32(_)) => self.replicate_op_convert_to_f32_node(node),
            Simple(OpConvertToBool(_)) => self.replicate_op_convert_to_bool_node(node),
            Simple(ValueProxy(_)) => self.replicate_value_proxy_node(node),
            Simple(Reaggregation(_)) => self.replicate_reaggregation_node(node),
            Function(_) | UniformBinding(_) | StorageBinding(_) | WorkgroupBinding(_)
            | Constant(_) => {
                panic!("node kind should not appear inside a region")
            }
        }
    }

    fn replicate_switch_node(&mut self, node: Node) -> Node {
        let data = self.rvsdg[node].expect_switch();
        let value_inputs = data
            .value_inputs()
            .iter()
            .map(|input| self.mapped_value_input(input))
            .collect();
        let value_outputs = data
            .value_outputs()
            .iter()
            .map(|output| ValueOutput::new(output.ty))
            .collect();
        let state_origin = data
            .state()
            .map(|state| self.mapped_state_origin(&state.origin));

        let replicate_node =
            self.rvsdg
                .add_switch(self.dst_region, value_inputs, value_outputs, state_origin);

        // Replicate each of the node's branch regions
        let branch_count = self.rvsdg[node].expect_switch().branches().len();
        for i in 0..branch_count {
            let src_region = self.rvsdg[node].expect_switch().branches()[i];
            let replicate_region = self.rvsdg.add_switch_branch(replicate_node);
            let value_argument_mapping = (0..self.rvsdg[src_region].value_arguments().len())
                .map(|i| ValueOrigin::Argument(i as u32))
                .collect();

            let mut region_replicator = RegionReplicator::new(
                self.module,
                self.rvsdg,
                src_region,
                replicate_region,
                value_argument_mapping,
                Some(StateOrigin::Argument),
            );

            let result_mapping = region_replicator.replicate_region();

            // Connect the replicated region's results
            for (i, origin) in result_mapping.into_iter().enumerate() {
                self.rvsdg
                    .reconnect_region_result(replicate_region, i as u32, origin);
            }
        }

        replicate_node
    }

    fn replicate_loop_node(&mut self, node: Node) -> Node {
        let data = self.rvsdg[node].expect_loop();
        let value_inputs = data
            .value_inputs()
            .iter()
            .map(|input| self.mapped_value_input(input))
            .collect();
        let state_origin = data
            .state()
            .map(|state| self.mapped_state_origin(&state.origin));
        let src_region = data.loop_region();

        let (replicate_node, replicate_region) =
            self.rvsdg
                .add_loop(self.dst_region, value_inputs, state_origin);

        let value_argument_mapping = (0..self.rvsdg[src_region].value_arguments().len())
            .map(|i| ValueOrigin::Argument(i as u32))
            .collect();

        let mut region_replicator = RegionReplicator::new(
            self.module,
            self.rvsdg,
            src_region,
            replicate_region,
            value_argument_mapping,
            Some(StateOrigin::Argument),
        );

        let result_mapping = region_replicator.replicate_region();

        // Connect the replicated region's results
        for (i, origin) in result_mapping.into_iter().enumerate() {
            self.rvsdg
                .reconnect_region_result(replicate_region, i as u32, origin);
        }

        replicate_node
    }

    fn replicate_const_u32_node(&mut self, node: Node) -> Node {
        let value = self.rvsdg[node].expect_const_u32().value();

        self.rvsdg.add_const_u32(self.dst_region, value)
    }

    fn replicate_const_i32_node(&mut self, node: Node) -> Node {
        let value = self.rvsdg[node].expect_const_i32().value();

        self.rvsdg.add_const_i32(self.dst_region, value)
    }

    fn replicate_const_f32_node(&mut self, node: Node) -> Node {
        let value = self.rvsdg[node].expect_const_f32().value();

        self.rvsdg.add_const_f32(self.dst_region, value)
    }

    fn replicate_const_bool_node(&mut self, node: Node) -> Node {
        let value = self.rvsdg[node].expect_const_bool().value();

        self.rvsdg.add_const_bool(self.dst_region, value)
    }

    fn replicate_const_predicate_node(&mut self, node: Node) -> Node {
        let value = self.rvsdg[node].expect_const_predicate().value();

        self.rvsdg.add_const_predicate(self.dst_region, value)
    }

    fn replicate_const_ptr_node(&mut self, node: Node) -> Node {
        let data = self.rvsdg[node].expect_const_ptr();
        let pointee_ty = data.pointee_ty();
        let base = self.mapped_value_input(data.base());

        self.rvsdg.add_const_ptr(self.dst_region, pointee_ty, base)
    }

    fn replicate_const_fallback_node(&mut self, node: Node) -> Node {
        let ty = self.rvsdg[node].expect_const_fallback().ty();

        self.rvsdg.add_const_fallback(self.dst_region, ty)
    }

    fn replicate_op_alloca_node(&mut self, node: Node) -> Node {
        let ty = self.rvsdg[node].expect_op_alloca().ty();

        self.rvsdg.add_op_alloca(self.dst_region, ty)
    }

    fn replicate_op_load_node(&mut self, node: Node) -> Node {
        let data = self.rvsdg[node].expect_op_load();
        let ptr_input = self.mapped_value_input(data.ptr_input());
        let state_origin = self.mapped_state_origin(&data.state().unwrap().origin);

        self.rvsdg
            .add_op_load(self.dst_region, ptr_input, state_origin)
    }

    fn replicate_op_store_node(&mut self, node: Node) -> Node {
        let data = self.rvsdg[node].expect_op_store();
        let ptr_input = self.mapped_value_input(data.ptr_input());
        let value_input = self.mapped_value_input(data.value_input());
        let state_origin = self.mapped_state_origin(&data.state().unwrap().origin);

        self.rvsdg
            .add_op_store(self.dst_region, ptr_input, value_input, state_origin)
    }

    fn replicate_op_ptr_element_ptr_node(&mut self, node: Node) -> Node {
        let data = self.rvsdg[node].expect_op_ptr_element_ptr();
        let ptr_input = self.mapped_value_input(data.ptr_input());
        let index_inputs = data
            .index_inputs()
            .iter()
            .map(|input| self.mapped_value_input(input))
            .collect::<Vec<_>>();

        self.rvsdg
            .add_op_ptr_element_ptr(self.dst_region, ptr_input, index_inputs)
    }

    fn replicate_op_ptr_discriminant_ptr_node(&mut self, node: Node) -> Node {
        let data = self.rvsdg[node].expect_op_ptr_discriminant_ptr();
        let input = self.mapped_value_input(data.input());

        self.rvsdg
            .add_op_ptr_discriminant_ptr(self.dst_region, input)
    }

    fn replicate_op_ptr_variant_ptr_node(&mut self, node: Node) -> Node {
        let data = self.rvsdg[node].expect_op_ptr_variant_ptr();
        let input = self.mapped_value_input(data.input());
        let variant_index = data.variant_index();

        self.rvsdg
            .add_op_ptr_variant_ptr(self.dst_region, input, variant_index)
    }

    fn replicate_op_extract_element(&mut self, node: Node) -> Node {
        let data = self.rvsdg[node].expect_op_extract_element();
        let aggregate_input = self.mapped_value_input(data.aggregate());
        let index_inputs = data
            .indices()
            .iter()
            .map(|input| self.mapped_value_input(input))
            .collect::<Vec<_>>();

        self.rvsdg
            .add_op_extract_element(self.dst_region, aggregate_input, index_inputs)
    }

    fn replicate_op_get_discriminant_node(&mut self, node: Node) -> Node {
        let data = self.rvsdg[node].expect_op_get_discriminant();
        let ptr_input = self.mapped_value_input(data.input());
        let state_origin = self.mapped_state_origin(&data.state().unwrap().origin);

        self.rvsdg
            .add_op_get_discriminant(self.dst_region, ptr_input, state_origin)
    }

    fn replicate_op_set_discriminant_node(&mut self, node: Node) -> Node {
        let data = self.rvsdg[node].expect_op_set_discriminant();
        let ptr_input = self.mapped_value_input(data.input());
        let variant_index = data.variant_index();
        let state_origin = self.mapped_state_origin(&data.state().unwrap().origin);

        self.rvsdg
            .add_op_set_discriminant(self.dst_region, ptr_input, variant_index, state_origin)
    }

    fn replicate_op_add_ptr_offset_node(&mut self, node: Node) -> Node {
        let data = self.rvsdg[node].expect_op_add_ptr_offset();
        let slice_ptr = self.mapped_value_input(data.slice_ptr());
        let offset = self.mapped_value_input(data.offset());

        self.rvsdg
            .add_op_add_ptr_offset(self.dst_region, slice_ptr, offset)
    }

    fn replicate_op_get_ptr_offset_node(&mut self, node: Node) -> Node {
        let data = self.rvsdg[node].expect_op_get_ptr_offset();
        let slice_ptr = self.mapped_value_input(data.ptr());

        self.rvsdg.add_op_get_ptr_offset(self.dst_region, slice_ptr)
    }

    fn replicate_op_call_node(&mut self, node: Node) -> Node {
        let data = self.rvsdg[node].expect_op_call();
        let fn_input = self.mapped_value_input(data.fn_input());
        let argument_inputs = data
            .argument_inputs()
            .iter()
            .map(|input| self.mapped_value_input(input))
            .collect::<Vec<_>>();
        let state_origin = self.mapped_state_origin(&data.state().unwrap().origin);

        self.rvsdg.add_op_call(
            self.module,
            self.dst_region,
            fn_input,
            argument_inputs,
            state_origin,
        )
    }

    fn replicate_op_call_builtin_node(&mut self, node: Node) -> Node {
        let data = self.rvsdg[node].expect_op_call_builtin();
        let builtin_function = data.callee().clone();
        let argument_inputs = data
            .argument_inputs()
            .iter()
            .map(|input| self.mapped_value_input(input))
            .collect::<Vec<_>>();

        self.rvsdg.add_op_call_builtin(
            self.module,
            self.dst_region,
            builtin_function,
            argument_inputs,
        )
    }

    fn replicate_op_unary_node(&mut self, node: Node) -> Node {
        let data = self.rvsdg[node].expect_op_unary();
        let operator = data.operator();
        let input = self.mapped_value_input(data.input());

        self.rvsdg.add_op_unary(self.dst_region, operator, input)
    }

    fn replicate_op_binary_node(&mut self, node: Node) -> Node {
        let data = self.rvsdg[node].expect_op_binary();
        let operator = data.operator();
        let lhs_input = self.mapped_value_input(data.lhs_input());
        let rhs_input = self.mapped_value_input(data.rhs_input());

        self.rvsdg
            .add_op_binary(self.dst_region, operator, lhs_input, rhs_input)
    }

    fn replicate_op_vector_node(&mut self, node: Node) -> Node {
        let data = self.rvsdg[node].expect_op_vector();
        let vector_ty = *data.vector_ty();
        let inputs = data
            .value_inputs()
            .iter()
            .map(|input| self.mapped_value_input(input))
            .collect::<Vec<_>>();

        self.rvsdg.add_op_vector(self.dst_region, vector_ty, inputs)
    }

    fn replicate_op_matrix_node(&mut self, node: Node) -> Node {
        let data = self.rvsdg[node].expect_op_matrix();
        let matrix_ty = *data.matrix_ty();
        let inputs = data
            .value_inputs()
            .iter()
            .map(|input| self.mapped_value_input(input))
            .collect::<Vec<_>>();

        self.rvsdg.add_op_matrix(self.dst_region, matrix_ty, inputs)
    }

    fn replicate_op_case_to_switch_predicate_node(&mut self, node: Node) -> Node {
        let data = self.rvsdg[node].expect_op_case_to_switch_predicate();
        let input = self.mapped_value_input(data.input());
        let cases = data.cases().to_vec();

        self.rvsdg
            .add_op_case_to_switch_predicate(self.dst_region, input, cases)
    }

    fn replicate_op_bool_to_switch_predicate_node(&mut self, node: Node) -> Node {
        let data = self.rvsdg[node].expect_op_bool_to_switch_predicate();
        let input = self.mapped_value_input(data.input());

        self.rvsdg
            .add_op_bool_to_switch_predicate(self.dst_region, input)
    }

    fn replicate_op_u32_to_switch_predicate_node(&mut self, node: Node) -> Node {
        let data = self.rvsdg[node].expect_op_u32_to_switch_predicate();
        let branch_count = data.branch_count();
        let input = self.mapped_value_input(data.input());

        self.rvsdg
            .add_op_u32_to_switch_predicate(self.dst_region, branch_count, input)
    }

    fn replicate_op_switch_predicate_to_case_node(&mut self, node: Node) -> Node {
        let data = self.rvsdg[node].expect_op_switch_predicate_to_case();
        let input = self.mapped_value_input(data.input());
        let cases = data.cases().to_vec();

        self.rvsdg
            .add_op_switch_predicate_to_case(self.dst_region, input, cases)
    }

    fn replicate_op_convert_to_u32_node(&mut self, node: Node) -> Node {
        let data = self.rvsdg[node].expect_op_convert_to_u32();
        let input = self.mapped_value_input(data.input());

        self.rvsdg.add_op_convert_to_u32(self.dst_region, input)
    }

    fn replicate_op_convert_to_i32_node(&mut self, node: Node) -> Node {
        let data = self.rvsdg[node].expect_op_convert_to_i32();
        let input = self.mapped_value_input(data.input());

        self.rvsdg.add_op_convert_to_i32(self.dst_region, input)
    }

    fn replicate_op_convert_to_f32_node(&mut self, node: Node) -> Node {
        let data = self.rvsdg[node].expect_op_convert_to_f32();
        let input = self.mapped_value_input(data.input());

        self.rvsdg.add_op_convert_to_f32(self.dst_region, input)
    }

    fn replicate_op_convert_to_bool_node(&mut self, node: Node) -> Node {
        let data = self.rvsdg[node].expect_op_convert_to_bool();
        let input = self.mapped_value_input(data.input());

        self.rvsdg.add_op_convert_to_bool(self.dst_region, input)
    }

    fn replicate_value_proxy_node(&mut self, node: Node) -> Node {
        let data = self.rvsdg[node].expect_value_proxy();
        let proxy_kind = data.proxy_kind();
        let input = self.mapped_value_input(data.input());

        self.rvsdg
            .add_value_proxy(self.dst_region, input, proxy_kind)
    }

    fn replicate_reaggregation_node(&mut self, node: Node) -> Node {
        let data = self.rvsdg[node].expect_reaggregation();
        let original_input = self.mapped_value_input(data.original());
        let part_inputs = data
            .parts()
            .iter()
            .map(|input| self.mapped_value_input(input))
            .collect::<Vec<_>>();

        self.rvsdg
            .add_reaggregation(self.dst_region, original_input, part_inputs)
    }

    fn mapped_value_input(&self, input: &ValueInput) -> ValueInput {
        let origin = match input.origin {
            ValueOrigin::Argument(i) => self.value_argument_mapping[i as usize],
            ValueOrigin::Output { producer, output } => ValueOrigin::Output {
                producer: self
                    .node_mapping
                    .get(&producer)
                    .copied()
                    .expect("producer should have been visited earlier"),
                output,
            },
        };

        ValueInput {
            ty: input.ty,
            origin,
        }
    }

    fn mapped_state_origin(&self, origin: &StateOrigin) -> StateOrigin {
        match *origin {
            StateOrigin::Argument => self
                .state_argument_mapping
                .expect("a state argument should have been provided for a region that uses state"),
            StateOrigin::Node(node) => StateOrigin::Node(
                self.node_mapping
                    .get(&node)
                    .copied()
                    .expect("dependency should have been visited earlier"),
            ),
        }
    }
}

pub fn replicate_region(
    module: &mut Module,
    rvsdg: &mut Rvsdg,
    src_region: Region,
    dst_region: Region,
    value_argument_mapping: Vec<ValueOrigin>,
    state_argument_mapping: Option<StateOrigin>,
) -> Vec<ValueOrigin> {
    let mut region_replicator = RegionReplicator::new(
        module,
        rvsdg,
        src_region,
        dst_region,
        value_argument_mapping,
        state_argument_mapping,
    );

    region_replicator.replicate_region()
}
