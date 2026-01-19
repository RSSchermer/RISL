use std::collections::VecDeque;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::intrinsic::Intrinsic;
use crate::rvsdg::analyse::region_stratification::RegionStratifier;
use crate::rvsdg::{Connectivity, NodeKind, Rvsdg, SimpleNode, ValueOrigin};
use crate::scf::{BlockPosition, LocalBinding, LoopControl, Scf};
use crate::{
    Constant, Function, Module, StorageBinding, UniformBinding, WorkgroupBinding, rvsdg, scf,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Value {
    Local(LocalBinding),
    Uniform(UniformBinding),
    Storage(StorageBinding),
    Workgroup(WorkgroupBinding),
    Constant(Constant),
}

impl Value {
    fn expect_local(&self) -> LocalBinding {
        if let Value::Local(binding) = self {
            *binding
        } else {
            panic!("expected a local value");
        }
    }
}

impl From<LocalBinding> for Value {
    fn from(binding: LocalBinding) -> Self {
        Value::Local(binding)
    }
}

#[derive(Clone, Debug)]
struct ValueMapping {
    value_mapping: FxHashMap<rvsdg::ValueOrigin, Value>,
}

impl ValueMapping {
    fn new() -> Self {
        Self {
            value_mapping: Default::default(),
        }
    }

    fn map_argument(&mut self, argument: u32, value: Value) {
        self.value_mapping
            .insert(ValueOrigin::Argument(argument), value);
    }

    fn map_output(&mut self, node: rvsdg::Node, output: u32, value: LocalBinding) {
        self.value_mapping.insert(
            ValueOrigin::Output {
                producer: node,
                output,
            },
            value.into(),
        );
    }

    fn mapping(&self, origin: ValueOrigin) -> Value {
        *self
            .value_mapping
            .get(&origin)
            .expect("input values should have been recorded before visiting this node")
    }
}

struct PreparedBindIntrinsic<T> {
    node: rvsdg::Node,
    intrinsic: T,
    arguments: SmallVec<[LocalBinding; 6]>,
}

impl<T> PreparedBindIntrinsic<T>
where
    T: Intrinsic,
    scf::ExpressionKind: From<scf::IntrinsicOp<T>>,
{
    fn apply(self, visitor: &mut RegionVisitor) {
        let PreparedBindIntrinsic {
            node,
            intrinsic,
            arguments,
        } = self;

        let (_, binding) = visitor.scf.add_bind_intrinsic_op(
            visitor.dst_block,
            BlockPosition::Append,
            intrinsic,
            arguments,
        );

        visitor.value_mapping.map_output(node, 0, binding);
    }
}

struct RegionVisitor<'a, 'b, 'c> {
    module: &'a Module,
    rvsdg: &'a Rvsdg,
    region_stratifier: &'b mut RegionStratifier,
    scf: &'c mut Scf,
    region: rvsdg::Region,
    dst_block: scf::Block,
    value_mapping: ValueMapping,
    node_queue: VecDeque<rvsdg::Node>,
}

impl<'a, 'b, 'c> RegionVisitor<'a, 'b, 'c> {
    fn visit_region(&mut self) {
        // First organize the region's nodes into a queue sorted by stratum. This means that if we
        // process the queue front-to-back and record a mapping for each of the node's outputs,
        // every node is guaranteed to have a value mapping for each of its inputs.
        self.region_stratifier
            .stratify(&self.rvsdg, self.region, |node, _| {
                self.node_queue.push_back(node);
            });

        while let Some(node) = self.node_queue.pop_front() {
            self.visit_node(node)
        }
    }

    fn visit_node(&mut self, node: rvsdg::Node) {
        use crate::rvsdg::NodeKind::*;

        match self.rvsdg[node].kind() {
            Switch(_) => self.visit_switch_node(node),
            Loop(_) => self.visit_loop_node(node),
            Simple(_) => self.visit_simple_node(node),
            _ => unreachable!("node kind cannot be part of a function (sub)region"),
        }
    }

    fn visit_switch_node(&mut self, node: rvsdg::Node) {
        use NodeKind::*;
        use SimpleNode::*;

        let data = self.rvsdg[node].expect_switch();
        let predicate_origin = data.predicate().origin;
        let predicate_binding = self.value_mapping.mapping(predicate_origin).expect_local();

        match predicate_origin {
            ValueOrigin::Argument(_) => {
                self.generate_switch(node, predicate_binding, None);
            }
            ValueOrigin::Output { producer, .. } => match self.rvsdg[producer].kind() {
                Simple(OpCaseToBranchSelector(op)) => {
                    self.generate_switch(node, predicate_binding, Some(op.cases()));
                }
                Simple(OpBoolToBranchSelector(_)) => {
                    self.generate_if(node, predicate_binding);
                }
                Simple(OpU32ToBranchSelector(_)) => {
                    self.generate_switch(node, predicate_binding, None);
                }
                _ => panic!(
                    "a switch node's predicate input must directly connect to a \
                        predicate-generating operation before conversion to SCF"
                ),
            },
        };
    }

    fn generate_if(&mut self, switch_node: rvsdg::Node, condition: scf::LocalBinding) {
        let data = self.rvsdg[switch_node].expect_switch();

        assert_eq!(data.branches().len(), 2);

        let then_branch = data.branches()[0];
        let else_branch = data.branches()[1];

        let (if_stmt, then_block) =
            self.scf
                .add_if(self.dst_block, BlockPosition::Append, condition);
        let else_block = self.scf.add_else_block(if_stmt);

        // Add out variables to the if statement and record them in the value mapping.
        for (i, output) in data.value_outputs().iter().enumerate() {
            let binding = self.scf.add_if_out_var(if_stmt, output.ty);

            self.value_mapping
                .map_output(switch_node, i as u32, binding);
        }

        // Create an argument mapping. Since both the `then` and `else` branches receive the same
        // arguments, we construct it once, then pass a clone to each branch region visitor.
        let mut argument_mapping = ValueMapping::new();
        for (arg, input) in data.value_inputs()[1..].iter().enumerate() {
            let expr = self.value_mapping.mapping(input.origin);

            argument_mapping.map_argument(arg as u32, expr);
        }

        self.generate_branch(then_branch, then_block, argument_mapping.clone());
        self.generate_branch(else_branch, else_block, argument_mapping);
    }

    fn generate_switch(
        &mut self,
        switch_node: rvsdg::Node,
        on: scf::LocalBinding,
        cases: Option<&[u32]>,
    ) {
        let data = self.rvsdg[switch_node].expect_switch();

        let switch_stmt = self
            .scf
            .add_switch(self.dst_block, BlockPosition::Append, on);
        let default_block = self.scf[switch_stmt].kind().expect_switch().default();

        // Add out variables to the switch statement and record them in the value mapping.
        for (i, output) in data.value_outputs().iter().enumerate() {
            let binding = self.scf.add_switch_out_var(switch_stmt, output.ty);

            self.value_mapping
                .map_output(switch_node, i as u32, binding);
        }

        // Create an argument mapping. Since each branch receives the same arguments, we construct
        // it once, then pass a clone to each branch region visitor.
        let mut argument_mapping = ValueMapping::new();
        for (arg, input) in data.value_inputs()[1..].iter().enumerate() {
            let expr = self.value_mapping.mapping(input.origin);

            argument_mapping.map_argument(arg as u32, expr);
        }

        for (i, branch) in data.branches().iter().copied().enumerate() {
            let is_last = i == data.branches().len() - 1;

            // We use the switch's default block for the last case.
            let branch_block = if is_last {
                default_block
            } else {
                let case = cases.map(|cases| cases[i]).unwrap_or(i as u32);

                self.scf.add_switch_case(switch_stmt, case)
            };

            self.generate_branch(branch, branch_block, argument_mapping.clone());
        }
    }

    fn generate_branch(
        &mut self,
        branch_region: rvsdg::Region,
        branch_block: scf::Block,
        argument_mapping: ValueMapping,
    ) {
        let sub_region_mapping =
            self.visit_sub_region(branch_region, branch_block, argument_mapping);

        for (res, input) in self.rvsdg[branch_region].value_results().iter().enumerate() {
            let value = sub_region_mapping.mapping(input.origin).expect_local();

            self.scf.set_control_flow_var(branch_block, res, value);
        }
    }

    fn visit_loop_node(&mut self, node: rvsdg::Node) {
        let data = self.rvsdg[node].expect_loop();
        let loop_region = data.loop_region();
        let (loop_stmt, loop_block) = self.scf.add_loop(self.dst_block, BlockPosition::Append);
        let mut argument_mapping = ValueMapping::new();

        for (i, input) in data.value_inputs().iter().enumerate() {
            let initializer = self.value_mapping.mapping(input.origin).expect_local();
            let binding = self.scf.add_loop_var(loop_stmt, initializer);

            // Add a mapping for the loop value's corresponding argument.
            argument_mapping.map_argument(i as u32, binding.into());

            // Add a mapping for the loop value's corresponding output.
            self.value_mapping.map_output(node, i as u32, binding);
        }

        // Process the sub-region and use the value mapping it produces to set the loop block's
        // control expression and control-flow variables.
        let sub_region_mapping = self.visit_sub_region(loop_region, loop_block, argument_mapping);

        let control_origin = self.rvsdg[loop_region].value_results()[0].origin;
        let control_expr = sub_region_mapping.mapping(control_origin).expect_local();

        self.scf
            .set_loop_control(loop_stmt, LoopControl::Tail(control_expr));

        for (i, result) in self.rvsdg[loop_region].value_results()[1..]
            .iter()
            .enumerate()
        {
            let binding = sub_region_mapping.mapping(result.origin).expect_local();

            self.scf.set_control_flow_var(loop_block, i, binding);
        }
    }

    fn visit_simple_node(&mut self, node: rvsdg::Node) {
        let data = self.rvsdg[node].expect_simple();

        use crate::rvsdg::SimpleNode::*;

        match data {
            ConstU32(_) => self.visit_const_u32(node),
            ConstI32(_) => self.visit_const_i32(node),
            ConstF32(_) => self.visit_const_f32(node),
            ConstBool(_) => self.visit_const_bool(node),
            ConstPtr(_) => self.visit_const_ptr(node),
            ConstFallback(_) => self.visit_const_fallback(node),
            OpAlloca(_) => self.visit_op_alloca(node),
            OpStore(_) => self.visit_op_store(node),
            OpLoad(op) => self.prepare_bind_intrinsic(node, op).apply(self),
            OpFieldPtr(op) => self.prepare_bind_intrinsic(node, op).apply(self),
            OpElementPtr(op) => self.prepare_bind_intrinsic(node, op).apply(self),
            OpExtractField(op) => self.prepare_bind_intrinsic(node, op).apply(self),
            OpExtractElement(op) => self.prepare_bind_intrinsic(node, op).apply(self),
            OpUnary(op) => self.prepare_bind_intrinsic(node, op).apply(self),
            OpBinary(op) => self.prepare_bind_intrinsic(node, op).apply(self),
            OpVector(op) => self.prepare_bind_intrinsic(node, op).apply(self),
            OpMatrix(op) => self.prepare_bind_intrinsic(node, op).apply(self),
            OpConvertToU32(op) => self.prepare_bind_intrinsic(node, op).apply(self),
            OpConvertToI32(op) => self.prepare_bind_intrinsic(node, op).apply(self),
            OpConvertToF32(op) => self.prepare_bind_intrinsic(node, op).apply(self),
            OpConvertToBool(op) => self.prepare_bind_intrinsic(node, op).apply(self),

            // We don't express switch predicates in the SCF, instead we translate the switch nodes
            // either if statements or switch statements with the approprate cases. Therefore,
            // though these operations are implemented as RVSDG intrinsics, they get special
            // treatment here.
            OpCaseToBranchSelector(_) => self.visit_op_case_to_branch_selector(node),
            OpBoolToBranchSelector(_) => self.visit_op_bool_to_branch_selector(node),
            OpU32ToBranchSelector(_) => self.visit_op_u32_to_branch_selector(node),
            _ => {
                panic!("node kind not currently supported by SLIR's structured control-flow format")
            }
        }
    }

    fn visit_const_u32(&mut self, node: rvsdg::Node) {
        let value = self.rvsdg[node].expect_const_u32().value();
        let (_, binding) =
            self.scf
                .add_bind_const_u32(self.dst_block, BlockPosition::Append, value);

        self.value_mapping.map_output(node, 0, binding);
    }

    fn visit_const_i32(&mut self, node: rvsdg::Node) {
        let value = self.rvsdg[node].expect_const_i32().value();
        let (_, binding) =
            self.scf
                .add_bind_const_i32(self.dst_block, BlockPosition::Append, value);

        self.value_mapping.map_output(node, 0, binding);
    }

    fn visit_const_f32(&mut self, node: rvsdg::Node) {
        let value = self.rvsdg[node].expect_const_f32().value();
        let (_, binding) =
            self.scf
                .add_bind_const_f32(self.dst_block, BlockPosition::Append, value);

        self.value_mapping.map_output(node, 0, binding);
    }

    fn visit_const_bool(&mut self, node: rvsdg::Node) {
        let value = self.rvsdg[node].expect_const_bool().value();
        let (_, binding) =
            self.scf
                .add_bind_const_bool(self.dst_block, BlockPosition::Append, value);

        self.value_mapping.map_output(node, 0, binding);
    }

    fn visit_const_ptr(&mut self, node: rvsdg::Node) {
        let data = self.rvsdg[node].expect_const_ptr();
        let base = self.value_mapping.mapping(data.base().origin);

        let (_, binding) = match base {
            Value::Uniform(b) => self.scf.add_bind_uniform_ptr(
                self.dst_block,
                BlockPosition::Append,
                &self.module.uniform_bindings,
                b,
            ),
            Value::Storage(b) => self.scf.add_bind_storage_ptr(
                self.dst_block,
                BlockPosition::Append,
                &self.module.storage_bindings,
                b,
            ),
            Value::Workgroup(b) => self.scf.add_bind_workgroup_ptr(
                self.dst_block,
                BlockPosition::Append,
                &self.module.workgroup_bindings,
                b,
            ),
            Value::Constant(c) => self.scf.add_bind_constant_ptr(
                self.dst_block,
                BlockPosition::Append,
                &self.module.constants,
                c,
            ),
            Value::Local(_) => panic!("cannot create pointer to a local value"),
        };

        self.value_mapping.map_output(node, 0, binding);
    }

    fn visit_const_fallback(&mut self, node: rvsdg::Node) {
        let data = self.rvsdg[node].expect_const_fallback();
        let (_, binding) =
            self.scf
                .add_bind_fallback_value(self.dst_block, BlockPosition::Append, data.ty());

        self.value_mapping.map_output(node, 0, binding);
    }

    fn prepare_bind_intrinsic<T>(
        &self,
        node: rvsdg::Node,
        node_data: &rvsdg::IntrinsicNode<T>,
    ) -> PreparedBindIntrinsic<T>
    where
        T: Intrinsic + Clone,
        scf::ExpressionKind: From<scf::IntrinsicOp<T>>,
    {
        let arguments = node_data
            .value_inputs()
            .iter()
            .map(|i| self.value_mapping.mapping(i.origin).expect_local())
            .collect();

        PreparedBindIntrinsic {
            node,
            intrinsic: node_data.intrinsic().clone(),
            arguments,
        }
    }

    fn visit_op_alloca(&mut self, node: rvsdg::Node) {
        let data = self.rvsdg[node].expect_op_alloca();
        let (_, binding) = self
            .scf
            .add_alloca(self.dst_block, BlockPosition::Append, data.ty());

        self.value_mapping.map_output(node, 0, binding);
    }

    fn visit_op_store(&mut self, node: rvsdg::Node) {
        let data = self.rvsdg[node].expect_op_store();
        let ptr = self
            .value_mapping
            .mapping(data.ptr_input().origin)
            .expect_local();
        let value = self
            .value_mapping
            .mapping(data.value_input().origin)
            .expect_local();

        self.scf
            .add_store(self.dst_block, BlockPosition::Append, ptr, value);
    }

    fn visit_op_bool_to_branch_selector(&mut self, node: rvsdg::Node) {
        // We don't express bool-to-switch-predicate expression in the SCF, instead we translate
        // the switch nodes that use this predicate into "if" statements in the SCF; see
        // `visit_switch_node`. We therefore simply forward the output mapping to the input mapping.

        let data = self.rvsdg[node].expect_op_bool_to_branch_selector();
        let binding = self
            .value_mapping
            .mapping(data.value_input().origin)
            .expect_local();

        self.value_mapping.map_output(node, 0, binding);
    }

    fn visit_op_u32_to_branch_selector(&mut self, node: rvsdg::Node) {
        // We don't express u32-to-switch-predicate expression in the SCF, instead we translate any
        // switch node that uses this predicate into a switch statement in the SCF that uses the
        // appropriate cases; see `visit_switch_node`. We therefore simply forward the output
        // mapping to the input mapping.

        let data = self.rvsdg[node].expect_op_u32_to_branch_selector();
        let binding = self
            .value_mapping
            .mapping(data.value_input().origin)
            .expect_local();

        self.value_mapping.map_output(node, 0, binding);
    }

    fn visit_op_case_to_branch_selector(&mut self, node: rvsdg::Node) {
        // We don't express case-to-switch-predicate expression in the SCF, instead we translate any
        // switch node that uses this predicate into a switch statement in the SCF that uses the
        // appropriate cases; see `visit_switch_node`. We therefore simply forward the output
        // mapping to the input mapping.

        let data = self.rvsdg[node].expect_op_case_to_branch_selector();
        let binding = self
            .value_mapping
            .mapping(data.value_input().origin)
            .expect_local();

        self.value_mapping.map_output(node, 0, binding);
    }

    fn visit_sub_region(
        &mut self,
        region: rvsdg::Region,
        dst_block: scf::Block,
        argument_mapping: ValueMapping,
    ) -> ValueMapping {
        let mut sub_region_visitor = RegionVisitor {
            module: self.module,
            rvsdg: self.rvsdg,
            region_stratifier: &mut self.region_stratifier,
            scf: &mut self.scf,
            region,
            dst_block,
            value_mapping: argument_mapping,
            node_queue: Default::default(),
        };

        sub_region_visitor.visit_region();

        sub_region_visitor.value_mapping
    }
}

fn visit_region(
    module: &Module,
    rvsdg: &Rvsdg,
    region_stratifier: &mut RegionStratifier,
    scf: &mut Scf,
    region: rvsdg::Region,
    dst_block: scf::Block,
    argument_mapping: ValueMapping,
) -> ValueMapping {
    let mut region_visitor = RegionVisitor {
        module,
        rvsdg,
        region_stratifier,
        scf,
        region,
        dst_block,
        value_mapping: argument_mapping,
        node_queue: Default::default(),
    };

    region_visitor.visit_region();

    region_visitor.value_mapping
}

pub struct ScfBuilder<'a, 'b> {
    module: &'a Module,
    rvsdg: &'b Rvsdg,
    region_stratifier: RegionStratifier,
    scf: Scf,
}

impl<'a, 'b> ScfBuilder<'a, 'b> {
    pub fn new(module: &'a Module, rvsdg: &'b Rvsdg) -> Self {
        Self {
            module,
            rvsdg,
            region_stratifier: RegionStratifier::new(),
            scf: Scf::new(rvsdg.ty().clone()),
        }
    }

    pub fn build_function_body(&mut self, function: Function) {
        let body = self.scf.register_function(&self.module, function);
        let body_block = body.block();
        let arg_bindings = body.argument_bindings().to_vec();

        let fn_node = self
            .rvsdg
            .get_function_node(function)
            .expect("function not registered with RVSDG");
        let fn_data = self.rvsdg[fn_node].expect_function();
        let body_region = fn_data.body_region();

        let mut argument_mapping = ValueMapping::new();

        // Set up value mappings for the function arguments
        let arg_start = fn_data.dependencies().len();
        for (i, binding) in arg_bindings.into_iter().enumerate() {
            let arg = arg_start + i;

            argument_mapping.map_argument(arg as u32, binding.into());
        }

        // Set up value mappings for the global (uniform/storage/workgroup/constant) value
        // dependencies
        for (arg, dep) in fn_data.dependencies().iter().enumerate() {
            let ValueOrigin::Output {
                producer: dep_node,
                output: 0,
            } = dep.origin
            else {
                panic!("dependencies must connect to single-output nodes")
            };

            use crate::rvsdg::NodeKind::*;

            match self.rvsdg[dep_node].kind() {
                UniformBinding(n) => {
                    argument_mapping.map_argument(arg as u32, Value::Uniform(n.binding()));
                }
                StorageBinding(n) => {
                    argument_mapping.map_argument(arg as u32, Value::Storage(n.binding()));
                }
                WorkgroupBinding(n) => {
                    argument_mapping.map_argument(arg as u32, Value::Workgroup(n.binding()));
                }
                Constant(n) => {
                    argument_mapping.map_argument(arg as u32, Value::Constant(n.constant()));
                }
                _ => (),
            }
        }

        let value_mapping = visit_region(
            self.module,
            self.rvsdg,
            &mut self.region_stratifier,
            &mut self.scf,
            body_region,
            body_block,
            argument_mapping,
        );

        assert!(
            self.rvsdg[body_region].value_results().len() <= 1,
            "SLIR does not support multiple return values"
        );

        if let Some(result) = self.rvsdg[body_region].value_results().first() {
            let value = value_mapping.mapping(result.origin).expect_local();

            self.scf
                .add_return(body_block, BlockPosition::Append, Some(value));
        }
    }

    pub fn into_result(self) -> Scf {
        self.scf
    }
}

pub fn rvsdg_entry_points_to_scf(module: &Module, rvsdg: &Rvsdg) -> Scf {
    let mut builder = ScfBuilder::new(module, rvsdg);

    for (entry_point, _) in module.entry_points.iter() {
        builder.build_function_body(entry_point);
    }

    builder.into_result()
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::rvsdg::{ValueInput, ValueOutput};
    use crate::ty::{TY_DUMMY, TY_PREDICATE, TY_U32};
    use crate::{BinaryOperator, FnArg, FnSig, Symbol};

    #[test]
    fn test_single_region() {
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

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let node = rvsdg.add_op_binary(
            region,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 0),
            ValueInput::argument(TY_U32, 1),
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: node,
                output: 0,
            },
        );

        let mut builder = ScfBuilder::new(&module, &rvsdg);

        builder.build_function_body(function);

        let scf = builder.into_result();

        let body = scf
            .get_function_body(function)
            .expect("should have registered a function body");

        assert_eq!(body.argument_bindings().len(), 2);

        let arg_0 = body.argument_bindings()[0];
        let arg_1 = body.argument_bindings()[1];

        assert_eq!(scf[arg_0].ty(), TY_U32);
        assert_eq!(scf[arg_1].ty(), TY_U32);

        let block_data = &scf[body.block()];

        assert_eq!(block_data.statements().len(), 2);

        let statement_0 = block_data.statements()[0];
        let statement_1 = block_data.statements()[1];

        let statement_0_data = scf[statement_0].kind().expect_expr_binding();
        let statement_0_binding = statement_0_data.binding();
        let op_binary = statement_0_data.expression().kind().expect_op_binary();

        assert_eq!(op_binary.operator(), BinaryOperator::Add);
        assert_eq!(op_binary.lhs(), arg_0);
        assert_eq!(op_binary.rhs(), arg_1);

        let statement_1_data = scf[statement_1].kind().expect_return();

        let return_value = statement_1_data
            .value()
            .expect("should have a return value");

        assert_eq!(return_value, statement_0_binding);
    }

    #[test]
    fn test_switch_node() {
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
                        ty: TY_PREDICATE,
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

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::argument(TY_U32, 1),
            ],
            vec![ValueOutput::new(TY_U32)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);

        let branch_0_const_node = rvsdg.add_const_u32(branch_0, 1);
        let branch_0_add_node = rvsdg.add_op_binary(
            branch_0,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 0),
            ValueInput::output(TY_U32, branch_0_const_node, 0),
        );

        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: branch_0_add_node,
                output: 0,
            },
        );

        let branch_1 = rvsdg.add_switch_branch(switch_node);

        let branch_1_const_node = rvsdg.add_const_u32(branch_1, 0);

        rvsdg.reconnect_region_result(
            branch_1,
            0,
            ValueOrigin::Output {
                producer: branch_1_const_node,
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

        let mut builder = ScfBuilder::new(&module, &rvsdg);

        builder.build_function_body(function);

        let scf = builder.into_result();

        let body = scf
            .get_function_body(function)
            .expect("should have registered a function body");

        assert_eq!(body.argument_bindings().len(), 2);

        let arg_0 = body.argument_bindings()[0];
        let arg_1 = body.argument_bindings()[1];

        assert_eq!(scf[arg_0].ty(), TY_PREDICATE);
        assert_eq!(scf[arg_1].ty(), TY_U32);

        let block_data = &scf[body.block()];

        assert_eq!(block_data.statements().len(), 2);

        let stmt_switch = block_data.statements()[0];
        let stmt_return = block_data.statements()[1];

        let stmt_switch_data = scf[stmt_switch].kind().expect_switch();

        assert_eq!(stmt_switch_data.out_vars().len(), 1);

        let stmt_switch_out_var = stmt_switch_data.out_vars()[0];

        assert_eq!(stmt_switch_data.cases().len(), 1);

        let case_0_block = stmt_switch_data.cases()[0].block();
        let case_0_block_data = &scf[case_0_block];

        assert_eq!(case_0_block_data.statements().len(), 2);

        let case_0_stmt_const = case_0_block_data.statements()[0];
        let case_0_stmt_add = case_0_block_data.statements()[1];

        let case_0_stmt_const_data = scf[case_0_stmt_const].kind().expect_expr_binding();
        let case_0_stmt_const_binding = case_0_stmt_const_data.binding();
        let case_0_stmt_const_expr = case_0_stmt_const_data.expression();

        assert_eq!(case_0_stmt_const_expr.kind().expect_const_u32(), 1);

        let case_0_stmt_add_data = scf[case_0_stmt_add].kind().expect_expr_binding();
        let case_0_stmt_add_binding = case_0_stmt_add_data.binding();
        let case_0_stmt_add_expr = case_0_stmt_add_data.expression();

        let case_0_add_data = case_0_stmt_add_expr.kind().expect_op_binary();

        assert_eq!(case_0_add_data.operator(), BinaryOperator::Add);
        assert_eq!(case_0_add_data.lhs(), arg_1);
        assert_eq!(case_0_add_data.rhs(), case_0_stmt_const_binding);

        let case_0_var_0_binding = case_0_block_data.control_flow_var(stmt_switch_out_var);

        assert_eq!(case_0_var_0_binding, case_0_stmt_add_binding);

        let default_block_data = &scf[stmt_switch_data.default()];

        assert_eq!(default_block_data.statements().len(), 1);
        let default_stmt_const = default_block_data.statements()[0];

        let default_stmt_const_data = scf[default_stmt_const].kind().expect_expr_binding();
        let default_stmt_const_binding = default_stmt_const_data.binding();
        let default_stmt_const_expr = default_stmt_const_data.expression();

        assert_eq!(default_stmt_const_expr.kind().expect_const_u32(), 0);

        let default_var_0_binding = default_block_data.control_flow_var(stmt_switch_out_var);

        assert_eq!(default_var_0_binding, default_stmt_const_binding);

        let stmt_return_data = scf[stmt_return].kind().expect_return();

        let return_value = stmt_return_data
            .value()
            .expect("should have a return value");

        assert_eq!(return_value, stmt_switch_out_var);
    }

    #[test]
    fn test_loop_node() {
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

        let (loop_node, loop_region) =
            rvsdg.add_loop(region, vec![ValueInput::argument(TY_U32, 0)], None);

        let added_value_node = rvsdg.add_const_u32(loop_region, 1);
        let add_node = rvsdg.add_op_binary(
            loop_region,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 0),
            ValueInput::output(TY_U32, added_value_node, 0),
        );
        let compare_value_node = rvsdg.add_const_u32(loop_region, 10);
        let compare_node = rvsdg.add_op_binary(
            loop_region,
            BinaryOperator::Lt,
            ValueInput::output(TY_U32, add_node, 0),
            ValueInput::output(TY_U32, compare_value_node, 0),
        );

        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: compare_node,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            loop_region,
            1,
            ValueOrigin::Output {
                producer: add_node,
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

        let mut builder = ScfBuilder::new(&module, &rvsdg);

        builder.build_function_body(function);

        let scf = builder.into_result();

        let body = scf
            .get_function_body(function)
            .expect("should have registered a function body");

        assert_eq!(body.argument_bindings().len(), 1);

        let arg = body.argument_bindings()[0];

        assert_eq!(scf[arg].ty(), TY_U32);

        let block_data = &scf[body.block()];

        assert_eq!(block_data.statements().len(), 2);

        let stmt_loop = block_data.statements()[0];
        let stmt_return = block_data.statements()[1];

        let stmt_loop_data = scf[stmt_loop].kind().expect_loop();

        assert_eq!(stmt_loop_data.loop_vars().len(), 1);

        let loop_var = stmt_loop_data.loop_vars()[0];

        assert_eq!(loop_var.initial_value(), body.argument_bindings()[0]);

        let loop_block = stmt_loop_data.block();
        let loop_block_data = &scf[loop_block];

        assert_eq!(loop_block_data.statements().len(), 4);

        let loop_block_stmt_0 = loop_block_data.statements()[0];
        let loop_block_stmt_1 = loop_block_data.statements()[1];
        let loop_block_stmt_2 = loop_block_data.statements()[2];
        let loop_block_stmt_3 = loop_block_data.statements()[3];

        let added_value_data = scf[loop_block_stmt_0].kind().expect_expr_binding();
        let added_value_binding = added_value_data.binding();
        let added_value_expr = added_value_data.expression();

        assert_eq!(added_value_expr.kind().expect_const_u32(), 1);

        let compare_value_data = scf[loop_block_stmt_1].kind().expect_expr_binding();
        let compare_value_binding = compare_value_data.binding();
        let compare_value_expr = compare_value_data.expression();

        assert_eq!(compare_value_expr.kind().expect_const_u32(), 10);

        let add_data = scf[loop_block_stmt_2].kind().expect_expr_binding();
        let add_binding = add_data.binding();
        let add_expr_data = add_data.expression().kind().expect_op_binary();

        assert_eq!(add_expr_data.operator(), BinaryOperator::Add);
        assert_eq!(add_expr_data.lhs(), loop_var.binding());
        assert_eq!(add_expr_data.rhs(), added_value_binding);

        let compare_data = scf[loop_block_stmt_3].kind().expect_expr_binding();
        let compare_binding = compare_data.binding();
        let compare_expr_data = compare_data.expression().kind().expect_op_binary();

        assert_eq!(compare_expr_data.operator(), BinaryOperator::Lt);
        assert_eq!(compare_expr_data.lhs(), add_binding);
        assert_eq!(compare_expr_data.rhs(), compare_value_binding);

        let loop_var_binding = loop_block_data.control_flow_var(loop_var.binding());

        assert_eq!(loop_var_binding, add_binding);

        let LoopControl::Tail(reentry_control_binding) = stmt_loop_data.control() else {
            panic!("should be tail-controlled loop")
        };

        assert_eq!(reentry_control_binding, compare_binding);

        let stmt_return_data = scf[stmt_return].kind().expect_return();

        let return_value = stmt_return_data
            .value()
            .expect("should have a return value");

        assert_eq!(return_value, loop_var.binding());
    }
}
