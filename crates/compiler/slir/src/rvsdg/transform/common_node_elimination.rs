use bumpalo::Bump;
use rustc_hash::FxHashMap;

use crate::rvsdg::visit::region_nodes::RegionNodesVisitor;
use crate::rvsdg::{
    Connectivity, Node, NodeKind, Region, Rvsdg, SimpleNode, ValueInput, ValueOrigin, visit,
};
use crate::ty::{Matrix, Type, Vector};
use crate::{BinaryOperator, UnaryOperator};

#[derive(Clone, PartialEq, Eq, Hash)]
enum SimpleNodeMetadata {
    ConstU32(u32),
    ConstI32(i32),
    ConstF32(u32), // Use u32 bits for Hash/Eq
    ConstBool(bool),
    ConstPredicate(u32),
    ConstPtr(Type),
    ConstFallback(Type),
    OpExtractField(u32),
    OpExtractElement,
    OpFieldPtr(u32),
    OpElementPtr,
    OpDiscriminantPtr,
    OpVariantPtr(u32),
    OpGetDiscriminant,
    OpSetDiscriminant(u32),
    OpOffsetSlice,
    OpGetSliceOffset,
    OpUnary(UnaryOperator),
    OpBinary(BinaryOperator),
    OpVector(Vector),
    OpMatrix(Matrix),
    OpCaseToBranchSelector(Vec<u32>),
    OpBoolToBranchSelector,
    OpU32ToBranchSelector,
    OpBranchSelectorToCase(Vec<u32>),
    OpConvertToU32,
    OpConvertToI32,
    OpConvertToF32,
    OpConvertToBool,
    OpArrayLength,
    ValueProxy,
    Reaggregation,
}

impl SimpleNodeMetadata {
    fn from_simple_node(node: &SimpleNode) -> Option<Self> {
        let metadata = match node {
            SimpleNode::ConstU32(n) => SimpleNodeMetadata::ConstU32(n.value()),
            SimpleNode::ConstI32(n) => SimpleNodeMetadata::ConstI32(n.value()),
            SimpleNode::ConstF32(n) => SimpleNodeMetadata::ConstF32(n.value().to_bits()),
            SimpleNode::ConstBool(n) => SimpleNodeMetadata::ConstBool(n.value()),
            SimpleNode::ConstPredicate(n) => SimpleNodeMetadata::ConstPredicate(n.value()),
            SimpleNode::ConstPtr(n) => SimpleNodeMetadata::ConstPtr(n.pointee_ty()),
            SimpleNode::ConstFallback(n) => SimpleNodeMetadata::ConstFallback(n.ty()),
            SimpleNode::OpExtractField(n) => SimpleNodeMetadata::OpExtractField(n.field_index()),
            SimpleNode::OpExtractElement(_) => SimpleNodeMetadata::OpExtractElement,
            SimpleNode::OpFieldPtr(n) => SimpleNodeMetadata::OpFieldPtr(n.field_index()),
            SimpleNode::OpElementPtr(_) => SimpleNodeMetadata::OpElementPtr,
            SimpleNode::OpDiscriminantPtr(_) => SimpleNodeMetadata::OpDiscriminantPtr,
            SimpleNode::OpVariantPtr(n) => SimpleNodeMetadata::OpVariantPtr(n.variant_index()),
            SimpleNode::OpGetDiscriminant(_) => SimpleNodeMetadata::OpGetDiscriminant,
            SimpleNode::OpSetDiscriminant(n) => {
                SimpleNodeMetadata::OpSetDiscriminant(n.variant_index())
            }
            SimpleNode::OpOffsetSlice(_) => SimpleNodeMetadata::OpOffsetSlice,
            SimpleNode::OpGetSliceOffset(_) => SimpleNodeMetadata::OpGetSliceOffset,
            SimpleNode::OpUnary(n) => SimpleNodeMetadata::OpUnary(n.intrinsic().operator),
            SimpleNode::OpBinary(n) => SimpleNodeMetadata::OpBinary(n.intrinsic().operator),
            SimpleNode::OpVector(n) => SimpleNodeMetadata::OpVector(n.intrinsic().ty),
            SimpleNode::OpMatrix(n) => SimpleNodeMetadata::OpMatrix(n.intrinsic().ty),
            SimpleNode::OpCaseToBranchSelector(n) => {
                SimpleNodeMetadata::OpCaseToBranchSelector(n.cases().to_vec())
            }
            SimpleNode::OpBoolToBranchSelector(_) => SimpleNodeMetadata::OpBoolToBranchSelector,
            SimpleNode::OpU32ToBranchSelector(_) => SimpleNodeMetadata::OpU32ToBranchSelector,
            SimpleNode::OpBranchSelectorToCase(n) => {
                SimpleNodeMetadata::OpBranchSelectorToCase(n.cases().to_vec())
            }
            SimpleNode::OpConvertToU32(_) => SimpleNodeMetadata::OpConvertToU32,
            SimpleNode::OpConvertToI32(_) => SimpleNodeMetadata::OpConvertToI32,
            SimpleNode::OpConvertToF32(_) => SimpleNodeMetadata::OpConvertToF32,
            SimpleNode::OpConvertToBool(_) => SimpleNodeMetadata::OpConvertToBool,
            SimpleNode::OpArrayLength(_) => SimpleNodeMetadata::OpArrayLength,
            SimpleNode::ValueProxy(_) => SimpleNodeMetadata::ValueProxy,
            SimpleNode::Reaggregation(_) => SimpleNodeMetadata::Reaggregation,

            // We don't ever consider alloca nodes or nodes that are part of the state chain to be
            // duplicates.
            SimpleNode::OpAlloca(_)
            | SimpleNode::OpCall(_)
            | SimpleNode::OpLoad(_)
            | SimpleNode::OpStore(_) => return None,
        };

        Some(metadata)
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct SimpleNodeDescription<'a> {
    region: Region,
    metadata: SimpleNodeMetadata,
    inputs: &'a [ValueInput],
}

struct ReconnectionJob {
    region: Region,
    original_origin: ValueOrigin,
    new_origin: ValueOrigin,
}

struct CommonNodeEliminatorInternal<'a> {
    arena: &'a Bump,
    primary_nodes: FxHashMap<SimpleNodeDescription<'a>, Node>,
    job_queue: &'a mut Vec<ReconnectionJob>,
    did_eliminate_nodes: bool,
}

impl<'a> RegionNodesVisitor for CommonNodeEliminatorInternal<'a> {
    fn visit_node(&mut self, rvsdg: &Rvsdg, node: Node) {
        let node_data = &rvsdg[node];
        let region = node_data.region();

        if let NodeKind::Simple(simple_node) = node_data.kind()
            && let Some(metadata) = SimpleNodeMetadata::from_simple_node(simple_node)
        {
            let inputs = self.arena.alloc_slice_copy(simple_node.value_inputs());

            let description = SimpleNodeDescription {
                region,
                metadata,
                inputs,
            };

            if let Some(&primary_node) = self.primary_nodes.get(&description) {
                for output in 0..node_data.value_outputs().len() as u32 {
                    let original_origin = ValueOrigin::Output {
                        producer: node,
                        output,
                    };
                    let new_origin = ValueOrigin::Output {
                        producer: primary_node,
                        output,
                    };

                    self.job_queue.push(ReconnectionJob {
                        region,
                        original_origin,
                        new_origin,
                    });

                    self.did_eliminate_nodes = true;
                }
            } else {
                self.primary_nodes.insert(description, node);
            }
        }

        visit::region_nodes::visit_node(self, rvsdg, node);
    }
}

pub struct CommonNodeEliminator {
    arena: Bump,
    job_queue: Vec<ReconnectionJob>,
}

impl CommonNodeEliminator {
    pub fn new() -> Self {
        Self {
            arena: Bump::new(),
            job_queue: vec![],
        }
    }

    pub fn process_region(&mut self, rvsdg: &mut Rvsdg, region: Region) -> bool {
        self.arena.reset();
        self.job_queue.clear();

        let mut internal = CommonNodeEliminatorInternal {
            arena: &self.arena,
            primary_nodes: Default::default(),
            job_queue: &mut self.job_queue,
            did_eliminate_nodes: false,
        };

        internal.visit_region(rvsdg, region);

        let did_eliminate_nodes = internal.did_eliminate_nodes;

        for job in self.job_queue.drain(..) {
            rvsdg.reconnect_value_users(job.region, job.original_origin, job.new_origin);
        }

        did_eliminate_nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rvsdg::{StateOrigin, ValueInput, ValueOrigin};
    use crate::ty::{TY_DUMMY, TY_U32};
    use crate::{BinaryOperator, FnArg, FnSig, Function, Module, Symbol};

    #[test]
    fn test_common_node_elimination_basic() {
        let mut module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let function = Function {
            name: Symbol::from_ref("f"),
            module: Symbol::from_ref("m"),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
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
                ret_ty: None,
            },
        );

        let (_, region) = rvsdg.register_function(&module, function, Vec::new());

        let arg0 = ValueInput::argument(TY_U32, 0);
        let arg1 = ValueInput::argument(TY_U32, 1);

        // Add two identical binary nodes.
        let node0 = rvsdg.add_op_binary(region, BinaryOperator::Add, arg0.clone(), arg1.clone());
        let node1 = rvsdg.add_op_binary(region, BinaryOperator::Add, arg0.clone(), arg1.clone());

        // Use node1's output.
        let user_input = ValueInput::output(TY_U32, node1, 0);
        let user_node = rvsdg.add_value_proxy(region, user_input);

        let mut eliminator = CommonNodeEliminator::new();
        assert!(eliminator.process_region(&mut rvsdg, region));

        // Check if user_node now uses node0.
        assert_eq!(
            rvsdg[user_node].value_inputs()[0].origin,
            ValueOrigin::Output {
                producer: node0,
                output: 0
            }
        );
    }

    #[test]
    fn test_common_node_elimination_different_inputs() {
        let mut module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let function = Function {
            name: Symbol::from_ref("f"),
            module: Symbol::from_ref("m"),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
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
                    FnArg {
                        ty: TY_U32,
                        shader_io_binding: None,
                    },
                ],
                ret_ty: None,
            },
        );

        let (_, region) = rvsdg.register_function(&module, function, Vec::new());

        let arg0 = ValueInput::argument(TY_U32, 0);
        let arg1 = ValueInput::argument(TY_U32, 1);
        let arg2 = ValueInput::argument(TY_U32, 2);

        // Add two binary nodes with different second inputs.
        let _node0 = rvsdg.add_op_binary(region, BinaryOperator::Add, arg0.clone(), arg1.clone());
        let node1 = rvsdg.add_op_binary(region, BinaryOperator::Add, arg0.clone(), arg2.clone());

        // Use node1's output.
        let user_input = ValueInput::output(TY_U32, node1, 0);
        let user_node = rvsdg.add_value_proxy(region, user_input);

        let mut eliminator = CommonNodeEliminator::new();
        // Should NOT eliminate because they have different inputs.
        assert!(!eliminator.process_region(&mut rvsdg, region));

        // Check if user_node still uses node1.
        assert_eq!(
            rvsdg[user_node].value_inputs()[0].origin,
            ValueOrigin::Output {
                producer: node1,
                output: 0
            }
        );
    }

    #[test]
    fn test_common_node_elimination_three_identical() {
        let mut module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let function = Function {
            name: Symbol::from_ref("f"),
            module: Symbol::from_ref("m"),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
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
                ret_ty: None,
            },
        );

        let (_, region) = rvsdg.register_function(&module, function, Vec::new());

        let arg0 = ValueInput::argument(TY_U32, 0);
        let arg1 = ValueInput::argument(TY_U32, 1);

        // Add three identical binary nodes.
        let node0 = rvsdg.add_op_binary(region, BinaryOperator::Add, arg0.clone(), arg1.clone());
        let node1 = rvsdg.add_op_binary(region, BinaryOperator::Add, arg0.clone(), arg1.clone());
        let node2 = rvsdg.add_op_binary(region, BinaryOperator::Add, arg0.clone(), arg1.clone());

        // Use node1's output.
        let user_input1 = ValueInput::output(TY_U32, node1, 0);
        let user_node1 = rvsdg.add_value_proxy(region, user_input1);

        // Use node2's output.
        let user_input2 = ValueInput::output(TY_U32, node2, 0);
        let user_node2 = rvsdg.add_value_proxy(region, user_input2);

        let mut eliminator = CommonNodeEliminator::new();
        assert!(eliminator.process_region(&mut rvsdg, region));

        // Check if user_node1 now uses node0.
        assert_eq!(
            rvsdg[user_node1].value_inputs()[0].origin,
            ValueOrigin::Output {
                producer: node0,
                output: 0
            }
        );

        // Check if user_node2 now uses node0.
        assert_eq!(
            rvsdg[user_node2].value_inputs()[0].origin,
            ValueOrigin::Output {
                producer: node0,
                output: 0
            }
        );
    }

    #[test]
    fn test_common_node_elimination_constants() {
        let mut module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let function = Function {
            name: Symbol::from_ref("f"),
            module: Symbol::from_ref("m"),
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

        let (_, region) = rvsdg.register_function(&module, function, Vec::new());

        // Add two identical constant nodes.
        let node0 = rvsdg.add_const_u32(region, 42);
        let node1 = rvsdg.add_const_u32(region, 42);

        // Use node1's output.
        let user_input = ValueInput::output(TY_U32, node1, 0);
        let user_node = rvsdg.add_value_proxy(region, user_input);

        let mut eliminator = CommonNodeEliminator::new();
        assert!(eliminator.process_region(&mut rvsdg, region));

        // Check if user_node now uses node0.
        assert_eq!(
            rvsdg[user_node].value_inputs()[0].origin,
            ValueOrigin::Output {
                producer: node0,
                output: 0
            }
        );
    }

    #[test]
    fn test_common_node_elimination_state_exclusion() {
        let mut module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let function = Function {
            name: Symbol::from_ref("f"),
            module: Symbol::from_ref("m"),
        };

        let ptr_ty = rvsdg.ty().register(crate::ty::TypeKind::Ptr(TY_U32));

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
                ty: TY_DUMMY,
                args: vec![FnArg {
                    ty: ptr_ty,
                    shader_io_binding: None,
                }],
                ret_ty: None,
            },
        );

        let (_, region) = rvsdg.register_function(&module, function, Vec::new());

        let ptr_input = ValueInput::argument(ptr_ty, 0);
        let state_origin = StateOrigin::Argument;

        // Add two identical Load nodes (they have state connections).
        let _node0 = rvsdg.add_op_load(region, ptr_input.clone(), state_origin.clone());
        let _node1 = rvsdg.add_op_load(region, ptr_input.clone(), state_origin.clone());

        let mut eliminator = CommonNodeEliminator::new();
        // Should NOT eliminate because they are stateful.
        assert!(!eliminator.process_region(&mut rvsdg, region));
    }
}
