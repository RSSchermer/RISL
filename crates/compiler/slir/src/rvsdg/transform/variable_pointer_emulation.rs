//! Transforms (atomic) load and store nodes that read from variable pointers into "emulation
//! programs".
//!
//! WGSL does not support "variable pointers". For example, the following WGSL code would not
//! compile:
//!
//! ```wgsl
//! var data: array<u32, 2> = array(10, 20);
//! var p: ptr<function, u32> = &data[0];
//!
//! if condition {
//!     p = &data[1];
//! }
//!
//! *p = 0;
//! ```
//!
//! However, one might rewrite this pattern as follows to avoid the variable pointer:
//!
//! ```wgsl
//! var array: array<u32, 2> = array(10, 20);
//! var index: u32 = 0;
//!
//! if condition {
//!     index = 1;
//! }
//!
//! array[index] = 0;
//! ```
//!
//! While the rewrite in this example is fairly trivial, this transform extends this idea to any
//! arbitrary use of a variable pointer. To achieve this, we replace any user of a variable pointer
//! (e.g., a "load" operation) with a "pointer-emulation program".
//!
//! In the context of an RVSDG, there are three origins that may produce a variable pointer:
//!
//! 1. A value-output of a [Switch](crate::rvsdg::Switch) which is of a pointer type.
//! 2. A value-output of a [Loop](crate::rvsdg::Loop) node which is of a pointer type, if the
//!    associated loop-value is not a loop-constant (the loop-result connects directly to the
//!    corresponding loop-argument).
//! 3. A region argument of a [Loop](crate::rvsdg::Loop) node which is of a pointer type, if the
//!    associated loop-value is not a loop-constant (the corresponding loop-result connects directly
//!    to this loop-argument).
//!
//! In addition, any further refinements of variable pointers (e.g., with an
//! [OpElementPtr](crate::rvsdg::OpElementPtr) node) are also considered variable pointers. If a
//! variable pointer is passed into a nested region, then the region argument that represents the
//! pointer-value inside the region is also considered a variable pointer.
//!
//! Any downstream load-like or store-like operations that use a variable pointer will need to be
//! replaced with a pointer-emulation program. The variable-pointer emulation transform is part of
//! the larger [memory_promotion_and_legalization][0] transform; the identification of such
//! load-like or store-like nodes is done by as part of [memory_promotion_and_legalization][0]
//! transform.
//!
//! In order for this transform to be able to build a pointer-emulation program, it has to be able
//! to trace each variable pointer to its "root" pointers, that is, the output of an
//! [OpAlloca](crate::rvsdg::OpAlloca) node or the output of a [ConstPtr](crate::rvsdg::ConstPtr)
//! node. We cannot, in general, trace variable pointers through loads and stores from/to global
//! memory (uniform/storage/workgroup bindings): since other threads could also interact with a
//! pointer-value stored in global memory, we cannot statically determine the root of a
//! pointer-value stored in global memory. However, since our targets already disallow us from
//! storing pointer-type values in global memory, we only have to concern ourselves with pointers
//! stored into allocations in the function address-space ("on the stack"), as represented by
//! [OpAlloca] nodes in our [Rvsdg] representation; since only a single thread can load and store
//! into such allocations, a static analysis can determine the total ordering of all loads and
//! stores. However, we do not perform such an analysis here; we instead rely on the
//! [memory_promotion_and_legalization][0] pass to replace all loads and stores from/into [OpAlloca]
//! allocations with plain value-flow (a.k.a. "memory-to-value-flow promotion"). The
//! [memory_promotion_and_legalization][0] pass will ensure that all loads and stores in the reverse
//! value-flow graph of a variable pointer will have been promoted to plain value-flow before
//! invoking variable-pointer emulation for any of its users. As such, this transform never has to
//! concern itself with tracing such pointers through loads and stores; simple reverse value-flow
//! traces will always succeed at finding the root pointers.
//!
//! # Pointer Emulation Programs
//!
//! Before we replace the load or store operation with a pointer emulation program, we first
//! construct [PointerEmulationInfo] for value-origin of the operation's pointer input.
//! [PointerEmulationInfo] represents a pointer emulation program without being specific to the
//! particular operation we'll be emulating. Constructing the [PointerEmulationInfo] involves
//! tracing back the pointer-value through reverse value-flow, recursively constructing
//! [PointerEmulationInfo] for all values we encounter, until we reach the root-pointer origins.
//! It effectively represents a description of a pointer value's complete provenance. Since the
//! [PointerEmulationInfo] remains constant throughout the [memory_promotion_and_legalization][0]
//! pass, and since a pointer-value is regularly used by more than one operation that needs
//! emulating, we cache the [PointerEmulationInfo] for each value-origin in a hashmap.
//!
//! We store the [PointerEmulationInfo] as a tree. The [LeafNode]s of this tree each store a
//! [ValueOrigin] for a root pointer, along with a complete access chain that will construct the
//! pointer the emulated operation will operate on. When the access chain involves dynamic indexing,
//! then it stores the [ValueOrigin] for the dynamic index value(s).
//!
//! [BranchingNode]s in this tree represent variability over different reverse value-flow paths
//! through different branches of [Switch] nodes. They store the [ValueOrigin] for the
//! branch-selector of the [Switch] node that caused the path to split.
//!
//! All root pointer values, dynamic index values, and branch-selector values we collectively refer
//! to as the "emulation values". Note that all emulation values need to be available in the region
//! of the pointer-value we intend to emulate, as these values will be the inputs to our pointer
//! emulation program, and connecting values across region boundaries is not allowed in an RVSDG.
//! Therefore, while collecting the [PointerEmulationInfo] for a given pointer-value, we also make
//! sure that all its emulation values are made available in this pointer-value's region.
//! Specifically, whenever the pointer-value is the output of a [Switch] or [Loop] node, we add
//! additional value-outputs for all emulation values (that are not already available as a
//! value-output of the [Switch] or [Loop] node); whenever the pointer-value is a region argument
//! for a [Switch] branch or [Loop] region, we add additional value-inputs for all emulation values
//! (that are not already available as a value-input of the [Switch] branch or [Loop] region).
//!
//! When we're constructing the pointer emulation program from the [PointerEmulationInfo], the
//! [BranchNode]s become [Switch] nodes; each [LeafNode] becomes a chain of [OpElementPtr] and
//! [OpFieldPtr] nodes that reconstruct the access chain, followed by a copy of the operation we're
//! emulating. This copy of the operation now uses the [LeafNode]'s pointer construction instead of
//! the original variable pointer. Some operations also take additional inputs (e.g., an [OpStore]
//! node also takes the value to be stored as an input); we make sure that the [Switch] nodes we
//! construct to represent [BranchingNode]s make all emulation values and all such additional inputs
//! available to each of their branches.
//!
//! Note that all emulation values need to "live" until the end of the emulation program. This
//! represents the main cost of variable pointer emulation. With a compilation target that supports
//! true physical pointers, all the information that describes the pointer can be condensed into
//! a single physical address value, no matter the complexity of the access and control-flow that
//! produced the pointer. With variable pointer emulation, emulation values may need to be kept
//! alive for longer than they otherwise would. Variable pointer emulation may therefore result in
//! increased register use relative to true physical pointers. Certain later optimizations, such as
//! switch-merging, may be able to reduce this cost a little.
//!
//! [0]: super::memory_promotion_and_legalization

use indexmap::IndexSet;
use smallvec::SmallVec;

use crate::rvsdg::transform::loop_pointer_normalization::VariableLoopPointerNormalizer;
use crate::rvsdg::transform::pointer_reconstruction::{
    Access, BranchingNode, LeafNode, PointerReconstructionContext, PointerReconstructionError,
    PointerReconstructionInfo, PointerReconstructionNode,
};
use crate::rvsdg::{
    Connectivity, Node, Region, Rvsdg, StateOrigin, ValueInput, ValueOrigin, ValueOutput,
};
use crate::ty::{TY_PREDICATE, TY_U32, Type};

pub struct EmulationContext {
    reconstruction_context: PointerReconstructionContext,
    loop_pointer_normalizer: VariableLoopPointerNormalizer,
}

impl EmulationContext {
    pub fn new() -> Self {
        EmulationContext {
            reconstruction_context: PointerReconstructionContext::new(),
            loop_pointer_normalizer: VariableLoopPointerNormalizer::new(),
        }
    }

    /// Replaces an [OpLoad] node where the input pointer is a variable pointer, with an emulation
    /// sub-graph that does not use variable pointers.
    pub fn emulate_op_load(&mut self, rvsdg: &mut Rvsdg, op_load: Node) {
        let region = rvsdg[op_load].region();
        let data = rvsdg[op_load].expect_op_load();
        let output_ty = data.value_output().ty;
        let state_origin = data.state().unwrap().origin;
        let ptr_origin = data.ptr_input().origin;
        let info = self.resolve_reconstruction_info(rvsdg, region, ptr_origin);

        let gen_op_load = |rvsdg: &mut Rvsdg,
                           region: Region,
                           ptr_input: ValueInput,
                           _additional_inputs: &[ValueInput],
                           state_origin: StateOrigin| {
            rvsdg.add_op_load(region, ptr_input, state_origin)
        };

        let mut emulator = Emulator {
            rvsdg,
            outer_region: region,
            state_origin,
            output_ty: Some(output_ty),
            op_gen: gen_op_load,
            additional_values: &[],
        };

        let emulated = emulator.emulate(info.reconstruction_root);

        // Now that we've constructed a node to emulate the original OpLoad, reconnect all users of
        // its value output to the emulated node's output and remove the original OpLoad.

        let user_count = rvsdg[op_load].expect_op_load().value_output().users.len();

        for i in (0..user_count).rev() {
            let user = rvsdg[op_load].expect_op_load().value_output().users[i];

            rvsdg.reconnect_value_user(
                region,
                user,
                ValueOrigin::Output {
                    producer: emulated,
                    output: 0,
                },
            );
        }

        rvsdg.remove_node(op_load);
    }

    /// Replaces an [OpStore] node where the input pointer is a variable pointer, with an emulation
    /// sub-graph that does not use variable pointers.
    pub fn emulate_op_store(&mut self, rvsdg: &mut Rvsdg, op_store: Node) {
        let outer_region = rvsdg[op_store].region();
        let data = rvsdg[op_store].expect_op_store();
        let state_origin = data.state().unwrap().origin;
        let ptr_origin = data.ptr_input().origin;
        let value_input = *data.value_input();
        let info = self.resolve_reconstruction_info(rvsdg, outer_region, ptr_origin);

        let gen_op_store = |rvsdg: &mut Rvsdg,
                            region: Region,
                            ptr_input: ValueInput,
                            additional_inputs: &[ValueInput],
                            state_origin: StateOrigin| {
            rvsdg.add_op_store(region, ptr_input, additional_inputs[0], state_origin)
        };

        let mut emulator = Emulator {
            rvsdg,
            outer_region,
            state_origin,
            output_ty: None,
            op_gen: gen_op_store,
            additional_values: &[value_input],
        };

        emulator.emulate(info.reconstruction_root);

        rvsdg.remove_node(op_store);
    }

    pub fn clear(&mut self) {
        self.reconstruction_context.clear();
    }

    fn resolve_reconstruction_info(
        &mut self,
        rvsdg: &mut Rvsdg,
        region: Region,
        origin: ValueOrigin,
    ) -> PointerReconstructionInfo {
        use PointerReconstructionError::*;

        loop {
            match self
                .reconstruction_context
                .resolve_reconstruction_info(rvsdg, region, origin)
            {
                Err(NeedsLoopPointerNormalization {
                    loop_node,
                    loop_value,
                }) => {
                    self.loop_pointer_normalizer
                        .normalize_loop_value(rvsdg, loop_node, loop_value);
                }
                Err(err) => panic!("unexpected error: {:?}", err),
                Ok(info) => return info.clone(),
            }
        }
    }
}

struct Emulator<'a, F> {
    rvsdg: &'a mut Rvsdg,

    /// The region that hosts the "load" or "store" node that we're emulating.
    outer_region: Region,

    /// The state origin of the "load" or "store" node that we're emulating.
    state_origin: StateOrigin,

    /// If we're emulating a "load" node, the output type of the load operation.
    output_ty: Option<Type>,

    /// A function that will generate a new "load" or "store" node operating on an emulated pointer.
    ///
    /// The first argument passed to this function is a `mut` reference to the [Rvsdg].
    ///
    /// The second argument is the region into which the generated node must be added.
    ///
    /// The third argument is a [ValueInput] that is to be used as the "pointer" input for the
    /// operation.
    ///
    /// The fourth argument is a slice of additional input values that are to be used as the
    /// remaining value inputs for the generated node. For example, if we're emulating an [OpStore],
    /// this slice will have a length of one and represents the value to be stored.
    ///
    /// The fifth argument is the [StateOrigin] that the generated node must use.
    op_gen: F,

    /// A slice containing the original additional value inputs for the "load" or "store" node we're
    /// emulating.
    ///
    /// For example, if we're emulating an [OpLoad], then this slice should be empty. If we're
    /// emulating an [OpStore], then this slice should have a length of one and represents the value
    /// to be stored.
    additional_values: &'a [ValueInput],
}

impl<'a, F> Emulator<'a, F>
where
    F: Fn(&mut Rvsdg, Region, ValueInput, &[ValueInput], StateOrigin) -> Node,
{
    fn emulate(&mut self, root_node: PointerReconstructionNode) -> Node {
        match root_node {
            PointerReconstructionNode::Branching(mut node) => {
                node.propagate_sub_tree_inputs(|leaf_node, input_set| {
                    input_set.insert(leaf_node.root_pointer.origin);

                    for access in &leaf_node.access_chain {
                        if let &Access::DynamicElement(index_origin) = access {
                            input_set.insert(index_origin);
                        }
                    }
                });

                self.visit_branching_node(self.outer_region, &node, None)
            }
            PointerReconstructionNode::Leaf(node) => {
                self.visit_leaf_node(self.outer_region, &node, None)
            }
        }
    }

    fn visit_branching_node(
        &mut self,
        region: Region,
        branching_node: &BranchingNode,
        input_mapping: Option<&IndexSet<ValueOrigin>>,
    ) -> Node {
        let resolve_input = |origin, ty| {
            if let Some(input_mapping) = input_mapping {
                let argument = input_mapping
                    .get_index_of(&origin)
                    .expect("input mapping was not correctly constructed");

                ValueInput::argument(ty, argument as u32)
            } else {
                ValueInput { ty, origin }
            }
        };

        let mut value_inputs = Vec::with_capacity(branching_node.sub_tree_inputs.len() + 1);

        // Connect the first input to the branch selector predicate.
        value_inputs.push(resolve_input(branching_node.branch_selector, TY_PREDICATE));

        // Connect inputs for the emulation values required by the branching node's child nodes.
        for child_input in branching_node.sub_tree_inputs.iter().copied() {
            let ty = self.rvsdg.value_origin_ty(self.outer_region, child_input);

            value_inputs.push(resolve_input(child_input, ty));
        }

        // Connect inputs for any non-pointer additional values passed in for the operation (e.g.
        // the value to be stored by an OpStore).
        if let Some(input_mapping) = input_mapping {
            // We're inside a branching node, the additional inputs will be available as
            // arguments after the emulation arguments represented in the input_mapping.
            let base_index = input_mapping.len();

            for (i, additional_value) in self.additional_values.iter().enumerate() {
                let argument_index = base_index + i;

                value_inputs.push(ValueInput::argument(
                    additional_value.ty,
                    argument_index as u32,
                ));
            }
        } else {
            // We're processing the root node, the additional inputs should be available
            // directly
            value_inputs.extend(self.additional_values.iter().copied());
        }

        let value_outputs = if let Some(output_ty) = self.output_ty {
            vec![ValueOutput::new(output_ty)]
        } else {
            vec![]
        };

        let state_origin = if input_mapping.is_some() {
            // We're inside a switch node for a branching node, directly connect to the state
            // argument.
            StateOrigin::Argument
        } else {
            // This was the root of the emulation tree, so we're inside the outer region. Connect
            // to the provided state origin for the original load/store operation.
            self.state_origin
        };

        let switch_node =
            self.rvsdg
                .add_switch(region, value_inputs, value_outputs, Some(state_origin));

        for child in &branching_node.branches {
            let branch = self.rvsdg.add_switch_branch(switch_node);

            let node = match child {
                PointerReconstructionNode::Branching(node) => {
                    self.visit_branching_node(branch, node, Some(&branching_node.sub_tree_inputs))
                }
                PointerReconstructionNode::Leaf(node) => {
                    self.visit_leaf_node(branch, node, Some(&branching_node.sub_tree_inputs))
                }
            };

            if self.output_ty.is_some() {
                self.rvsdg.reconnect_region_result(
                    branch,
                    0,
                    ValueOrigin::Output {
                        producer: node,
                        output: 0,
                    },
                )
            }
        }

        switch_node
    }

    fn visit_leaf_node(
        &mut self,
        region: Region,
        leaf_node: &LeafNode,
        input_mapping: Option<&IndexSet<ValueOrigin>>,
    ) -> Node {
        let resolve_input = |origin, ty| {
            if let Some(input_mapping) = input_mapping {
                let argument = input_mapping
                    .get_index_of(&origin)
                    .expect("input mapping was not correctly constructed");

                ValueInput::argument(ty, argument as u32)
            } else {
                ValueInput { ty, origin }
            }
        };

        let ptr_ty = leaf_node.root_pointer.ty;
        let ptr_input = resolve_input(leaf_node.root_pointer.origin, ptr_ty);

        let ptr_input = if leaf_node.access_chain.is_empty() {
            ptr_input
        } else {
            let mut ptr_input = ptr_input;

            for access in &leaf_node.access_chain {
                let (node, ty) = match access {
                    Access::Field(field_index) => {
                        let node = self.rvsdg.add_op_field_ptr(region, ptr_input, *field_index);
                        let ty = self.rvsdg[node].expect_op_field_ptr().value_output().ty;

                        (node, ty)
                    }
                    Access::StaticElement(element_index) => {
                        let index_node = self.rvsdg.add_const_u32(region, *element_index);
                        let node = self.rvsdg.add_op_element_ptr(
                            region,
                            ptr_input,
                            ValueInput::output(TY_U32, index_node, 0),
                        );
                        let ty = self.rvsdg[node].expect_op_element_ptr().value_output().ty;

                        (node, ty)
                    }
                    Access::DynamicElement(origin) => {
                        let input = resolve_input(*origin, TY_U32);
                        let node = self.rvsdg.add_op_element_ptr(region, ptr_input, input);
                        let ty = self.rvsdg[node].expect_op_element_ptr().value_output().ty;

                        (node, ty)
                    }
                };

                ptr_input = ValueInput::output(ty, node, 0);
            }

            ptr_input
        };

        if let Some(input_mapping) = input_mapping {
            // We're inside a switch node that represents a parent branching node. The additional
            // values will have been passed in as arguments after all the arguments used for pointer
            // emulation itself.

            let base_index = input_mapping.len();
            let additional_values = self
                .additional_values
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let argument_index = base_index + i;

                    ValueInput::argument(v.ty, argument_index as u32)
                })
                .collect::<SmallVec<[ValueInput; 2]>>();

            (self.op_gen)(
                self.rvsdg,
                region,
                ptr_input,
                &additional_values,
                StateOrigin::Argument,
            )
        } else {
            // There is no parent branching node, so we are inside the outer region and the
            // additional values can be connected to directly.
            (self.op_gen)(
                self.rvsdg,
                region,
                ptr_input,
                self.additional_values,
                self.state_origin,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::rvsdg::ValueUser;
    use crate::ty::{TY_DUMMY, TY_PTR_U32, TypeKind};
    use crate::{BinaryOperator, FnArg, FnSig, Function, Module, Symbol, thin_set};

    #[test]
    fn test_emulate_single_switch_output_load() {
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

        let ptr_0 = rvsdg.add_op_alloca(region, TY_U32);
        let ptr_1 = rvsdg.add_op_alloca(region, TY_U32);

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(ptr_ty, ptr_0, 0),
                ValueInput::output(ptr_ty, ptr_1, 0),
            ],
            vec![ValueOutput::new(ptr_ty)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);

        rvsdg.reconnect_region_result(branch_0, 0, ValueOrigin::Argument(0));

        let branch_1 = rvsdg.add_switch_branch(switch_node);

        rvsdg.reconnect_region_result(branch_1, 0, ValueOrigin::Argument(1));

        let load_op = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, switch_node, 0),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_op,
                output: 0,
            },
        );

        let mut emulation_context = EmulationContext::new();

        emulation_context.emulate_op_load(&mut rvsdg, load_op);

        let ValueOrigin::Output {
            producer: emulation_switch_node,
            output: 0,
        } = rvsdg[region].value_results()[0].origin
        else {
            panic!("the function result should connect to the first output of a node")
        };

        let emulation_switch_data = rvsdg[emulation_switch_node].expect_switch();

        assert_eq!(
            emulation_switch_data.value_inputs(),
            &[
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(ptr_ty, ptr_0, 0),
                ValueInput::output(ptr_ty, ptr_1, 0),
            ]
        );
        assert_eq!(
            emulation_switch_data.value_outputs(),
            &[ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Result(0)]
            }]
        );
        assert_eq!(emulation_switch_data.branches().len(), 2);

        let emulation_branch_0 = emulation_switch_data.branches()[0];

        let ValueOrigin::Output {
            producer: emulation_load_0_node,
            output: 0,
        } = rvsdg[emulation_branch_0].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_load_0_data = rvsdg[emulation_load_0_node].expect_op_load();

        assert_eq!(
            emulation_load_0_data.value_inputs(),
            &[ValueInput::argument(ptr_ty, 0)]
        );

        let emulation_branch_1 = emulation_switch_data.branches()[1];

        let ValueOrigin::Output {
            producer: emulation_load_1_node,
            output: 0,
        } = rvsdg[emulation_branch_1].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_load_1_data = rvsdg[emulation_load_1_node].expect_op_load();

        assert_eq!(
            emulation_load_1_data.value_inputs(),
            &[ValueInput::argument(ptr_ty, 1)]
        );

        assert_eq!(
            &rvsdg[ptr_0].expect_op_alloca().value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: switch_node,
                    input: 1,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 1,
                }
            ],
            "the emulation node should have been added as a user to the first alloca node"
        );

        assert_eq!(
            &rvsdg[ptr_1].expect_op_alloca().value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: switch_node,
                    input: 2,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 2,
                }
            ],
            "the emulation node should have been added as a user to the second alloca node"
        );

        assert!(
            rvsdg[switch_node].value_outputs()[0].users.is_empty(),
            "original switch node should no longer have any users"
        );
    }

    #[test]
    fn test_emulate_single_switch_output_load_inside_switch() {
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

        let ptr_0 = rvsdg.add_op_alloca(region, TY_U32);
        let ptr_1 = rvsdg.add_op_alloca(region, TY_U32);

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let switch_0_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(ptr_ty, ptr_0, 0),
                ValueInput::output(ptr_ty, ptr_1, 0),
            ],
            vec![ValueOutput::new(ptr_ty)],
            None,
        );

        let switch_0_branch_0 = rvsdg.add_switch_branch(switch_0_node);

        rvsdg.reconnect_region_result(switch_0_branch_0, 0, ValueOrigin::Argument(0));

        let switch_0_branch_1 = rvsdg.add_switch_branch(switch_0_node);

        rvsdg.reconnect_region_result(switch_0_branch_1, 0, ValueOrigin::Argument(1));

        let switch_1_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(ptr_ty, switch_0_node, 0),
            ],
            vec![ValueOutput::new(TY_U32)],
            Some(StateOrigin::Argument),
        );

        let switch_1_branch_0 = rvsdg.add_switch_branch(switch_1_node);

        let load_op = rvsdg.add_op_load(
            switch_1_branch_0,
            ValueInput::argument(ptr_ty, 0),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            switch_1_branch_0,
            0,
            ValueOrigin::Output {
                producer: load_op,
                output: 0,
            },
        );

        let switch_1_branch_1 = rvsdg.add_switch_branch(switch_1_node);

        let fallback = rvsdg.add_const_u32(switch_1_branch_1, 0);

        rvsdg.reconnect_region_result(
            switch_1_branch_1,
            0,
            ValueOrigin::Output {
                producer: fallback,
                output: 0,
            },
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: switch_1_node,
                output: 0,
            },
        );

        let mut emulation_context = EmulationContext::new();

        emulation_context.emulate_op_load(&mut rvsdg, load_op);

        let ValueOrigin::Output {
            producer: emulation_switch_node,
            output: 0,
        } = rvsdg[switch_1_branch_0].value_results()[0].origin
        else {
            panic!("the second switch's result should connect to the first output of a node")
        };

        let emulation_switch_data = rvsdg[emulation_switch_node].expect_switch();

        assert_eq!(
            emulation_switch_data.value_inputs(),
            &[
                ValueInput::argument(TY_PREDICATE, 1),
                ValueInput::argument(ptr_ty, 2),
                ValueInput::argument(ptr_ty, 3),
            ]
        );
        assert_eq!(
            emulation_switch_data.value_outputs(),
            &[ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Result(0)]
            }]
        );

        assert_eq!(
            rvsdg[switch_1_node].value_inputs(),
            &[
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(ptr_ty, switch_0_node, 0),
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(ptr_ty, ptr_0, 0),
                ValueInput::output(ptr_ty, ptr_1, 0),
            ],
            "the predicate and both alloca nodes have been added as inputs to the second switch node"
        );

        assert!(
            rvsdg[switch_1_branch_0].value_arguments()[0]
                .users
                .is_empty(),
            "the second switch's variable pointer argument in the first branch should no longer \
            have any users"
        );
        assert_eq!(
            &rvsdg[switch_1_branch_0].value_arguments()[1].users,
            &thin_set![ValueUser::Input {
                consumer: emulation_switch_node,
                input: 0
            }],
            "the newly added predicate argument in the second switch's first branch should be used \
            by the emulation node"
        );
        assert_eq!(
            &rvsdg[switch_1_branch_0].value_arguments()[2].users,
            &thin_set![ValueUser::Input {
                consumer: emulation_switch_node,
                input: 1
            }],
            "the newly added ptr_0 argument in the second switch's first branch should be used \
            by the emulation node"
        );
        assert_eq!(
            &rvsdg[switch_1_branch_0].value_arguments()[3].users,
            &thin_set![ValueUser::Input {
                consumer: emulation_switch_node,
                input: 2
            }],
            "the newly added ptr_1 argument in the second switch's first branch should be used \
            by the emulation node"
        );

        assert!(
            rvsdg[switch_1_branch_1].value_arguments()[0]
                .users
                .is_empty(),
            "the second switch's variable pointer argument in the second branch should not have \
            any users"
        );
        assert!(
            rvsdg[switch_1_branch_1].value_arguments()[1]
                .users
                .is_empty(),
            "the newly added predicate argument in the second switch's second branch should not \
            have any users"
        );
        assert!(
            rvsdg[switch_1_branch_1].value_arguments()[2]
                .users
                .is_empty(),
            "the newly added ptr_0 argument in the second switch's second branch should not \
            have any users"
        );
        assert!(
            rvsdg[switch_1_branch_1].value_arguments()[3]
                .users
                .is_empty(),
            "the newly added ptr_1 argument in the second switch's second branch should not \
            have any users"
        );

        assert_eq!(
            &rvsdg[region].value_arguments()[0].users,
            &thin_set![
                ValueUser::Input {
                    consumer: switch_0_node,
                    input: 0,
                },
                ValueUser::Input {
                    consumer: switch_1_node,
                    input: 0,
                },
                ValueUser::Input {
                    consumer: switch_1_node,
                    input: 2,
                }
            ],
            "the second switch node should have been added as a user to the predicate function \
            argument"
        );

        assert_eq!(
            &rvsdg[ptr_0].expect_op_alloca().value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: switch_0_node,
                    input: 1,
                },
                ValueUser::Input {
                    consumer: switch_1_node,
                    input: 3,
                }
            ],
            "the second switch node should have been added as a user to the first alloca node"
        );

        assert_eq!(
            &rvsdg[ptr_1].expect_op_alloca().value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: switch_0_node,
                    input: 2,
                },
                ValueUser::Input {
                    consumer: switch_1_node,
                    input: 4,
                }
            ],
            "the second switch node should have been added as a user to the second alloca node"
        );
    }

    #[test]
    fn test_emulate_single_static_access_single_switch_output_load() {
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

        let array_ty = module.ty.register(TypeKind::Array {
            element_ty: TY_U32,
            count: 2,
            stride: 4,
        });
        let array_ptr_ty = module.ty.register(TypeKind::Ptr(array_ty));

        let array_alloca_node = rvsdg.add_op_alloca(region, array_ty);
        let index_node = rvsdg.add_const_u32(region, 1);

        let ptr_0_node = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(array_ptr_ty, array_alloca_node, 0),
            ValueInput::output(TY_U32, index_node, 0),
        );
        let ptr_1_node = rvsdg.add_op_alloca(region, TY_U32);

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(ptr_ty, ptr_0_node, 0),
                ValueInput::output(ptr_ty, ptr_1_node, 0),
            ],
            vec![ValueOutput::new(ptr_ty)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);

        rvsdg.reconnect_region_result(branch_0, 0, ValueOrigin::Argument(0));

        let branch_1 = rvsdg.add_switch_branch(switch_node);

        rvsdg.reconnect_region_result(branch_1, 0, ValueOrigin::Argument(1));

        let load_op = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, switch_node, 0),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_op,
                output: 0,
            },
        );

        let mut emulation_context = EmulationContext::new();

        emulation_context.emulate_op_load(&mut rvsdg, load_op);

        let ValueOrigin::Output {
            producer: emulation_switch_node,
            output: 0,
        } = rvsdg[region].value_results()[0].origin
        else {
            panic!("the function result should connect to the first output of a node")
        };

        let emulation_switch_data = rvsdg[emulation_switch_node].expect_switch();

        assert_eq!(
            emulation_switch_data.value_inputs(),
            &[
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(array_ptr_ty, array_alloca_node, 0),
                ValueInput::output(ptr_ty, ptr_1_node, 0),
            ]
        );
        assert_eq!(
            emulation_switch_data.value_outputs(),
            &[ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Result(0)]
            }]
        );
        assert_eq!(emulation_switch_data.branches().len(), 2);

        let emulation_branch_0 = emulation_switch_data.branches()[0];

        let ValueOrigin::Output {
            producer: emulation_0_load_node,
            output: 0,
        } = rvsdg[emulation_branch_0].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_0_load_data = rvsdg[emulation_0_load_node].expect_op_load();

        let ValueOrigin::Output {
            producer: emulation_0_ep_node,
            output: 0,
        } = emulation_0_load_data.ptr_input().origin
        else {
            panic!("the load node in branch `0` should connect to the first output of a node")
        };

        let emulation_ep_data = rvsdg[emulation_0_ep_node].expect_op_element_ptr();

        assert_eq!(emulation_ep_data.value_inputs().len(), 2);
        assert_eq!(
            emulation_ep_data.ptr_input(),
            &ValueInput::argument(array_ptr_ty, 0)
        );

        let ValueOrigin::Output {
            producer: emulation_0_index_node,
            output: 0,
        } = emulation_ep_data.index_input().origin
        else {
            panic!("the second element-ptr node input should connect to the first output of a node")
        };

        let emulation_0_index_data = rvsdg[emulation_0_index_node].expect_const_u32();

        assert_eq!(emulation_0_index_data.value(), 1);

        let emulation_branch_1 = emulation_switch_data.branches()[1];

        let ValueOrigin::Output {
            producer: emulation_1_load_node,
            output: 0,
        } = rvsdg[emulation_branch_1].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_1_load_data = rvsdg[emulation_1_load_node].expect_op_load();

        assert_eq!(
            emulation_1_load_data.value_inputs(),
            &[ValueInput::argument(ptr_ty, 1)]
        );

        assert_eq!(
            &rvsdg[array_alloca_node]
                .expect_op_alloca()
                .value_output()
                .users,
            &thin_set![
                ValueUser::Input {
                    consumer: ptr_0_node,
                    input: 0,
                },
                ValueUser::Input {
                    consumer: switch_node,
                    input: 3,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 1,
                }
            ],
            "the switch node and the emulation node should have been added as a user to the first \
            alloca node"
        );

        assert!(
            rvsdg[branch_0].value_arguments()[2].users.is_empty(),
            "the array pointer should have been made available inside the first branch of the \
            switch node, but should not be used"
        );
        assert!(
            rvsdg[branch_1].value_arguments()[2].users.is_empty(),
            "the array pointer should have been made available inside the second branch of the \
            switch node, but should not be used"
        );

        assert_eq!(
            &rvsdg[ptr_1_node].expect_op_alloca().value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: switch_node,
                    input: 2,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 2,
                }
            ],
            "the emulation node should have been added as a user to the second alloca node"
        );

        assert!(
            rvsdg[switch_node].value_outputs()[0].users.is_empty(),
            "original switch node should no longer have any users"
        );
    }

    #[test]
    fn test_emulate_single_dynamic_access_single_switch_output_load() {
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

        let array_ty = module.ty.register(TypeKind::Array {
            element_ty: TY_U32,
            count: 2,
            stride: 4,
        });
        let array_ptr_ty = module.ty.register(TypeKind::Ptr(array_ty));

        let array_alloca_node = rvsdg.add_op_alloca(region, array_ty);

        let ptr_0_node = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(array_ptr_ty, array_alloca_node, 0),
            ValueInput::argument(TY_U32, 1),
        );
        let ptr_1_node = rvsdg.add_op_alloca(region, TY_U32);

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(ptr_ty, ptr_0_node, 0),
                ValueInput::output(ptr_ty, ptr_1_node, 0),
            ],
            vec![ValueOutput::new(ptr_ty)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);

        rvsdg.reconnect_region_result(branch_0, 0, ValueOrigin::Argument(0));

        let branch_1 = rvsdg.add_switch_branch(switch_node);

        rvsdg.reconnect_region_result(branch_1, 0, ValueOrigin::Argument(1));

        let load_op = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, switch_node, 0),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_op,
                output: 0,
            },
        );

        let mut emulation_context = EmulationContext::new();

        emulation_context.emulate_op_load(&mut rvsdg, load_op);

        let ValueOrigin::Output {
            producer: emulation_switch_node,
            output: 0,
        } = rvsdg[region].value_results()[0].origin
        else {
            panic!("the function result should connect to the first output of a node")
        };

        let emulation_switch_data = rvsdg[emulation_switch_node].expect_switch();

        assert_eq!(
            emulation_switch_data.value_inputs(),
            &[
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(array_ptr_ty, array_alloca_node, 0),
                ValueInput::argument(TY_U32, 1),
                ValueInput::output(ptr_ty, ptr_1_node, 0),
            ]
        );
        assert_eq!(
            emulation_switch_data.value_outputs(),
            &[ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Result(0)]
            }]
        );
        assert_eq!(emulation_switch_data.branches().len(), 2);

        let emulation_branch_0 = emulation_switch_data.branches()[0];

        let ValueOrigin::Output {
            producer: emulation_0_load_node,
            output: 0,
        } = rvsdg[emulation_branch_0].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_0_load_data = rvsdg[emulation_0_load_node].expect_op_load();

        let ValueOrigin::Output {
            producer: emulation_0_ep_node,
            output: 0,
        } = emulation_0_load_data.ptr_input().origin
        else {
            panic!("the load node in branch `0` should connect to the first output of a node")
        };

        let emulation_ep_data = rvsdg[emulation_0_ep_node].expect_op_element_ptr();

        assert_eq!(
            emulation_ep_data.value_inputs(),
            &[
                ValueInput::argument(array_ptr_ty, 0),
                ValueInput::argument(TY_U32, 1),
            ]
        );

        let emulation_branch_1 = emulation_switch_data.branches()[1];

        let ValueOrigin::Output {
            producer: emulation_1_load_node,
            output: 0,
        } = rvsdg[emulation_branch_1].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_1_load_data = rvsdg[emulation_1_load_node].expect_op_load();

        assert_eq!(
            emulation_1_load_data.value_inputs(),
            &[ValueInput::argument(ptr_ty, 2)]
        );

        assert_eq!(
            &rvsdg[array_alloca_node]
                .expect_op_alloca()
                .value_output()
                .users,
            &thin_set![
                ValueUser::Input {
                    consumer: ptr_0_node,
                    input: 0,
                },
                ValueUser::Input {
                    consumer: switch_node,
                    input: 3,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 1,
                }
            ],
            "the switch node and the emulation node should have been added as a user to the first \
            alloca node"
        );

        assert!(
            rvsdg[branch_0].value_arguments()[2].users.is_empty(),
            "the array pointer should have been made available inside the first branch of the \
            switch node, but should not be used"
        );
        assert!(
            rvsdg[branch_0].value_arguments()[3].users.is_empty(),
            "the dynamic index should have been made available inside the first branch of the \
            switch node, but should not be used"
        );
        assert!(
            rvsdg[branch_1].value_arguments()[2].users.is_empty(),
            "the array pointer should have been made available inside the second branch of the \
            switch node, but should not be used"
        );
        assert!(
            rvsdg[branch_1].value_arguments()[3].users.is_empty(),
            "the dynamic index should have been made available inside the second branch of the \
            switch node, but should not be used"
        );

        assert_eq!(
            &rvsdg[ptr_1_node].expect_op_alloca().value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: switch_node,
                    input: 2,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 3,
                }
            ],
            "the emulation node should have been added as a user to the second alloca node"
        );

        assert_eq!(
            &rvsdg[region].value_arguments()[1].users,
            &thin_set![
                ValueUser::Input {
                    consumer: ptr_0_node,
                    input: 1,
                },
                ValueUser::Input {
                    consumer: switch_node,
                    input: 4,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 2,
                }
            ],
            "the switch node and the emulation node should have been added as a user to the \
            dynamic index argument"
        );

        assert!(
            rvsdg[switch_node].value_outputs()[0].users.is_empty(),
            "original switch node should no longer have any users"
        );
    }

    #[test]
    fn test_emulate_double_static_access_single_switch_output_load() {
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

        let array_ty = module.ty.register(TypeKind::Array {
            element_ty: TY_U32,
            count: 2,
            stride: 4,
        });
        let array_of_array_ty = module.ty.register(TypeKind::Array {
            element_ty: array_ty,
            count: 2,
            stride: 8,
        });
        let array_ptr_ty = module.ty.register(TypeKind::Ptr(array_ty));
        let array_of_array_ptr_ty = module.ty.register(TypeKind::Ptr(array_of_array_ty));
        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let array_alloca_node = rvsdg.add_op_alloca(region, array_of_array_ty);
        let index_node = rvsdg.add_const_u32(region, 1);

        let ptr_0_node = rvsdg.add_op_element_ptr(
            region,
            ValueInput::output(array_of_array_ptr_ty, array_alloca_node, 0),
            ValueInput::output(TY_U32, index_node, 0),
        );
        let ptr_1_node = rvsdg.add_op_alloca(region, TY_U32);

        let switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(array_ptr_ty, ptr_0_node, 0),
                ValueInput::output(ptr_ty, ptr_1_node, 0),
            ],
            vec![ValueOutput::new(ptr_ty)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);

        let branch_0_index_node = rvsdg.add_const_u32(branch_0, 0);
        let branch_0_ptr_node = rvsdg.add_op_element_ptr(
            branch_0,
            ValueInput::argument(array_ptr_ty, 0),
            ValueInput::output(TY_U32, branch_0_index_node, 0),
        );

        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: branch_0_ptr_node,
                output: 0,
            },
        );

        let branch_1 = rvsdg.add_switch_branch(switch_node);

        rvsdg.reconnect_region_result(branch_1, 0, ValueOrigin::Argument(1));

        let load_op = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, switch_node, 0),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_op,
                output: 0,
            },
        );

        let mut emulation_context = EmulationContext::new();

        emulation_context.emulate_op_load(&mut rvsdg, load_op);

        let ValueOrigin::Output {
            producer: emulation_switch_node,
            output: 0,
        } = rvsdg[region].value_results()[0].origin
        else {
            panic!("the function result should connect to the first output of a node")
        };

        let emulation_switch_data = rvsdg[emulation_switch_node].expect_switch();

        assert_eq!(
            emulation_switch_data.value_inputs(),
            &[
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(array_of_array_ptr_ty, array_alloca_node, 0),
                ValueInput::output(ptr_ty, ptr_1_node, 0),
            ]
        );
        assert_eq!(
            emulation_switch_data.value_outputs(),
            &[ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Result(0)]
            }]
        );
        assert_eq!(emulation_switch_data.branches().len(), 2);

        let emulation_branch_0 = emulation_switch_data.branches()[0];

        let ValueOrigin::Output {
            producer: emulation_0_load_node,
            output: 0,
        } = rvsdg[emulation_branch_0].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_0_load_data = rvsdg[emulation_0_load_node].expect_op_load();

        let ValueOrigin::Output {
            producer: emulation_0_ep1_node,
            output: 0,
        } = emulation_0_load_data.ptr_input().origin
        else {
            panic!("the load node in branch `0` should connect to the first output of a node")
        };

        let emulation_ep1_data = rvsdg[emulation_0_ep1_node].expect_op_element_ptr();

        let ValueOrigin::Output {
            producer: emulation_0_index_1_node,
            output: 0,
        } = emulation_ep1_data.index_input().origin
        else {
            panic!("the third element-ptr node input should connect to the first output of a node")
        };

        let emulation_0_index_1_data = rvsdg[emulation_0_index_1_node].expect_const_u32();

        assert_eq!(emulation_0_index_1_data.value(), 0);

        let ValueOrigin::Output {
            producer: emulation_0_ep0_node,
            output: 0,
        } = emulation_ep1_data.ptr_input().origin
        else {
            panic!(
                "the second element-pointer node input should connect to the first output of a \
                node"
            )
        };

        let emulation_0_ep0_data = rvsdg[emulation_0_ep0_node].expect_op_element_ptr();

        assert_eq!(
            emulation_0_ep0_data.ptr_input(),
            &ValueInput::argument(array_of_array_ptr_ty, 0)
        );

        let ValueOrigin::Output {
            producer: emulation_0_index_0_node,
            output: 0,
        } = emulation_0_ep0_data.index_input().origin
        else {
            panic!("the second element-ptr node input should connect to the first output of a node")
        };

        let emulation_0_index_0_data = rvsdg[emulation_0_index_0_node].expect_const_u32();

        assert_eq!(emulation_0_index_0_data.value(), 1);

        let emulation_branch_1 = emulation_switch_data.branches()[1];

        let ValueOrigin::Output {
            producer: emulation_1_load_node,
            output: 0,
        } = rvsdg[emulation_branch_1].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_1_load_data = rvsdg[emulation_1_load_node].expect_op_load();

        assert_eq!(
            emulation_1_load_data.value_inputs(),
            &[ValueInput::argument(ptr_ty, 1)]
        );

        assert_eq!(
            &rvsdg[array_alloca_node]
                .expect_op_alloca()
                .value_output()
                .users,
            &thin_set![
                ValueUser::Input {
                    consumer: ptr_0_node,
                    input: 0,
                },
                ValueUser::Input {
                    consumer: switch_node,
                    input: 3,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 1,
                }
            ],
            "the switch node and the emulation node should have been added as a user to the first \
            alloca node"
        );

        assert!(
            rvsdg[branch_0].value_arguments()[2].users.is_empty(),
            "the array pointer should have been made available inside the first branch of the \
            switch node, but should not be used"
        );
        assert!(
            rvsdg[branch_1].value_arguments()[2].users.is_empty(),
            "the array pointer should have been made available inside the second branch of the \
            switch node, but should not be used"
        );

        assert_eq!(
            &rvsdg[ptr_1_node].expect_op_alloca().value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: switch_node,
                    input: 2,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 2,
                }
            ],
            "the emulation node should have been added as a user to the second alloca node"
        );

        assert!(
            rvsdg[switch_node].value_outputs()[0].users.is_empty(),
            "original switch node should no longer have any users"
        );
    }

    #[test]
    fn test_emulate_double_switch_output_load() {
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
                        ty: TY_PREDICATE,
                        shader_io_binding: None,
                    },
                ],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let ptr_0_node = rvsdg.add_op_alloca(region, TY_U32);
        let ptr_1_node = rvsdg.add_op_alloca(region, TY_U32);
        let ptr_2_node = rvsdg.add_op_alloca(region, TY_U32);

        let first_switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 1),
                ValueInput::output(ptr_ty, ptr_0_node, 0),
                ValueInput::output(ptr_ty, ptr_1_node, 0),
            ],
            vec![ValueOutput::new(ptr_ty)],
            None,
        );

        let first_switch_branch_0 = rvsdg.add_switch_branch(first_switch_node);

        rvsdg.reconnect_region_result(first_switch_branch_0, 0, ValueOrigin::Argument(0));

        let first_switch_branch_1 = rvsdg.add_switch_branch(first_switch_node);

        rvsdg.reconnect_region_result(first_switch_branch_1, 0, ValueOrigin::Argument(1));

        let second_switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(ptr_ty, first_switch_node, 0),
                ValueInput::output(ptr_ty, ptr_2_node, 0),
            ],
            vec![ValueOutput::new(ptr_ty)],
            None,
        );

        let second_switch_branch_0 = rvsdg.add_switch_branch(second_switch_node);

        rvsdg.reconnect_region_result(second_switch_branch_0, 0, ValueOrigin::Argument(0));

        let second_switch_branch_1 = rvsdg.add_switch_branch(second_switch_node);

        rvsdg.reconnect_region_result(second_switch_branch_1, 0, ValueOrigin::Argument(1));

        let load_op = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, second_switch_node, 0),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_op,
                output: 0,
            },
        );

        let mut emulation_context = EmulationContext::new();

        emulation_context.emulate_op_load(&mut rvsdg, load_op);

        let ValueOrigin::Output {
            producer: emulation_switch_node,
            output: 0,
        } = rvsdg[region].value_results()[0].origin
        else {
            panic!("the function result should connect to the first output of a node")
        };

        let emulation_switch_data = rvsdg[emulation_switch_node].expect_switch();

        assert_eq!(
            emulation_switch_data.value_inputs(),
            &[
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::argument(TY_PREDICATE, 1),
                ValueInput::output(ptr_ty, ptr_0_node, 0),
                ValueInput::output(ptr_ty, ptr_1_node, 0),
                ValueInput::output(ptr_ty, ptr_2_node, 0),
            ]
        );
        assert_eq!(
            emulation_switch_data.value_outputs(),
            &[ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Result(0)]
            }]
        );
        assert_eq!(emulation_switch_data.branches().len(), 2);

        let emulation_branch_0 = emulation_switch_data.branches()[0];

        let ValueOrigin::Output {
            producer: emulation_0_switch_node,
            output: 0,
        } = rvsdg[emulation_branch_0].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_0_switch_data = rvsdg[emulation_0_switch_node].expect_switch();

        assert_eq!(
            emulation_0_switch_data.value_inputs(),
            &[
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::argument(ptr_ty, 1),
                ValueInput::argument(ptr_ty, 2),
            ]
        );
        assert_eq!(
            emulation_0_switch_data.value_outputs(),
            &[ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Result(0)]
            }]
        );
        assert_eq!(emulation_0_switch_data.branches().len(), 2);

        let emulation_branch_0_0 = emulation_0_switch_data.branches()[0];

        let ValueOrigin::Output {
            producer: emulation_0_0_load_node,
            output: 0,
        } = rvsdg[emulation_branch_0_0].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_0_0_load_data = rvsdg[emulation_0_0_load_node].expect_op_load();

        assert_eq!(
            emulation_0_0_load_data.value_inputs(),
            &[ValueInput::argument(ptr_ty, 0)]
        );

        let emulation_branch_0_1 = emulation_0_switch_data.branches()[1];

        let ValueOrigin::Output {
            producer: emulation_0_1_load_node,
            output: 0,
        } = rvsdg[emulation_branch_0_1].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_0_1_load_data = rvsdg[emulation_0_1_load_node].expect_op_load();

        assert_eq!(
            emulation_0_1_load_data.value_inputs(),
            &[ValueInput::argument(ptr_ty, 1)]
        );

        let emulation_branch_1 = emulation_switch_data.branches()[1];

        let ValueOrigin::Output {
            producer: emulation_1_load_node,
            output: 0,
        } = rvsdg[emulation_branch_1].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_1_load_data = rvsdg[emulation_1_load_node].expect_op_load();

        assert_eq!(
            emulation_1_load_data.value_inputs(),
            &[ValueInput::argument(ptr_ty, 3)]
        );

        assert_eq!(
            &rvsdg[ptr_0_node].expect_op_alloca().value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: first_switch_node,
                    input: 1,
                },
                ValueUser::Input {
                    consumer: second_switch_node,
                    input: 4,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 2,
                }
            ],
            "the second switch node and the emulation should have been added as a user to the \
            first alloca node"
        );

        assert_eq!(
            &rvsdg[ptr_1_node].expect_op_alloca().value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: first_switch_node,
                    input: 2,
                },
                ValueUser::Input {
                    consumer: second_switch_node,
                    input: 5,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 3,
                }
            ],
            "the second switch node and the emulation should have been added as a user to the \
            second alloca node"
        );

        assert_eq!(
            &rvsdg[ptr_2_node].expect_op_alloca().value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: second_switch_node,
                    input: 2,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 4,
                }
            ],
            "the emulation node should have been added as a user to the third alloca node"
        );

        assert!(
            rvsdg[second_switch_node].value_outputs()[0]
                .users
                .is_empty(),
            "the second switch node should no longer have any users"
        );
    }

    #[test]
    fn test_emulate_nested_switch_output_load() {
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
                        ty: TY_PREDICATE,
                        shader_io_binding: None,
                    },
                ],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let ptr_0_node = rvsdg.add_op_alloca(region, TY_U32);
        let ptr_1_node = rvsdg.add_op_alloca(region, TY_U32);
        let ptr_2_node = rvsdg.add_op_alloca(region, TY_U32);

        let outer_switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::argument(TY_PREDICATE, 1),
                ValueInput::output(ptr_ty, ptr_0_node, 0),
                ValueInput::output(ptr_ty, ptr_1_node, 0),
                ValueInput::output(ptr_ty, ptr_2_node, 0),
            ],
            vec![ValueOutput::new(ptr_ty)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(outer_switch_node);

        let inner_switch_node = rvsdg.add_switch(
            branch_0,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::argument(ptr_ty, 1),
                ValueInput::argument(ptr_ty, 2),
            ],
            vec![ValueOutput::new(ptr_ty)],
            None,
        );

        let branch_0_branch_0 = rvsdg.add_switch_branch(inner_switch_node);

        rvsdg.reconnect_region_result(branch_0_branch_0, 0, ValueOrigin::Argument(0));

        let branch_0_branch_1 = rvsdg.add_switch_branch(inner_switch_node);

        rvsdg.reconnect_region_result(branch_0_branch_1, 0, ValueOrigin::Argument(1));

        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: inner_switch_node,
                output: 0,
            },
        );

        let branch_1 = rvsdg.add_switch_branch(outer_switch_node);

        rvsdg.reconnect_region_result(branch_1, 0, ValueOrigin::Argument(3));

        let load_op = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, outer_switch_node, 0),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_op,
                output: 0,
            },
        );

        let mut emulation_context = EmulationContext::new();

        emulation_context.emulate_op_load(&mut rvsdg, load_op);

        let ValueOrigin::Output {
            producer: emulation_switch_node,
            output: 0,
        } = rvsdg[region].value_results()[0].origin
        else {
            panic!("the function result should connect to the first output of a node")
        };

        let emulation_switch_data = rvsdg[emulation_switch_node].expect_switch();

        assert_eq!(
            emulation_switch_data.value_inputs(),
            &[
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::argument(TY_PREDICATE, 1),
                ValueInput::output(ptr_ty, ptr_0_node, 0),
                ValueInput::output(ptr_ty, ptr_1_node, 0),
                ValueInput::output(ptr_ty, ptr_2_node, 0),
            ]
        );
        assert_eq!(
            emulation_switch_data.value_outputs(),
            &[ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Result(0)]
            }]
        );
        assert_eq!(emulation_switch_data.branches().len(), 2);

        let emulation_branch_0 = emulation_switch_data.branches()[0];

        let ValueOrigin::Output {
            producer: emulation_0_switch_node,
            output: 0,
        } = rvsdg[emulation_branch_0].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_0_switch_data = rvsdg[emulation_0_switch_node].expect_switch();

        assert_eq!(
            emulation_0_switch_data.value_inputs(),
            &[
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::argument(ptr_ty, 1),
                ValueInput::argument(ptr_ty, 2),
            ]
        );
        assert_eq!(
            emulation_0_switch_data.value_outputs(),
            &[ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Result(0)]
            }]
        );
        assert_eq!(emulation_0_switch_data.branches().len(), 2);

        let emulation_branch_0_0 = emulation_0_switch_data.branches()[0];

        let ValueOrigin::Output {
            producer: emulation_0_0_load_node,
            output: 0,
        } = rvsdg[emulation_branch_0_0].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_0_0_load_data = rvsdg[emulation_0_0_load_node].expect_op_load();

        assert_eq!(
            emulation_0_0_load_data.value_inputs(),
            &[ValueInput::argument(ptr_ty, 0)]
        );

        let emulation_branch_0_1 = emulation_0_switch_data.branches()[1];

        let ValueOrigin::Output {
            producer: emulation_0_1_load_node,
            output: 0,
        } = rvsdg[emulation_branch_0_1].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_0_1_load_data = rvsdg[emulation_0_1_load_node].expect_op_load();

        assert_eq!(
            emulation_0_1_load_data.value_inputs(),
            &[ValueInput::argument(ptr_ty, 1)]
        );

        let emulation_branch_1 = emulation_switch_data.branches()[1];

        let ValueOrigin::Output {
            producer: emulation_1_load_node,
            output: 0,
        } = rvsdg[emulation_branch_1].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_1_load_data = rvsdg[emulation_1_load_node].expect_op_load();

        assert_eq!(
            emulation_1_load_data.value_inputs(),
            &[ValueInput::argument(ptr_ty, 3)]
        );

        assert_eq!(
            &rvsdg[ptr_0_node].expect_op_alloca().value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: outer_switch_node,
                    input: 2,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 2,
                }
            ],
            "the emulation node should have been added as a user to the first alloca node"
        );

        assert_eq!(
            &rvsdg[ptr_1_node].expect_op_alloca().value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: outer_switch_node,
                    input: 3,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 3,
                }
            ],
            "the emulation node should have been added as a user to the second alloca node"
        );

        assert_eq!(
            &rvsdg[ptr_2_node].expect_op_alloca().value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: outer_switch_node,
                    input: 4,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 4,
                }
            ],
            "the emulation node should have been added as a user to the third alloca node"
        );

        assert!(
            rvsdg[outer_switch_node].value_outputs()[0].users.is_empty(),
            "original switch node should no longer have any users"
        );
    }

    #[test]
    fn test_emulate_single_switch_output_store() {
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
                ret_ty: None,
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let ptr_0 = rvsdg.add_op_alloca(region, TY_U32);
        let ptr_1 = rvsdg.add_op_alloca(region, TY_U32);

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(ptr_ty, ptr_0, 0),
                ValueInput::output(ptr_ty, ptr_1, 0),
            ],
            vec![ValueOutput::new(ptr_ty)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);

        rvsdg.reconnect_region_result(branch_0, 0, ValueOrigin::Argument(0));

        let branch_1 = rvsdg.add_switch_branch(switch_node);

        rvsdg.reconnect_region_result(branch_1, 0, ValueOrigin::Argument(1));

        let store_op = rvsdg.add_op_store(
            region,
            ValueInput::output(ptr_ty, switch_node, 0),
            ValueInput::argument(TY_U32, 1),
            StateOrigin::Argument,
        );

        let mut emulation_context = EmulationContext::new();

        emulation_context.emulate_op_store(&mut rvsdg, store_op);

        assert_eq!(rvsdg[region].value_arguments()[0].users.len(), 2);
        assert_eq!(
            rvsdg[region].value_arguments()[0].users[0],
            ValueUser::Input {
                consumer: switch_node,
                input: 0,
            }
        );

        let ValueUser::Input {
            consumer: emulation_switch_node,
            input: 0,
        } = rvsdg[region].value_arguments()[0].users[1]
        else {
            panic!(
                "the second user of the first argument should connect to the first input of a node"
            )
        };

        let emulation_switch_data = rvsdg[emulation_switch_node].expect_switch();

        assert_eq!(
            emulation_switch_data.value_inputs(),
            &[
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(ptr_ty, ptr_0, 0),
                ValueInput::output(ptr_ty, ptr_1, 0),
                ValueInput::argument(TY_U32, 1),
            ]
        );
        assert!(emulation_switch_data.value_outputs().is_empty());
        assert_eq!(emulation_switch_data.branches().len(), 2);

        let emulation_branch_0 = emulation_switch_data.branches()[0];

        assert_eq!(
            rvsdg[emulation_branch_0].value_arguments()[0].users.len(),
            1
        );
        assert_eq!(
            rvsdg[emulation_branch_0].value_arguments()[1].users.len(),
            0
        );
        assert_eq!(
            rvsdg[emulation_branch_0].value_arguments()[2].users.len(),
            1
        );

        let ValueUser::Input {
            consumer: emulation_0_store_node,
            input: 0,
        } = rvsdg[emulation_branch_0].value_arguments()[0].users[0]
        else {
            panic!(
                "the first user of the first argument should connect to the first input of a node"
            )
        };

        let emulation_0_store_data = rvsdg[emulation_0_store_node].expect_op_store();

        assert_eq!(
            emulation_0_store_data.value_input(),
            &ValueInput::argument(TY_U32, 2)
        );
        assert_eq!(
            rvsdg[emulation_branch_0].value_arguments()[2].users[0],
            ValueUser::Input {
                consumer: emulation_0_store_node,
                input: 1,
            }
        );

        let emulation_branch_1 = emulation_switch_data.branches()[1];

        assert_eq!(
            rvsdg[emulation_branch_1].value_arguments()[0].users.len(),
            0
        );
        assert_eq!(
            rvsdg[emulation_branch_1].value_arguments()[1].users.len(),
            1
        );
        assert_eq!(
            rvsdg[emulation_branch_1].value_arguments()[2].users.len(),
            1
        );

        let ValueUser::Input {
            consumer: emulation_1_store_node,
            input: 0,
        } = rvsdg[emulation_branch_1].value_arguments()[1].users[0]
        else {
            panic!(
                "the first user of the first argument should connect to the first input of a node"
            )
        };

        let emulation_1_store_data = rvsdg[emulation_1_store_node].expect_op_store();

        assert_eq!(
            emulation_1_store_data.value_input(),
            &ValueInput::argument(TY_U32, 2)
        );
        assert_eq!(
            rvsdg[emulation_branch_1].value_arguments()[2].users[0],
            ValueUser::Input {
                consumer: emulation_1_store_node,
                input: 1,
            }
        );

        assert_eq!(
            &rvsdg[region].value_arguments()[0].users,
            &thin_set![
                ValueUser::Input {
                    consumer: switch_node,
                    input: 0,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 0,
                }
            ],
            "the emulation node should have been added as a user to the first function argument"
        );

        assert_eq!(
            &rvsdg[region].value_arguments()[1].users,
            &thin_set![ValueUser::Input {
                consumer: emulation_switch_node,
                input: 3,
            }],
            "the store node should have been replaced with the emulation node as a user of the \
            second function argument"
        );

        assert_eq!(
            &rvsdg[ptr_0].expect_op_alloca().value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: switch_node,
                    input: 1,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 1,
                }
            ],
            "the emulation node should have been added as a user to the first alloca node"
        );

        assert_eq!(
            &rvsdg[ptr_1].expect_op_alloca().value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: switch_node,
                    input: 2,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 2,
                }
            ],
            "the emulation node should have been added as a user to the second alloca node"
        );

        assert!(
            rvsdg[switch_node].value_outputs()[0].users.is_empty(),
            "original switch node should no longer have any users"
        );
    }

    #[test]
    fn test_emulate_load_switch_output_reuse() {
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

        let array_ty = module.ty.register(TypeKind::Array {
            element_ty: TY_U32,
            count: 2,
            stride: 4,
        });
        let array_of_array_ty = module.ty.register(TypeKind::Array {
            element_ty: array_ty,
            count: 2,
            stride: 8,
        });
        let array_ptr_ty = module.ty.register(TypeKind::Ptr(array_ty));
        let array_of_array_ptr_ty = module.ty.register(TypeKind::Ptr(array_of_array_ty));
        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let array_alloca_node = rvsdg.add_op_alloca(region, array_of_array_ty);

        let switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(array_of_array_ptr_ty, array_alloca_node, 0),
                ValueInput::argument(TY_U32, 1),
                ValueInput::argument(TY_U32, 2),
            ],
            vec![ValueOutput::new(ptr_ty)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);

        let branch_0_const_node = rvsdg.add_const_u32(branch_0, 1);

        // Both index values are dynamic and originate inside the switch node for the first branch
        let branch_0_index_0 = rvsdg.add_op_binary(
            branch_0,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 1),
            ValueInput::output(TY_U32, branch_0_const_node, 0),
        );
        let branch_0_index_1 = rvsdg.add_op_binary(
            branch_0,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 2),
            ValueInput::output(TY_U32, branch_0_const_node, 0),
        );

        let branch_0_ptr_0_node = rvsdg.add_op_element_ptr(
            branch_0,
            ValueInput::argument(array_of_array_ptr_ty, 0),
            ValueInput::output(TY_U32, branch_0_index_0, 0),
        );
        let branch_0_ptr_1_node = rvsdg.add_op_element_ptr(
            branch_0,
            ValueInput::output(array_ptr_ty, branch_0_ptr_0_node, 0),
            ValueInput::output(TY_U32, branch_0_index_1, 0),
        );

        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: branch_0_ptr_1_node,
                output: 0,
            },
        );

        let branch_1 = rvsdg.add_switch_branch(switch_node);

        let branch_1_const_node = rvsdg.add_const_u32(branch_1, 1);

        // For the second branch, the first index is a dynamic value that originates inside the
        // switch node, the second index is a static value
        let branch_1_index_0 = rvsdg.add_op_binary(
            branch_1,
            BinaryOperator::Add,
            ValueInput::argument(TY_U32, 1),
            ValueInput::output(TY_U32, branch_1_const_node, 0),
        );
        let branch_1_index_1 = rvsdg.add_const_u32(branch_1, 0);

        let branch_1_ptr_0_node = rvsdg.add_op_element_ptr(
            branch_1,
            ValueInput::argument(array_of_array_ptr_ty, 0),
            ValueInput::output(TY_U32, branch_1_index_0, 0),
        );
        let branch_1_ptr_1_node = rvsdg.add_op_element_ptr(
            branch_1,
            ValueInput::output(array_ptr_ty, branch_1_ptr_0_node, 0),
            ValueInput::output(TY_U32, branch_1_index_1, 0),
        );

        rvsdg.reconnect_region_result(
            branch_1,
            0,
            ValueOrigin::Output {
                producer: branch_1_ptr_1_node,
                output: 0,
            },
        );

        let load_op = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, switch_node, 0),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_op,
                output: 0,
            },
        );

        let mut emulation_context = EmulationContext::new();

        emulation_context.emulate_op_load(&mut rvsdg, load_op);

        let ValueOrigin::Output {
            producer: emulation_switch_node,
            output: 0,
        } = rvsdg[region].value_results()[0].origin
        else {
            panic!("the function result should connect to the first output of a node")
        };

        let emulation_switch_data = rvsdg[emulation_switch_node].expect_switch();

        assert_eq!(
            emulation_switch_data.value_inputs(),
            &[
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(array_of_array_ptr_ty, array_alloca_node, 0),
                ValueInput::output(TY_U32, switch_node, 1),
                ValueInput::output(TY_U32, switch_node, 2),
            ]
        );
        assert_eq!(
            emulation_switch_data.value_outputs(),
            &[ValueOutput {
                ty: TY_U32,
                users: thin_set![ValueUser::Result(0)]
            }]
        );
        assert_eq!(emulation_switch_data.branches().len(), 2);

        let emulation_branch_0 = emulation_switch_data.branches()[0];

        let ValueOrigin::Output {
            producer: emulation_0_load_node,
            output: 0,
        } = rvsdg[emulation_branch_0].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_0_load_data = rvsdg[emulation_0_load_node].expect_op_load();

        let ValueOrigin::Output {
            producer: emulation_0_ep_1_node,
            output: 0,
        } = emulation_0_load_data.ptr_input().origin
        else {
            panic!("the load node in branch `0` should connect to the first output of a node")
        };

        let emulation_0_ep1_data = rvsdg[emulation_0_ep_1_node].expect_op_element_ptr();

        assert_eq!(
            emulation_0_ep1_data.index_input(),
            &ValueInput::argument(TY_U32, 2)
        );

        let ValueOrigin::Output {
            producer: emulation_0_ep_0_node,
            output: 0,
        } = emulation_0_ep1_data.ptr_input().origin
        else {
            panic!(
                "the second element-ptr node in branch `0` should connect to the first output of a \
                node"
            )
        };

        let emulation_0_ep0_data = rvsdg[emulation_0_ep_0_node].expect_op_element_ptr();

        assert_eq!(
            emulation_0_ep0_data.ptr_input(),
            &ValueInput::argument(array_of_array_ptr_ty, 0)
        );
        assert_eq!(
            emulation_0_ep0_data.index_input(),
            &ValueInput::argument(TY_U32, 1)
        );

        let emulation_branch_1 = emulation_switch_data.branches()[1];

        let ValueOrigin::Output {
            producer: emulation_1_load_node,
            output: 0,
        } = rvsdg[emulation_branch_1].value_results()[0].origin
        else {
            panic!("the branch result should connect to the first output of a node")
        };

        let emulation_1_load_data = rvsdg[emulation_1_load_node].expect_op_load();

        let ValueOrigin::Output {
            producer: emulation_1_ep1_node,
            output: 0,
        } = emulation_1_load_data.ptr_input().origin
        else {
            panic!("the load node in branch `0` should connect to the first output of a node")
        };

        let emulation_1_ep1_data = rvsdg[emulation_1_ep1_node].expect_op_element_ptr();

        let ValueOrigin::Output {
            producer: emulation_1_index_node,
            output: 0,
        } = emulation_1_ep1_data.index_input().origin
        else {
            panic!(
                "the index-input of the second element-ptr node in branch `1` should connect to \
            the first output of a node"
            )
        };

        let emulation_1_index_data = rvsdg[emulation_1_index_node].expect_const_u32();

        assert_eq!(emulation_1_index_data.value(), 0);

        let ValueOrigin::Output {
            producer: emulation_1_ep0_data,
            output: 0,
        } = emulation_1_ep1_data.ptr_input().origin
        else {
            panic!(
                "the ptr-input of the second element-ptr node in branch `1` should connect to \
            the first output of a node"
            )
        };

        let emulation_1_ep0_data = rvsdg[emulation_1_ep0_data].expect_op_element_ptr();

        assert_eq!(
            emulation_1_ep0_data.ptr_input(),
            &ValueInput::argument(array_of_array_ptr_ty, 0)
        );
        assert_eq!(
            emulation_1_ep0_data.index_input(),
            &ValueInput::argument(TY_U32, 1)
        );

        assert_eq!(
            &rvsdg[array_alloca_node]
                .expect_op_alloca()
                .value_output()
                .users,
            &thin_set![
                ValueUser::Input {
                    consumer: switch_node,
                    input: 1,
                },
                ValueUser::Input {
                    consumer: emulation_switch_node,
                    input: 1,
                }
            ],
            "the emulation node should have been added as a user to the alloca node"
        );

        assert!(
            rvsdg[switch_node].value_outputs()[0].users.is_empty(),
            "original switch node pointer output should no longer have any users"
        );

        assert_eq!(
            rvsdg[switch_node].value_outputs().len(),
            3,
            "two additional outputs should have been added to the original switch node"
        );
        assert_eq!(
            &rvsdg[switch_node].value_outputs()[1].users,
            &thin_set![ValueUser::Input {
                consumer: emulation_switch_node,
                input: 2
            }],
            "the first new switch output should connect to the third input of the emulation node"
        );
        assert_eq!(
            &rvsdg[switch_node].value_outputs()[2].users,
            &thin_set![ValueUser::Input {
                consumer: emulation_switch_node,
                input: 3
            }],
            "the second new switch output should connect to the fourth input of the emulation node"
        );
    }

    #[test]
    fn test_emulate_identical_switch_branches_bypass() {
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

        let ptr_0 = rvsdg.add_op_alloca(region, TY_U32);

        let switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(TY_PTR_U32, ptr_0, 0),
            ],
            vec![ValueOutput::new(TY_PTR_U32)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);
        rvsdg.reconnect_region_result(branch_0, 0, ValueOrigin::Argument(0));

        let branch_1 = rvsdg.add_switch_branch(switch_node);
        rvsdg.reconnect_region_result(branch_1, 0, ValueOrigin::Argument(0));

        let load_op = rvsdg.add_op_load(
            region,
            ValueInput::output(TY_PTR_U32, switch_node, 0),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_op,
                output: 0,
            },
        );

        let mut emulation_context = EmulationContext::new();

        emulation_context.emulate_op_load(&mut rvsdg, load_op);

        let ValueOrigin::Output {
            producer: emulation_load_node,
            output: 0,
        } = rvsdg[region].value_results()[0].origin
        else {
            panic!("the function result should connect to the first output of a node")
        };

        // It should NOT be a Switch node. It should be an OpLoad.
        let op_load_data = rvsdg[emulation_load_node].expect_op_load();

        // The pointer input of the load should be ptr_0.
        assert_eq!(
            op_load_data.ptr_input(),
            &ValueInput::output(TY_PTR_U32, ptr_0, 0)
        );
    }

    #[test]
    fn test_emulate_identical_switch_branches_with_access_bypass() {
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

        let array_ty = module.ty.register(TypeKind::Array {
            element_ty: TY_U32,
            count: 2,
            stride: 4,
        });
        let array_ptr_ty = module.ty.register(TypeKind::Ptr(array_ty));
        let ptr_0 = rvsdg.add_op_alloca(region, array_ty);

        let ptr_ty = module.ty.register(TypeKind::Ptr(TY_U32));

        let switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(array_ptr_ty, ptr_0, 0),
            ],
            vec![ValueOutput::new(ptr_ty)],
            None,
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);
        let index_0 = rvsdg.add_const_u32(branch_0, 0);
        let element_ptr_0 = rvsdg.add_op_element_ptr(
            branch_0,
            ValueInput::argument(array_ptr_ty, 0),
            ValueInput::output(TY_U32, index_0, 0),
        );
        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: element_ptr_0,
                output: 0,
            },
        );

        let branch_1 = rvsdg.add_switch_branch(switch_node);
        let index_1 = rvsdg.add_const_u32(branch_1, 0);
        let element_ptr_1 = rvsdg.add_op_element_ptr(
            branch_1,
            ValueInput::argument(array_ptr_ty, 0),
            ValueInput::output(TY_U32, index_1, 0),
        );
        rvsdg.reconnect_region_result(
            branch_1,
            0,
            ValueOrigin::Output {
                producer: element_ptr_1,
                output: 0,
            },
        );

        let load_op = rvsdg.add_op_load(
            region,
            ValueInput::output(ptr_ty, switch_node, 0),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: load_op,
                output: 0,
            },
        );

        let mut emulation_context = EmulationContext::new();

        emulation_context.emulate_op_load(&mut rvsdg, load_op);

        let ValueOrigin::Output {
            producer: emulation_load_node,
            output: 0,
        } = rvsdg[region].value_results()[0].origin
        else {
            panic!("the function result should connect to the first output of a node")
        };

        // It should NOT be a Switch node. It should be an OpLoad.
        let op_load_data = rvsdg[emulation_load_node].expect_op_load();

        // The pointer input of the load should be an ElementPtr node (emulated).
        let ValueOrigin::Output {
            producer: element_ptr_node,
            ..
        } = op_load_data.ptr_input().origin
        else {
            panic!("expected Output origin");
        };

        let element_ptr_data = rvsdg[element_ptr_node].expect_op_element_ptr();
        assert_eq!(
            element_ptr_data.ptr_input(),
            &ValueInput::output(array_ptr_ty, ptr_0, 0)
        );

        // The index input should be a constant 0
        let index_origin = element_ptr_data.index_input().origin;
        let ValueOrigin::Output {
            producer: index_producer,
            ..
        } = index_origin
        else {
            panic!("expected Output origin for index")
        };
        assert_eq!(rvsdg[index_producer].expect_const_u32().value(), 0);
    }
}
