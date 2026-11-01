//! Replaces enum-typed alloca nodes, constant dependencies, and fallback values with multiple
//! allocas, constants, or fallback values.
//!
//! Creates alloca/constant/fallback value one for the discriminant and one for each of the enum's
//! variants. We'll also refer to this as "splitting" an enum alloca or constant dependency.
//!
//! This is a key transformation for eliminating enums from the RVSDG, as enums are not supported by
//! our target shader languages. By splitting an enum into its parts, we can then replace uses of
//! the enum with uses of its parts.

use std::iter;

use internment::Intern;

use crate::rvsdg::{
    Connectivity, Node, NodeKind, Region, Rvsdg, SimpleNode, ValueInput, ValueOrigin, ValueUser,
};
use crate::ty::{Type, TypeKind, TypeRegistry};
use crate::{AllocId, Allocation, Constant, Module, ty};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstEnumReplacement {
    pub ty: Type,
    pub kind: ConstEnumReplacementKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstEnumReplacementKind {
    ConstDependency(u32),
    ConstFallback(Node),
}

#[derive(Clone, Copy)]
struct VariantInput {
    input: ValueInput,
    kind: VariantActivity,
}

/// Used to track the "activity" of a variant part.
///
/// After splitting the originating alloca/constant/fallback value, we propagate the split down
/// the value-flow as a discriminant part and a set of variant parts (one for each of the original
/// enum's variants). We attach this marker to each of the variant parts to track their activity
/// status. In particular, for variant parts that originates from a constant dependency, we can
/// statically determine which variant is the active variant, and which variant(s) is/are inactive
/// variants. For the other originating value kinds, we always treat all variants as "maybe active".
///
/// This activity information is used when "splitting" loads and stores of the whole enum value. If
/// we were to naively split such whole enum loads and stores into a separate load/store for each
/// variant part, we could introduce UB not previously present in the program: we represent the
/// inactive variant parts of a constant dependency with [ConstFallback] pointer values (null
/// pointers) and loads and stores on such pointers would be invalid. Therefore, for a load
/// operation, if the variant part is known to be inactive, then instead of adding a load of the
/// part, we add a [ConstFallback] node to represent the load's output instead. For a store
/// operation, if the variant part is known to be inactive, then we simply elide the store
/// operation.
///
/// Where multiple value-flow path merge (switch outputs and loop-values that are not
/// loop-invariant), we always conservatively mark the part as "maybe active".
///
/// [ConstFallback]: crate::rvsdg::ConstFallback
#[derive(Clone, Copy, PartialEq, Eq)]
enum VariantActivity {
    MaybeActive,
    Inactive,
}

type FallbackCallback<'a> = dyn FnMut(Node, Type) + 'a;

impl VariantInput {
    fn maybe_active(input: ValueInput) -> Self {
        Self {
            input,
            kind: VariantActivity::MaybeActive,
        }
    }

    fn inactive(input: ValueInput) -> Self {
        Self {
            input,
            kind: VariantActivity::Inactive,
        }
    }

    fn remap(self, origin: ValueOrigin) -> Self {
        Self {
            input: ValueInput {
                ty: self.input.ty,
                origin,
            },
            ..self
        }
    }

    fn remap_maybe_active(self, origin: ValueOrigin) -> Self {
        Self::maybe_active(ValueInput {
            ty: self.input.ty,
            origin,
        })
    }
}

pub struct EnumReplacer<'a> {
    rvsdg: &'a mut Rvsdg,
    ty: TypeRegistry,
    enum_ty: Type,
}

impl<'a> EnumReplacer<'a> {
    pub fn new(rvsdg: &'a mut Rvsdg, enum_ty: Type) -> Self {
        let ty_registry = rvsdg.ty().clone();

        assert!(ty_registry.kind(enum_ty).is_enum());

        EnumReplacer {
            ty: rvsdg.ty().clone(),
            rvsdg,
            enum_ty,
        }
    }

    pub fn replace_alloca<F>(&mut self, node: Node, mut with_replacements: F)
    where
        F: FnMut(Node, Type),
    {
        let node_data = &self.rvsdg[node];
        let region = node_data.region();
        let discr_ty = self
            .ty
            .kind(self.enum_ty)
            .expect_enum()
            .discriminant_ty
            .backend_ty();
        let discr_ptr_ty = self.ty.register(TypeKind::Ptr(discr_ty));
        let discriminant_node = self.rvsdg.add_op_alloca(region, discr_ty);

        let discriminant_input = ValueInput {
            ty: discr_ptr_ty,
            origin: ValueOrigin::Output {
                producer: discriminant_node,
                output: 0,
            },
        };

        let ty_kind = self.ty.kind(self.enum_ty);

        let variant_inputs = ty_kind
            .expect_enum()
            .variants
            .iter()
            .copied()
            .map(|variant| {
                let ptr_ty = self.ty.register(TypeKind::Ptr(variant.ty));
                let variant_node = self.rvsdg.add_op_alloca(region, variant.ty);

                with_replacements(variant_node, variant.ty);

                VariantInput::maybe_active(ValueInput {
                    ty: ptr_ty,
                    origin: ValueOrigin::Output {
                        producer: variant_node,
                        output: 0,
                    },
                })
            })
            .collect::<Vec<_>>();

        let mut report_fallback = |_: Node, _: Type| {
            unreachable!("alloca replacement cannot create inactive payload fallbacks")
        };

        self.visit_users(
            node,
            0,
            discriminant_input,
            &variant_inputs,
            &mut report_fallback,
        );

        // The OpAlloca node now should not have any users left, so we can remove it
        self.rvsdg.remove_node(node);
    }

    pub fn replace_const_fallback<F>(&mut self, node: Node, mut with_replacements: F)
    where
        F: FnMut(Node, Type),
    {
        let node_data = &self.rvsdg[node];
        let region = node_data.region();
        let fallback_ty = node_data.expect_const_fallback().ty();
        let (enum_ty, is_pointer) = match *self.ty.kind(fallback_ty) {
            TypeKind::Ptr(pointee_ty) => (pointee_ty, true),
            _ => (fallback_ty, false),
        };
        let enum_kind = self.ty.kind(enum_ty);
        let enum_data = enum_kind.expect_enum();
        let discriminant_ty = enum_data.discriminant_ty.backend_ty();

        let discriminant_ty = if is_pointer {
            self.ty.register(TypeKind::Ptr(discriminant_ty))
        } else {
            discriminant_ty
        };
        let discriminant_node = self.rvsdg.add_const_fallback(region, discriminant_ty);

        let discriminant_input = ValueInput::output(discriminant_ty, discriminant_node, 0);
        let mut variant_inputs = Vec::with_capacity(enum_data.variants.len());

        for variant in &enum_data.variants {
            let variant_ty = if is_pointer {
                self.ty.register(TypeKind::Ptr(variant.ty))
            } else {
                variant.ty
            };

            let variant_node = self.rvsdg.add_const_fallback(region, variant_ty);

            variant_inputs.push(VariantInput::maybe_active(ValueInput::output(
                variant_ty,
                variant_node,
                0,
            )));

            with_replacements(variant_node, variant_ty);
        }

        self.visit_users(
            node,
            0,
            discriminant_input,
            &variant_inputs,
            &mut with_replacements,
        );
        self.rvsdg.remove_node(node);
    }

    pub fn replace_constant_dependency<F>(
        &mut self,
        module: &mut Module,
        function_node: Node,
        dep_index: u32,
        constant: Constant,
        alloc_id: AllocId,
        offset: usize,
        mut with_replacements: F,
    ) where
        F: FnMut(ConstEnumReplacement),
    {
        let fn_data = self.rvsdg[function_node].expect_function();
        let region = fn_data.body_region();

        let ty_kind = self.ty.kind(self.enum_ty);
        let enum_data = ty_kind.expect_enum();

        let discriminant =
            enum_data.read_discriminant(&module.allocations[alloc_id].bytes[offset..]);
        let active_variant_index = enum_data
            .variant_index_for(discriminant)
            .expect("a valid discriminant should map to a variant-index");
        let active_variant = &enum_data.variants[active_variant_index];

        // Resolve a new discriminant constant node
        let discr_name = format!("{}.discriminant", constant.name);
        let discr_ty = discriminant.backend_ty();

        let discr_constant = Constant {
            name: Intern::new(discr_name),
            module: constant.module,
        };

        if !module.constants.contains(discr_constant) {
            let discr_bytes = discriminant.backend_bytes();

            let discr_alloc_id = module
                .allocations
                .register(Allocation { bytes: discr_bytes });

            module
                .constants
                .register_byte_data(discr_constant, discr_ty, discr_alloc_id, 0);
        }

        let discr_node = self.rvsdg.register_constant(module, discr_constant);

        // Resolve a constant node for the active variant
        let variant_name = format!("{}.{}", constant.name, active_variant_index);
        let variant_constant = Constant {
            name: Intern::new(variant_name),
            module: constant.module,
        };

        if !module.constants.contains(variant_constant) {
            module.constants.register_byte_data(
                variant_constant,
                active_variant.ty,
                alloc_id,
                offset,
            );
        }
        let variant_node = self.rvsdg.register_constant(module, variant_constant);

        let discriminant_ptr_ty = self.ty.register(TypeKind::Ptr(discr_ty));
        let discr_dep_index = self
            .rvsdg
            .add_function_dependency(function_node, discr_node);

        let discriminant_input = ValueInput::argument(discriminant_ptr_ty, discr_dep_index);
        let mut variant_inputs = Vec::with_capacity(enum_data.variants.len());

        for i in 0..enum_data.variants.len() {
            let variant_ty = enum_data.variants[i].ty;
            let variant_ptr_ty = self.ty.register(TypeKind::Ptr(variant_ty));

            if i == active_variant_index {
                let var_dep_index = self
                    .rvsdg
                    .add_function_dependency(function_node, variant_node);

                variant_inputs.push(VariantInput::maybe_active(ValueInput::argument(
                    variant_ptr_ty,
                    var_dep_index,
                )));

                with_replacements(ConstEnumReplacement {
                    ty: variant_ty,
                    kind: ConstEnumReplacementKind::ConstDependency(var_dep_index),
                });
            } else {
                let fallback_node = self.rvsdg.add_const_fallback(region, variant_ptr_ty);

                variant_inputs.push(VariantInput::inactive(ValueInput::output(
                    variant_ptr_ty,
                    fallback_node,
                    0,
                )));

                with_replacements(ConstEnumReplacement {
                    ty: variant_ptr_ty,
                    kind: ConstEnumReplacementKind::ConstFallback(fallback_node),
                });
            }
        }

        let mut report_fallback = |node, ty| {
            with_replacements(ConstEnumReplacement {
                ty,
                kind: ConstEnumReplacementKind::ConstFallback(node),
            });
        };

        self.redirect_region_argument(
            region,
            dep_index,
            discriminant_input,
            &variant_inputs,
            &mut report_fallback,
        );
    }

    fn visit_users(
        &mut self,
        node: Node,
        output: u32,
        discriminant_input: ValueInput,
        variant_inputs: &[VariantInput],
        with_replacements: &mut FallbackCallback<'_>,
    ) {
        let region = self.rvsdg[node].region();
        let user_count = self.rvsdg[node].value_outputs()[output as usize]
            .users
            .len();

        // We iterate over users in reverse order, so that users may more themselves from the user
        // set, without disrupting iteration
        for i in (0..user_count).rev() {
            let user = self.rvsdg[node].value_outputs()[output as usize].users[i];

            self.visit_user(
                region,
                user,
                discriminant_input,
                variant_inputs,
                with_replacements,
            );
        }
    }

    fn visit_user(
        &mut self,
        region: Region,
        user: ValueUser,
        discriminant_input: ValueInput,
        variant_inputs: &[VariantInput],
        with_replacements: &mut FallbackCallback<'_>,
    ) {
        match user {
            ValueUser::Result(result) => self.split_region_result(
                region,
                result,
                discriminant_input,
                variant_inputs,
                with_replacements,
            ),
            ValueUser::Input { consumer, input } => self.visit_node_input(
                consumer,
                input,
                discriminant_input,
                variant_inputs,
                with_replacements,
            ),
        }
    }

    fn split_region_result(
        &mut self,
        region: Region,
        result: u32,
        discriminant_input: ValueInput,
        variant_inputs: &[VariantInput],
        with_replacements: &mut FallbackCallback<'_>,
    ) {
        let owner = self.rvsdg[region].owner();

        match self.rvsdg[owner].kind() {
            NodeKind::Switch(_) => self.split_switch_result(
                region,
                result,
                discriminant_input,
                variant_inputs,
                with_replacements,
            ),
            NodeKind::Loop(_) => self.split_loop_result(
                region,
                result,
                discriminant_input,
                variant_inputs,
                with_replacements,
            ),
            NodeKind::Function(_) => panic!(
                "cannot split function result; \
            non-local-use analyses should have rejected the alloca or constant"
            ),
            _ => unreachable!("node kind cannot by a region owner"),
        }
    }

    fn visit_node_input(
        &mut self,
        node: Node,
        input: u32,
        discriminant_input: ValueInput,
        variant_inputs: &[VariantInput],
        with_replacements: &mut FallbackCallback<'_>,
    ) {
        use NodeKind::*;
        use SimpleNode::*;

        match self.rvsdg[node].kind() {
            Switch(_) => self.split_switch_input(
                node,
                input,
                discriminant_input,
                variant_inputs,
                with_replacements,
            ),
            Loop(_) => self.split_loop_input(
                node,
                input,
                discriminant_input,
                variant_inputs,
                with_replacements,
            ),
            Simple(OpLoad(_)) => {
                self.split_op_load(node, discriminant_input, variant_inputs, with_replacements)
            }
            Simple(OpStore(_)) => {
                self.split_op_store(node, input, discriminant_input, variant_inputs)
            }
            Simple(OpGetDiscriminant(_)) => {
                self.replace_op_get_discriminant(node, discriminant_input)
            }
            Simple(OpSetDiscriminant(_)) => {
                self.replace_op_set_discriminant(node, discriminant_input)
            }
            Simple(OpDiscriminantPtr(_)) => {
                self.elide_op_discriminant_ptr(node, discriminant_input)
            }
            Simple(OpVariantPtr(_)) => self.elide_op_variant_ptr(node, variant_inputs),
            Simple(ValueProxy(_)) => {
                self.visit_value_proxy(node, discriminant_input, variant_inputs, with_replacements)
            }
            _ => unreachable!("node kind cannot take a pointer to or a copy of an enum as input"),
        }
    }

    fn elide_op_discriminant_ptr(&mut self, node: Node, discriminant_input: ValueInput) {
        let node_data = &self.rvsdg[node];
        let region = node_data.region();
        let node_data = node_data.expect_op_discriminant_ptr();
        let new_user_origin = discriminant_input.origin;
        let user_count = node_data.value_output().users.len();

        for i in (0..user_count).rev() {
            let user = self.rvsdg[node]
                .expect_op_discriminant_ptr()
                .value_output()
                .users[i];

            self.rvsdg
                .reconnect_value_user(region, user, new_user_origin);
        }

        // We've reconnected all the node's users now. Consequently, it's now dead and can
        // be removed.
        self.rvsdg.remove_node(node);
    }

    fn elide_op_variant_ptr(&mut self, node: Node, variant_inputs: &[VariantInput]) {
        let node_data = &self.rvsdg[node];
        let region = node_data.region();
        let node_data = node_data.expect_op_variant_ptr();
        let variant_index = node_data.variant_index();
        let new_user_origin = variant_inputs[variant_index as usize].input.origin;
        let user_count = node_data.value_output().users.len();

        for i in (0..user_count).rev() {
            let user = self.rvsdg[node]
                .expect_op_variant_ptr()
                .value_output()
                .users[i];

            self.rvsdg
                .reconnect_value_user(region, user, new_user_origin);
        }

        // We've reconnected all the node's users now. Consequently, it's now dead and can
        // be removed.
        self.rvsdg.remove_node(node);
    }

    fn replace_op_get_discriminant(&mut self, node: Node, discriminant_input: ValueInput) {
        let region = self.rvsdg[node].region();
        let state_origin = self.rvsdg[node]
            .state()
            .expect("get-discriminant operation should part of state chain")
            .origin;
        let user_count = self.rvsdg[node]
            .expect_op_get_discriminant()
            .value_output()
            .users
            .len();

        let load_node = self
            .rvsdg
            .add_op_load(region, discriminant_input, state_origin);

        for i in (0..user_count).rev() {
            let user = self.rvsdg[node]
                .expect_op_get_discriminant()
                .value_output()
                .users[i];

            self.rvsdg.reconnect_value_user(
                region,
                user,
                ValueOrigin::Output {
                    producer: load_node,
                    output: 0,
                },
            );
        }

        // After visiting the users of the original node's output, all users should have been
        // reconnected to the new load node and the original node should have no users left; we
        // should be able to remove the node now.
        self.rvsdg.remove_node(node);
    }

    fn replace_op_set_discriminant(&mut self, node: Node, discriminant_input: ValueInput) {
        let region = self.rvsdg[node].region();
        let state_origin = self.rvsdg[node]
            .state()
            .expect("set-discriminant operation should part of state chain")
            .origin;
        let node_data = self.rvsdg[node].expect_op_set_discriminant();
        let variant_index = node_data.variant_index();

        let enum_kind = self.ty.kind(self.enum_ty);
        let enum_data = enum_kind.expect_enum();
        let discr_ty = enum_data.discriminant_ty.backend_ty();
        let discriminant = enum_data.variants[variant_index as usize].discriminant;

        let discr_node = match discr_ty {
            ty::TY_U32 => self.rvsdg.add_const_u32(region, discriminant as u32),
            ty::TY_I32 => self.rvsdg.add_const_i32(region, discriminant as i32),
            _ => panic!(
                "unsupported backend type for enum discriminant: {:?}",
                discr_ty
            ),
        };

        self.rvsdg.add_op_store(
            region,
            discriminant_input,
            ValueInput::output(discr_ty, discr_node, 0),
            state_origin,
        );
        self.rvsdg.remove_node(node);
    }

    fn split_op_load(
        &mut self,
        node: Node,
        discriminant_input: ValueInput,
        variant_inputs: &[VariantInput],
        with_replacements: &mut FallbackCallback<'_>,
    ) {
        let region = self.rvsdg[node].region();
        let state_origin = self.rvsdg[node]
            .state()
            .expect("load operation should part of state chain")
            .origin;

        // The OpLoad nodes we produce will have to be linked into the state chain. Though the
        // order in which this happens does not matter semantically to the program, for inspection
        // of the RVSDG it is more intuitive when this matches the input order. Because of the way
        // successively inserting a node with the same state origin behaves, we reverse the
        // iteration order to achieve this.
        let mut split_variant_outputs = variant_inputs
            .iter()
            .rev()
            .map(|variant_input| {
                let input = variant_input.input;
                let TypeKind::Ptr(payload_ty) = *self.rvsdg.ty().kind(input.ty) else {
                    panic!("expected input to load operation to be a pointer");
                };

                match variant_input.kind {
                    VariantActivity::MaybeActive => {
                        let split_node = self.rvsdg.add_op_load(region, input, state_origin);

                        VariantInput::maybe_active(ValueInput::output(payload_ty, split_node, 0))
                    }
                    VariantActivity::Inactive => {
                        let fallback = self.rvsdg.add_const_fallback(region, payload_ty);

                        with_replacements(fallback, payload_ty);

                        VariantInput::inactive(ValueInput::output(payload_ty, fallback, 0))
                    }
                }
            })
            .collect::<Vec<_>>();

        // We reversed the iteration order when we produced this list. Though again the order of
        // these does not matter semantically to the program, for inspection of the RVSDG it is
        // more intuitive to maintain a consistent order, so we unreverse the order here.
        split_variant_outputs.reverse();

        let TypeKind::Ptr(discriminant_ty) = *self.rvsdg.ty().kind(discriminant_input.ty) else {
            panic!("expected discriminant input to load operation to be a pointer");
        };
        let discriminant_node = self
            .rvsdg
            .add_op_load(region, discriminant_input, state_origin);
        let discriminant_output = ValueInput::output(discriminant_ty, discriminant_node, 0);

        self.visit_users(
            node,
            0,
            discriminant_output,
            &split_variant_outputs,
            with_replacements,
        );
        self.rvsdg.remove_node(node);
    }

    fn split_op_store(
        &mut self,
        node: Node,
        input: u32,
        discriminant_input: ValueInput,
        variant_inputs: &[VariantInput],
    ) {
        assert!(input < 2, "OpStore only has 2 value inputs");

        let node_data = &self.rvsdg[node];
        let region = node_data.region();
        let state_origin = node_data
            .state()
            .expect("store operation should be part of the state-chain")
            .origin;
        let node_data = node_data.expect_op_store();
        let ptr_input = *node_data.ptr_input();

        if input == 0 {
            // If we reach the OpStore node via its pointer-input, then we defer the splitting
            // until it gets reached via its value-input later. This is because SLIR does not
            // expose nodes that allow us to extract the parts of an enum value that is not behind
            // a pointer.
            return self.defer_split_op_store(
                region,
                node,
                ptr_input,
                discriminant_input,
                variant_inputs,
            );
        }

        // We've reached this OpStore via its value-input. In this case we can proceed with actually
        // splitting the OpStore node.

        let (discriminant_ptr, variant_ptrs) =
            self.split_op_store_ptr_input(region, node, ptr_input);

        assert_eq!(
            variant_ptrs.len(),
            variant_inputs.len(),
            "split pointer/value arity should match when splitting OpStore"
        );

        // Reverse iteration keeps state-chain order intuitive.
        for (pointer, value) in variant_ptrs.iter().zip(variant_inputs).rev() {
            // We only add store operations for active variants; storing inactive variants would be
            // UB.
            if value.kind == VariantActivity::MaybeActive {
                self.rvsdg
                    .add_op_store(region, pointer.input, value.input, state_origin);
            }
        }

        self.rvsdg
            .add_op_store(region, discriminant_ptr, discriminant_input, state_origin);

        self.rvsdg.remove_node(node);
    }

    fn defer_split_op_store(
        &mut self,
        region: Region,
        node: Node,
        ptr_input: ValueInput,
        discriminant_input: ValueInput,
        variant_inputs: &[VariantInput],
    ) {
        let synthetic_original_node = self.rvsdg.add_const_fallback(region, ptr_input.ty);
        let synthetic_original = ValueInput {
            ty: ptr_input.ty,
            origin: ValueOrigin::Output {
                producer: synthetic_original_node,
                output: 0,
            },
        };
        let value_inputs =
            iter::once(discriminant_input).chain(variant_inputs.iter().map(|input| input.input));
        let reaggregation = self
            .rvsdg
            .add_reaggregation(region, synthetic_original, value_inputs);

        self.rvsdg.reconnect_value_input(
            node,
            0,
            ValueOrigin::Output {
                producer: reaggregation,
                output: 0,
            },
        );
    }

    fn reaggregation_from_origin(&self, origin: ValueOrigin) -> Option<Node> {
        let ValueOrigin::Output {
            producer,
            output: 0,
        } = origin
        else {
            return None;
        };

        if self.rvsdg[producer].is_reaggregation() {
            return Some(producer);
        }

        if self.rvsdg[producer].is_value_proxy() {
            let proxied_origin = self.rvsdg[producer].expect_value_proxy().input().origin;

            if let ValueOrigin::Output {
                producer: reaggregation,
                output: 0,
            } = proxied_origin
                && self.rvsdg[reaggregation].is_reaggregation()
            {
                return Some(reaggregation);
            }
        }

        None
    }

    fn split_op_store_ptr_input(
        &mut self,
        region: Region,
        store_node: Node,
        mut ptr_input: ValueInput,
    ) -> (ValueInput, Vec<VariantInput>) {
        if let Some(reaggregation_node) = self.reaggregation_from_origin(ptr_input.origin) {
            if let ValueOrigin::Output {
                producer: proxy,
                output: 0,
            } = ptr_input.origin
                && self.rvsdg[proxy].is_value_proxy()
            {
                self.rvsdg.dissolve_value_proxy(proxy);
            }

            let parts = self.rvsdg[reaggregation_node]
                .expect_reaggregation()
                .parts()
                .to_vec();
            self.rvsdg.dissolve_reaggregation(reaggregation_node);

            let (discriminant_input, variant_inputs) = parts
                .split_first()
                .expect("reaggregated enum pointers include a discriminant pointer");

            return (
                *discriminant_input,
                variant_inputs
                    .iter()
                    .copied()
                    .map(VariantInput::maybe_active)
                    .collect(),
            );
        }

        // Defensively proxy the pointer origin, as we don't want to add more users to a user-set
        // that is potentially being iterated over. We instead add the new users to the new proxy
        // output.
        let proxy = self.rvsdg.proxy_origin_user(
            region,
            ptr_input.ty,
            ptr_input.origin,
            ValueUser::Input {
                consumer: store_node,
                input: 0,
            },
        );

        ptr_input.origin = ValueOrigin::Output {
            producer: proxy,
            output: 0,
        };

        let discriminant_ptr = self.rvsdg.add_op_discriminant_ptr(region, ptr_input);
        let discr_ptr_ty = self.rvsdg[discriminant_ptr]
            .expect_op_discriminant_ptr()
            .value_output()
            .ty;

        let discriminant_input = ValueInput::output(discr_ptr_ty, discriminant_ptr, 0);

        let ty_kind = self.ty.kind(self.enum_ty);
        let variant_count = ty_kind.expect_enum().variants.len();
        let mut variant_inputs = Vec::with_capacity(variant_count);

        for i in 0..variant_count {
            let variant_ptr = self.rvsdg.add_op_variant_ptr(region, ptr_input, i as u32);
            let variant_ptr_ty = self.rvsdg[variant_ptr]
                .expect_op_variant_ptr()
                .value_output()
                .ty;

            variant_inputs.push(VariantInput::maybe_active(ValueInput::output(
                variant_ptr_ty,
                variant_ptr,
                0,
            )));
        }

        (discriminant_input, variant_inputs)
    }

    fn split_switch_input(
        &mut self,
        node: Node,
        input: u32,
        discriminant_input: ValueInput,
        variant_inputs: &[VariantInput],
        with_replacements: &mut FallbackCallback<'_>,
    ) {
        assert_ne!(input, 0, "the branch selector input is never an aggregate");

        let arg_index = input as usize - 1;
        let node_data = self.rvsdg[node].expect_switch();
        let branch_count = node_data.branches().len();
        let split_args_base = node_data.value_inputs().len() - 1;

        self.rvsdg.add_switch_input(node, discriminant_input);

        let split_discriminant_arg =
            ValueInput::argument(discriminant_input.ty, split_args_base as u32);
        let split_variant_args = variant_inputs
            .iter()
            .copied()
            .enumerate()
            .map(|(i, variant_input)| {
                self.rvsdg.add_switch_input(node, variant_input.input);

                variant_input.remap(ValueOrigin::Argument((split_args_base + i + 1) as u32))
            })
            .collect::<Vec<_>>();

        for branch_index in 0..branch_count {
            let branch = self.rvsdg[node].expect_switch().branches()[branch_index];

            self.redirect_region_argument(
                branch,
                arg_index as u32,
                split_discriminant_arg,
                &split_variant_args,
                with_replacements,
            );
        }

        self.rvsdg.remove_switch_input(node, input);
    }

    fn split_switch_result(
        &mut self,
        branch: Region,
        result: u32,
        discriminant_input: ValueInput,
        variant_inputs: &[VariantInput],
        with_replacements: &mut FallbackCallback<'_>,
    ) {
        let node = self.rvsdg[branch].owner();
        let node_data = self.rvsdg[node].expect_switch();
        let branch_count = node_data.branches().len();
        let base_index = node_data.value_outputs().len();

        let mut split_outputs = Vec::with_capacity(variant_inputs.len());

        // First split the output/results and connect the new results for the provoking branch. Also
        // record a mapping for the split output

        self.rvsdg.add_switch_output(node, discriminant_input.ty);
        self.rvsdg
            .reconnect_region_result(branch, base_index as u32, discriminant_input.origin);
        let split_discriminant_output =
            ValueInput::output(discriminant_input.ty, node, base_index as u32);

        for (i, variant_input) in variant_inputs.iter().enumerate() {
            let input = variant_input.input;
            let index = base_index + i + 1;

            self.rvsdg.add_switch_output(node, input.ty);
            self.rvsdg
                .reconnect_region_result(branch, index as u32, input.origin);

            // We currently don't do any analysis on which variants are active for each of the
            // output's corresponding branch result values; we conservatively assume every variant
            // may be active.
            split_outputs.push(variant_input.remap_maybe_active(ValueOrigin::Output {
                producer: node,
                output: index as u32,
            }));
        }

        // Now reconnect the results for the other branches
        for i in 0..branch_count {
            let current_branch = self.rvsdg[node].expect_switch().branches()[i];

            if current_branch != branch {
                self.redirect_region_result(current_branch, result as usize, base_index);
            }
        }

        // Finally, split the output
        self.visit_users(
            node,
            result,
            split_discriminant_output,
            &split_outputs,
            with_replacements,
        );

        // Disconnect the original region result for the provoking branch, as its users have been
        // replaced by the split results
        self.rvsdg.disconnect_region_result(branch, result);
    }

    fn split_loop_input(
        &mut self,
        node: Node,
        input: u32,
        discriminant_input: ValueInput,
        variant_inputs: &[VariantInput],
        with_replacements: &mut FallbackCallback<'_>,
    ) {
        let node_data = self.rvsdg[node].expect_loop();
        let loop_region = node_data.loop_region();
        let prior_input_count = node_data.value_inputs().len();
        let prior_result_count = prior_input_count + 1;
        let is_loop_invariant = self.rvsdg[loop_region].value_results()[input as usize + 1].origin
            == ValueOrigin::Argument(input);

        let mut split_variant_args = Vec::with_capacity(variant_inputs.len());
        let mut split_variant_outputs = Vec::with_capacity(variant_inputs.len());

        self.rvsdg.add_loop_input(node, discriminant_input);

        let split_discriminant_arg =
            ValueInput::argument(discriminant_input.ty, prior_input_count as u32);
        let split_discriminant_output =
            ValueInput::output(discriminant_input.ty, node, prior_input_count as u32);

        for (i, variant_input) in variant_inputs.iter().enumerate() {
            self.rvsdg.add_loop_input(node, variant_input.input);

            let argument_origin = ValueOrigin::Argument((prior_input_count + i + 1) as u32);
            let output_origin = ValueOrigin::Output {
                producer: node,
                output: (prior_input_count + i + 1) as u32,
            };

            // If the loop-value is loop-invariant, we'll preserve the maybe-active/inactive marking
            // of the split variant value. If the loop-value is not loop-invariant, then we don't
            // attempt any analysis on whether repeated loop re-entry would preserve the input
            // value's status. Instead, we conservatively assume the variant may be active.
            if is_loop_invariant {
                split_variant_args.push(variant_input.remap(argument_origin));
                split_variant_outputs.push(variant_input.remap(output_origin));
            } else {
                split_variant_args.push(variant_input.remap_maybe_active(argument_origin));
                split_variant_outputs.push(variant_input.remap_maybe_active(output_origin));
            }
        }

        // First connect all region results that we've created to the unsplit input that the
        // original result connects to, via [OpGetDiscriminant] nodes or [OpPtrVariantPtr] nodes.
        // This also disconnects the original result.
        self.redirect_region_result(loop_region, input as usize + 1, prior_result_count);

        // Now redirect the argument using the argument mapping. We do this after redirecting the
        // results, because otherwise this might try to split the original result again; by doing
        // this after result redirection, all the argument's user tree should terminate at the
        // [OpGetDiscriminant]/[OpPtrVariantPtr] nodes that were inserted, since nothing should
        // be connected to the original result anymore.
        self.redirect_region_argument(
            loop_region,
            input,
            split_discriminant_arg,
            &split_variant_args,
            with_replacements,
        );

        // Finally, redirect the value output using the output mapping
        self.visit_users(
            node,
            input,
            split_discriminant_output,
            &split_variant_outputs,
            with_replacements,
        );

        // Now neither the argument nor the output should have any remaining users, so we can remove
        // the original input/output/argument/result and disconnect the input from its origin.
        self.rvsdg.disconnect_region_result(loop_region, input + 1);
        self.rvsdg.remove_loop_input(node, input);
    }

    fn split_loop_result(
        &mut self,
        region: Region,
        result: u32,
        discriminant_input: ValueInput,
        variant_inputs: &[VariantInput],
        with_replacements: &mut FallbackCallback<'_>,
    ) {
        assert_ne!(
            result, 0,
            "the reentry decider result is never a pointer to an enum"
        );

        let owner = self.rvsdg[region].owner();
        let outer_region = self.rvsdg[owner].region();
        let loop_data = self.rvsdg[owner].expect_loop();
        let prior_input_count = loop_data.value_inputs().len();
        let prior_result_count = prior_input_count + 1;

        let input_index = result - 1;
        let value_input = loop_data.value_inputs()[input_index as usize];
        let is_loop_invariant = self.rvsdg[region].value_results()[result as usize].origin
            == ValueOrigin::Argument(input_index);

        // A loop node's input and outputs must match, so splitting a result also means splitting
        // the corresponding input. However, propagating splits "upwards" runs the risk of
        // concurrently modifying a user set that is currently being traversed. To sidestep this,
        // we insert a proxy between the input and its origin, so that our modification will modify
        // the proxy's user set, which we know is not currently being traversed.
        let proxy = self.rvsdg.proxy_origin_user(
            outer_region,
            value_input.ty,
            value_input.origin,
            ValueUser::Input {
                consumer: owner,
                input: input_index,
            },
        );
        let proxy_input = ValueInput {
            ty: value_input.ty,
            origin: ValueOrigin::Output {
                producer: proxy,
                output: 0,
            },
        };

        let mut split_variant_args = Vec::with_capacity(variant_inputs.len());
        let mut split_variant_outputs = Vec::with_capacity(variant_inputs.len());

        // Add an input for the discriminant and for each variant and connect them to the proxy via
        // an OpPtrDiscriminantPtr node for the discriminant input or an OpPtrVariantPtr node for
        // the variant inputs. Also record an argument mapping in `split_args` and an output mapping
        // in `split_outputs`.

        let discriminant_node = self
            .rvsdg
            .add_op_discriminant_ptr(outer_region, proxy_input);
        let discr_ptr_ty = self.rvsdg[discriminant_node]
            .expect_op_discriminant_ptr()
            .value_output()
            .ty;

        self.rvsdg.add_loop_input(
            owner,
            ValueInput::output(discr_ptr_ty, discriminant_node, 0),
        );

        let split_discriminant_arg = ValueInput::argument(discr_ptr_ty, prior_input_count as u32);
        let split_discriminant_output =
            ValueInput::output(discr_ptr_ty, owner, prior_input_count as u32);

        let ty_kind = self.ty.kind(self.enum_ty);
        let variant_count = ty_kind.expect_enum().variants.len();

        for i in 0..variant_count {
            let input_index = (prior_input_count + 1 + i) as u32;
            let variant_node = self
                .rvsdg
                .add_op_variant_ptr(outer_region, proxy_input, i as u32);
            let variant_ptr_ty = self.rvsdg[variant_node]
                .expect_op_variant_ptr()
                .value_output()
                .ty;

            self.rvsdg
                .add_loop_input(owner, ValueInput::output(variant_ptr_ty, variant_node, 0));

            let argument_origin = ValueOrigin::Argument(input_index);
            let output_origin = ValueOrigin::Output {
                producer: owner,
                output: input_index,
            };

            // If the loop-value is loop-invariant, we'll preserve the maybe-active/inactive marking
            // of the split variant value. If the loop-value is not loop-invariant, then we don't
            // attempt any analysis on whether repeated loop re-entry would preserve the input
            // value's status. Instead, we conservatively assume the variant may be active.
            if is_loop_invariant {
                split_variant_args.push(variant_inputs[i].remap(argument_origin));
                split_variant_outputs.push(variant_inputs[i].remap(output_origin));
            } else {
                split_variant_args.push(variant_inputs[i].remap_maybe_active(argument_origin));
                split_variant_outputs.push(variant_inputs[i].remap_maybe_active(output_origin));
            }
        }

        // Reconnect the results we just created to the split inputs.
        self.rvsdg.reconnect_region_result(
            region,
            prior_result_count as u32,
            discriminant_input.origin,
        );
        for (i, variant_input) in variant_inputs.iter().enumerate() {
            let result_index = prior_result_count + i;

            self.rvsdg.reconnect_region_result(
                region,
                (result_index + 1) as u32,
                variant_input.input.origin,
            );
        }

        // Redirect the argument using the argument mapping
        self.redirect_region_argument(
            region,
            input_index,
            split_discriminant_arg,
            &split_variant_args,
            with_replacements,
        );

        // Redirect the value output using the output mapping
        self.visit_users(
            owner,
            input_index,
            split_discriminant_output,
            &split_variant_outputs,
            with_replacements,
        );

        // Now neither the argument nor the output should have any remaining users, so we can remove
        // the original input/output/argument/result and disconnect the input from its origin. We
        // don't have to worry about the input's removal affecting the user traversal of an upstream
        // node, because we proxied the input earlier.
        self.rvsdg.remove_loop_input(owner, input_index);
    }

    fn visit_value_proxy(
        &mut self,
        node: Node,
        discriminant_input: ValueInput,
        variant_inputs: &[VariantInput],
        with_replacements: &mut FallbackCallback<'_>,
    ) {
        self.visit_users(
            node,
            0,
            discriminant_input,
            variant_inputs,
            with_replacements,
        );
        self.rvsdg.remove_node(node);
    }

    /// Redirects all users of the `region`'s given `argument` to the `split_input` nodes.
    ///
    /// Leaves the `argument` without any users.
    fn redirect_region_argument(
        &mut self,
        region: Region,
        argument: u32,
        discriminant_input: ValueInput,
        variant_inputs: &[VariantInput],
        with_replacements: &mut FallbackCallback<'_>,
    ) {
        let arg_index = argument as usize;
        let user_count = self.rvsdg[region].value_arguments()[arg_index].users.len();

        // We iterate over users in reverse order, so that users may more themselves from the user
        // set, without disrupting iteration
        for user_index in (0..user_count).rev() {
            let user = self.rvsdg[region].value_arguments()[arg_index].users[user_index];

            self.visit_user(
                region,
                user,
                discriminant_input,
                variant_inputs,
                with_replacements,
            )
        }
    }

    /// Redirects the origin for the `region`'s given `result` to a set of "split" results that
    /// start at `split_results_start`, via either [OpPtrDiscriminant] for the discriminant or
    /// [OpPtrVariantPtr] nodes for the variants.
    ///
    /// Leaves the original result connected to the "placeholder" origin.
    pub fn redirect_region_result(
        &mut self,
        region: Region,
        original: usize,
        split_results_start: usize,
    ) {
        let original_input = self.rvsdg[region].value_results()[original];
        let discriminant_node = self.rvsdg.add_op_discriminant_ptr(region, original_input);

        self.rvsdg.reconnect_region_result(
            region,
            split_results_start as u32,
            ValueOrigin::Output {
                producer: discriminant_node,
                output: 0,
            },
        );

        let ty_kind = self.ty.kind(self.enum_ty);
        let variant_count = ty_kind.expect_enum().variants.len();

        for i in 0..variant_count {
            let result_index = (split_results_start + 1 + i) as u32;
            let variant_node = self
                .rvsdg
                .add_op_variant_ptr(region, original_input, i as u32);

            self.rvsdg.reconnect_region_result(
                region,
                result_index,
                ValueOrigin::Output {
                    producer: variant_node,
                    output: 0,
                },
            );
        }

        self.rvsdg.disconnect_region_result(region, original as u32);
    }
}

pub fn replace_enum_alloca<F>(rvsdg: &mut Rvsdg, node: Node, with_replacements: F)
where
    F: FnMut(Node, Type),
{
    let enum_ty = rvsdg[node].expect_op_alloca().ty();
    let mut replacer = EnumReplacer::new(rvsdg, enum_ty);

    replacer.replace_alloca(node, with_replacements);
}

pub fn replace_enum_constant_dependency<F>(
    rvsdg: &mut Rvsdg,
    module: &mut Module,
    function_node: Node,
    dep_index: u32,
    constant: Constant,
    alloc_id: AllocId,
    offset: usize,
    with_replacements: F,
) where
    F: FnMut(ConstEnumReplacement),
{
    let enum_ty = module.constants[constant].ty();
    let mut replacer = EnumReplacer::new(rvsdg, enum_ty);

    replacer.replace_constant_dependency(
        module,
        function_node,
        dep_index,
        constant,
        alloc_id,
        offset,
        with_replacements,
    );
}

pub fn replace_enum_const_fallback<F>(rvsdg: &mut Rvsdg, node: Node, with_replacements: F)
where
    F: FnMut(Node, Type),
{
    let fallback_ty = rvsdg[node].expect_const_fallback().ty();
    let enum_ty = match *rvsdg.ty().kind(fallback_ty) {
        TypeKind::Ptr(pointee_ty) => pointee_ty,
        _ => fallback_ty,
    };
    let mut replacer = EnumReplacer::new(rvsdg, enum_ty);

    replacer.replace_const_fallback(node, with_replacements);
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::rvsdg::{StateOrigin, ValueOutput};
    use crate::ty::{
        Enum, EnumTagEncoding, EnumVariant, Int, IntSize, Struct, StructField, TY_DUMMY,
        TY_PREDICATE, TY_PTR_U32, TY_U32,
    };
    use crate::{BranchCase, ConstantKind, FnArg, FnSig, Function, Module, Symbol, thin_set};

    #[test]
    fn test_enum_replacement() {
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

        let variant_0_ty = module.ty.register(TypeKind::Struct(Struct {
            fields: vec![StructField {
                offset: 0,
                ty: TY_U32,
                io_binding: None,
            }],
        }));
        let variant_0_ptr_ty = module.ty.register(TypeKind::Ptr(variant_0_ty));

        let variant_1_ty = module.ty.register(TypeKind::Struct(Struct {
            fields: vec![StructField {
                offset: 0,
                ty: TY_U32,
                io_binding: None,
            }],
        }));
        let variant_1_ptr_ty = module.ty.register(TypeKind::Ptr(variant_1_ty));

        let enum_ty = module.ty.register(TypeKind::Enum(Enum {
            variants: vec![
                EnumVariant {
                    ty: variant_0_ty,
                    discriminant: 0,
                },
                EnumVariant {
                    ty: variant_1_ty,
                    discriminant: 1,
                },
            ],
            discriminant_ty: Int {
                size: IntSize::I32,
                signed: false,
            },
            tag_ty: Int {
                size: IntSize::I32,
                signed: false,
            }
            .into(),
            tag_encoding: EnumTagEncoding::Direct,
            tag_offset: 0,
        }));
        let enum_ptr_ty = module.ty.register(TypeKind::Ptr(enum_ty));

        let alloca_node = rvsdg.add_op_alloca(region, enum_ty);
        let switch_0_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(enum_ptr_ty, alloca_node, 0),
            ],
            vec![],
            Some(StateOrigin::Argument),
        );

        let switch_0_branch_0 = rvsdg.add_switch_branch(switch_0_node);

        let switch_0_variant_0_node =
            rvsdg.add_op_variant_ptr(switch_0_branch_0, ValueInput::argument(enum_ptr_ty, 0), 0);
        let switch_0_field_0_node = rvsdg.add_op_field_ptr(
            switch_0_branch_0,
            ValueInput::output(variant_0_ptr_ty, switch_0_variant_0_node, 0),
            0,
        );
        let switch_0_value_0_node = rvsdg.add_const_u32(switch_0_branch_0, 0);
        let switch_0_store_0_node = rvsdg.add_op_store(
            switch_0_branch_0,
            ValueInput::output(TY_PTR_U32, switch_0_field_0_node, 0),
            ValueInput::output(TY_U32, switch_0_value_0_node, 0),
            StateOrigin::Argument,
        );
        let switch_0_set_discr_0_node = rvsdg.add_op_set_discriminant(
            switch_0_branch_0,
            ValueInput::argument(enum_ptr_ty, 0),
            0,
            StateOrigin::Node(switch_0_store_0_node),
        );

        let switch_0_branch_1 = rvsdg.add_switch_branch(switch_0_node);

        let switch_0_variant_1_node =
            rvsdg.add_op_variant_ptr(switch_0_branch_1, ValueInput::argument(enum_ptr_ty, 0), 1);
        let switch_0_field_1_node = rvsdg.add_op_field_ptr(
            switch_0_branch_1,
            ValueInput::output(variant_1_ptr_ty, switch_0_variant_1_node, 0),
            0,
        );
        let switch_0_value_1_node = rvsdg.add_const_u32(switch_0_branch_1, 0);
        let switch_0_store_1_node = rvsdg.add_op_store(
            switch_0_branch_1,
            ValueInput::output(TY_PTR_U32, switch_0_field_1_node, 0),
            ValueInput::output(TY_U32, switch_0_value_1_node, 0),
            StateOrigin::Argument,
        );
        let switch_0_set_discr_1_node = rvsdg.add_op_set_discriminant(
            switch_0_branch_1,
            ValueInput::argument(enum_ptr_ty, 0),
            1,
            StateOrigin::Node(switch_0_store_1_node),
        );

        let get_discr_node = rvsdg.add_op_get_discriminant(
            region,
            ValueInput::output(enum_ptr_ty, alloca_node, 0),
            StateOrigin::Node(switch_0_node),
        );
        let switch_1_predicate = rvsdg.add_op_case_to_branch_selector(
            region,
            ValueInput::output(TY_U32, get_discr_node, 0),
            Int::U32,
            [BranchCase::from(0u32)],
        );
        let switch_1_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::output(TY_PREDICATE, switch_1_predicate, 0),
                ValueInput::output(enum_ptr_ty, alloca_node, 0),
            ],
            vec![ValueOutput::new(TY_U32)],
            Some(StateOrigin::Node(get_discr_node)),
        );

        let switch_1_branch_0 = rvsdg.add_switch_branch(switch_1_node);
        let switch_1_variant_0_node =
            rvsdg.add_op_variant_ptr(switch_1_branch_0, ValueInput::argument(enum_ptr_ty, 0), 0);
        let switch_1_field_0_node = rvsdg.add_op_field_ptr(
            switch_1_branch_0,
            ValueInput::output(variant_0_ptr_ty, switch_1_variant_0_node, 0),
            0,
        );
        let switch_1_load_0_node = rvsdg.add_op_load(
            switch_1_branch_0,
            ValueInput::output(TY_PTR_U32, switch_1_field_0_node, 0),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            switch_1_branch_0,
            0,
            ValueOrigin::Output {
                producer: switch_1_load_0_node,
                output: 0,
            },
        );

        let switch_1_branch_1 = rvsdg.add_switch_branch(switch_1_node);
        let switch_1_variant_1_node =
            rvsdg.add_op_variant_ptr(switch_1_branch_1, ValueInput::argument(enum_ptr_ty, 0), 1);
        let switch_1_field_1_node = rvsdg.add_op_field_ptr(
            switch_1_branch_1,
            ValueInput::output(variant_1_ptr_ty, switch_1_variant_1_node, 0),
            0,
        );
        let switch_1_load_1_node = rvsdg.add_op_load(
            switch_1_branch_1,
            ValueInput::output(TY_PTR_U32, switch_1_field_1_node, 0),
            StateOrigin::Argument,
        );

        rvsdg.reconnect_region_result(
            switch_1_branch_1,
            0,
            ValueOrigin::Output {
                producer: switch_1_load_1_node,
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

        replace_enum_alloca(&mut rvsdg, alloca_node, |_, _| ());

        let switch_0_data = rvsdg[switch_0_node].expect_switch();

        // The second input of the first switch node (that was originally the enum pointer input)
        // should have been replaced with a pointer to the discriminant.
        let ValueOrigin::Output {
            producer: discriminant_node,
            output: 0,
        } = switch_0_data.value_inputs()[1].origin
        else {
            panic!(
                "the second input of the first switch node should connect to the first output \
            of a node"
            )
        };

        let discriminant_data = rvsdg[discriminant_node].expect_op_alloca();

        assert_eq!(
            discriminant_data.ty(),
            TY_U32,
            "the discriminant alloca should have a `u32` type"
        );

        assert_eq!(
            rvsdg[switch_0_node].value_inputs().len(),
            4,
            "two additional inputs have been added to the first switch node"
        );

        let ValueOrigin::Output {
            producer: variant_0_node,
            output: 0,
        } = switch_0_data.value_inputs()[2].origin
        else {
            panic!(
                "the third input of the first switch node should connect to the first output \
            of a node"
            )
        };

        let variant_0_data = rvsdg[variant_0_node].expect_op_alloca();

        assert_eq!(
            variant_0_data.ty(),
            variant_0_ty,
            "the alloca for the first variant should have the correct type"
        );

        let ValueOrigin::Output {
            producer: variant_1_node,
            output: 0,
        } = switch_0_data.value_inputs()[3].origin
        else {
            panic!(
                "the fourth input of the first switch node should connect to the first output \
            of a node"
            )
        };

        let variant_1_data = rvsdg[variant_1_node].expect_op_alloca();

        assert_eq!(
            variant_1_data.ty(),
            variant_1_ty,
            "the alloca for the second variant should have the correct type"
        );

        assert_eq!(rvsdg[switch_0_branch_0].value_arguments()[0].users.len(), 1);

        let ValueUser::Input {
            consumer: discr_store_0_node,
            input: 0,
        } = rvsdg[switch_0_branch_0].value_arguments()[0].users[0]
        else {
            panic!(
                "the discriminant argument of the first branch of the first node should connect \
            to a node"
            );
        };

        let discr_store_0_data = rvsdg[discr_store_0_node].expect_op_store();

        let ValueOrigin::Output {
            producer: variant_index_0_node,
            output: 0,
        } = discr_store_0_data.value_input().origin
        else {
            panic!(
                "the `value` input of the discriminant-store-node in the first branch of the \
            first switch should connect to a node"
            );
        };

        let variant_index_0_data = rvsdg[variant_index_0_node].expect_const_u32();

        assert_eq!(
            variant_index_0_data.value(),
            0,
            "the discriminant stored in the first branch of the first switch node should be `0`"
        );

        assert_eq!(
            rvsdg[switch_0_field_0_node]
                .expect_op_field_ptr()
                .ptr_input()
                .origin,
            ValueOrigin::Argument(1),
            "the field-ptr op in the first branch of the first switch should now take a pointer to \
            the first variant's alloca"
        );

        assert!(
            rvsdg[switch_0_branch_0].value_arguments()[2]
                .users
                .is_empty(),
            "the second variant pointer argument in the first branch of the first switch node \
            should not be used"
        );

        let ValueUser::Input {
            consumer: discr_store_1_node,
            input: 0,
        } = rvsdg[switch_0_branch_1].value_arguments()[0].users[0]
        else {
            panic!(
                "the discriminant argument of the second branch of the first node should connect \
            to a node"
            );
        };

        let discr_store_1_data = rvsdg[discr_store_1_node].expect_op_store();

        let ValueOrigin::Output {
            producer: variant_index_1_node,
            output: 0,
        } = discr_store_1_data.value_input().origin
        else {
            panic!(
                "the `value` input of the discriminant-store-node in the second branch of the \
            first switch should connect to a node"
            );
        };

        let variant_index_1_data = rvsdg[variant_index_1_node].expect_const_u32();

        assert_eq!(
            variant_index_1_data.value(),
            1,
            "the discriminant stored in the second branch of the first switch node should be `0`"
        );

        assert_eq!(
            rvsdg[switch_0_field_1_node]
                .expect_op_field_ptr()
                .ptr_input()
                .origin,
            ValueOrigin::Argument(2),
            "the field-ptr op in the second branch of the first switch should now take a pointer \
            to the second variant's alloca"
        );

        assert!(
            rvsdg[switch_0_branch_1].value_arguments()[1]
                .users
                .is_empty(),
            "the first variant pointer argument in the second branch of the first switch node \
            should not be used"
        );

        let ValueOrigin::Output {
            producer: discr_load_node,
            output: 0,
        } = rvsdg[switch_1_predicate].value_inputs()[0].origin
        else {
            panic!(
                "the input of the predicate for the second switch node should connect to a node"
            );
        };

        let discr_load_data = rvsdg[discr_load_node].expect_op_load();

        assert_eq!(
            discr_load_data.ptr_input().origin,
            ValueOrigin::Output {
                producer: discriminant_node,
                output: 0,
            },
            "the discriminant-load node should load from the discriminant alloca"
        );

        let switch_1_data = rvsdg[switch_1_node].expect_switch();

        assert_eq!(
            switch_1_data.value_inputs()[1].origin,
            ValueOrigin::Output {
                producer: discriminant_node,
                output: 0,
            },
            "the second input of the second switch node should have been replaced by a connect to \
        the discriminant-alloca node"
        );

        assert_eq!(
            switch_1_data.value_inputs().len(),
            4,
            "two additional inputs have been added to the second switch node"
        );

        assert_eq!(
            switch_1_data.value_inputs()[2].origin,
            ValueOrigin::Output {
                producer: variant_0_node,
                output: 0,
            },
            "the third input of the second switch node should connect to the variant-0-alloca node"
        );

        assert_eq!(
            switch_1_data.value_inputs()[3].origin,
            ValueOrigin::Output {
                producer: variant_1_node,
                output: 0,
            },
            "the fourth input of the second switch node should connect to the variant-1-alloca node"
        );

        assert_eq!(
            rvsdg[switch_1_field_0_node]
                .expect_op_field_ptr()
                .ptr_input()
                .origin,
            ValueOrigin::Argument(1),
            "the field-ptr op in the first branch of the second switch should now take a pointer \
            to the first variant's alloca"
        );

        assert!(
            rvsdg[switch_1_branch_0].value_arguments()[0]
                .users
                .is_empty(),
            "the argument for a pointer to the discriminant alloca should not be used in the \
            first branch of the second switch node"
        );

        assert!(
            rvsdg[switch_1_branch_0].value_arguments()[2]
                .users
                .is_empty(),
            "the argument for a pointer to the second variant's alloca should not be used in the \
            first branch of the second switch node"
        );

        assert_eq!(
            rvsdg[switch_1_field_1_node]
                .expect_op_field_ptr()
                .ptr_input()
                .origin,
            ValueOrigin::Argument(2),
            "the field-ptr op in the second branch of the second switch should now take a pointer \
            to the second variant's alloca"
        );

        assert!(
            rvsdg[switch_1_branch_1].value_arguments()[0]
                .users
                .is_empty(),
            "the argument for a pointer to the discriminant alloca should not be used in the \
            second branch of the second switch node"
        );

        assert!(
            rvsdg[switch_1_branch_1].value_arguments()[1]
                .users
                .is_empty(),
            "the argument for a pointer to the first variant's alloca should not be used in the \
            second branch of the second switch node"
        );

        assert_eq!(
            &discriminant_data.value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: switch_0_node,
                    input: 1,
                },
                ValueUser::Input {
                    consumer: switch_1_node,
                    input: 1,
                },
                ValueUser::Input {
                    consumer: discr_load_node,
                    input: 0,
                }
            ],
            "the new discriminant alloca node should be used by the both switch nodes and the \
        discriminant load node"
        );

        assert_eq!(
            &variant_0_data.value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: switch_0_node,
                    input: 2,
                },
                ValueUser::Input {
                    consumer: switch_1_node,
                    input: 2,
                },
            ],
            "the new variant-0 alloca node should be used by the both switch nodes"
        );

        assert_eq!(
            &variant_1_data.value_output().users,
            &thin_set![
                ValueUser::Input {
                    consumer: switch_0_node,
                    input: 3,
                },
                ValueUser::Input {
                    consumer: switch_1_node,
                    input: 3,
                },
            ],
            "the new variant-1 alloca node should be used by the both switch nodes"
        );

        assert!(
            !rvsdg.is_live_node(alloca_node),
            "the original enum alloca node should no longer be live"
        );
        assert!(
            !rvsdg.is_live_node(switch_0_variant_0_node),
            "the ptr-variant-ptr node in the first branch of the first switch node should no \
            longer be live"
        );
        assert!(
            !rvsdg.is_live_node(switch_0_set_discr_0_node),
            "the set-discriminant node in the first branch of the first switch node should no \
            longer be live"
        );
        assert!(
            !rvsdg.is_live_node(switch_0_variant_1_node),
            "the ptr-variant-ptr node in the second branch of the first switch node should no \
            longer be live"
        );
        assert!(
            !rvsdg.is_live_node(switch_0_set_discr_1_node),
            "the set-discriminant node in the second branch of the first switch node should no \
            longer be live"
        );
        assert!(
            !rvsdg.is_live_node(get_discr_node),
            "the get-discriminant node should no longer be live"
        );
        assert!(
            !rvsdg.is_live_node(switch_1_variant_0_node),
            "the ptr-variant-ptr node in the first branch of the second switch node should no \
            longer be live"
        );
        assert!(
            !rvsdg.is_live_node(switch_1_variant_1_node),
            "the ptr-variant-ptr node in the second branch of the second switch node should no \
            longer be live"
        );
    }

    #[test]
    fn test_enum_replacement_switch_passthrough() {
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

        let variant_0_ty = module.ty.register(TypeKind::Struct(Struct {
            fields: vec![StructField {
                offset: 0,
                ty: TY_U32,
                io_binding: None,
            }],
        }));

        let variant_1_ty = module.ty.register(TypeKind::Struct(Struct {
            fields: vec![StructField {
                offset: 0,
                ty: TY_U32,
                io_binding: None,
            }],
        }));

        let enum_ty = module.ty.register(TypeKind::Enum(Enum {
            variants: vec![
                EnumVariant {
                    ty: variant_0_ty,
                    discriminant: 0,
                },
                EnumVariant {
                    ty: variant_1_ty,
                    discriminant: 1,
                },
            ],
            discriminant_ty: Int {
                size: IntSize::I32,
                signed: false,
            },
            tag_ty: Int {
                size: IntSize::I32,
                signed: false,
            }
            .into(),
            tag_encoding: EnumTagEncoding::Direct,
            tag_offset: 0,
        }));
        let enum_ptr_ty = module.ty.register(TypeKind::Ptr(enum_ty));

        let alloca_node = rvsdg.add_op_alloca(region, enum_ty);
        let set_discr_node = rvsdg.add_op_set_discriminant(
            region,
            ValueInput::output(enum_ptr_ty, alloca_node, 0),
            1,
            StateOrigin::Argument,
        );
        let switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0),
                ValueInput::output(enum_ptr_ty, alloca_node, 0),
            ],
            vec![ValueOutput::new(enum_ptr_ty)],
            Some(StateOrigin::Node(set_discr_node)),
        );

        let branch_0 = rvsdg.add_switch_branch(switch_node);
        rvsdg.reconnect_region_result(branch_0, 0, ValueOrigin::Argument(0));

        let branch_1 = rvsdg.add_switch_branch(switch_node);
        rvsdg.reconnect_region_result(branch_1, 0, ValueOrigin::Argument(0));

        let get_discr_node = rvsdg.add_op_get_discriminant(
            region,
            ValueInput::output(enum_ptr_ty, switch_node, 0),
            StateOrigin::Node(switch_node),
        );

        rvsdg.reconnect_region_result(
            region,
            0,
            ValueOrigin::Output {
                producer: get_discr_node,
                output: 0,
            },
        );

        replace_enum_alloca(&mut rvsdg, alloca_node, |_, _| ());

        // This test was added because the "pass-through" of a ptr to an enum of an alloca through
        // a switch node was causing enum-replacement to panic. This test then simply verifies that
        // the fix no longer causes a panic; hence no further assertions.
    }

    #[test]
    fn test_enum_replacement_copy_known_constant() {
        let mut module = Module::new(Symbol::from_ref("test"));
        let function = Function {
            name: Symbol::from_ref("main"),
            module: module.name,
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Default::default(),
                ty: TY_DUMMY,
                args: vec![],
                ret_ty: None,
            },
        );

        let inactive_ty = module.ty.register(TypeKind::Struct(Struct {
            fields: vec![StructField {
                offset: 8,
                ty: TY_PTR_U32,
                io_binding: None,
            }],
        }));

        let active_ty = module.ty.register(TypeKind::Struct(Struct {
            fields: vec![StructField {
                offset: 4,
                ty: TY_U32,
                io_binding: None,
            }],
        }));
        let active_ptr_ty = module.ty.register(TypeKind::Ptr(active_ty));

        let enum_ty = module.ty.register(TypeKind::Enum(Enum {
            variants: vec![
                EnumVariant {
                    ty: inactive_ty,
                    discriminant: 7,
                },
                EnumVariant {
                    ty: active_ty,
                    discriminant: 42,
                },
            ],
            discriminant_ty: Int::U32,
            tag_ty: Int::U32.into(),
            tag_encoding: EnumTagEncoding::Direct,
            tag_offset: 0,
        }));
        let enum_ptr_ty = module.ty.register(TypeKind::Ptr(enum_ty));

        let alloc_id = module.allocations.register(Allocation {
            bytes: vec![42, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        });
        let constant = Constant {
            name: Symbol::from_ref("ENUM_CONST"),
            module: module.name,
        };

        module
            .constants
            .register_byte_data(constant, enum_ty, alloc_id, 0);

        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let constant_node = rvsdg.register_constant(&module, constant);
        let (function_node, region) = rvsdg.register_function(&module, function, [constant_node]);

        let destination = rvsdg.add_op_alloca(region, enum_ty);
        let load = rvsdg.add_op_load(
            region,
            ValueInput::argument(enum_ptr_ty, 0),
            StateOrigin::Argument,
        );
        let store = rvsdg.add_op_store(
            region,
            ValueInput::output(enum_ptr_ty, destination, 0),
            ValueInput::output(enum_ty, load, 0),
            StateOrigin::Node(load),
        );

        replace_enum_constant_dependency(
            &mut rvsdg,
            &mut module,
            function_node,
            0,
            constant,
            alloc_id,
            0,
            |_| (),
        );
        replace_enum_alloca(&mut rvsdg, destination, |_, _| ());

        let loads = rvsdg[region]
            .nodes()
            .iter()
            .copied()
            .filter(|node| rvsdg[*node].is_op_load())
            .collect::<Vec<_>>();
        let stores = rvsdg[region]
            .nodes()
            .iter()
            .copied()
            .filter(|node| rvsdg[*node].is_op_store())
            .collect::<Vec<_>>();

        assert_eq!(
            loads.len(),
            2,
            "only the discriminant and active payload are loaded"
        );
        assert_eq!(
            stores.len(),
            2,
            "only the discriminant and active payload are stored"
        );

        let discriminant_load = *loads
            .iter()
            .find(|node| rvsdg[**node].expect_op_load().ptr_input().ty == TY_PTR_U32)
            .unwrap();
        let active_load = *loads
            .iter()
            .find(|node| rvsdg[**node].expect_op_load().ptr_input().ty == active_ptr_ty)
            .unwrap();

        assert_eq!(
            *rvsdg[discriminant_load].expect_op_load().ptr_input(),
            ValueInput::argument(TY_PTR_U32, 1),
        );
        assert_eq!(
            *rvsdg[active_load].expect_op_load().ptr_input(),
            ValueInput::argument(active_ptr_ty, 2),
        );

        let dependencies = rvsdg[function_node].expect_function().dependencies();

        let ValueOrigin::Output {
            producer: discriminant_node,
            output: 0,
        } = dependencies[1].origin
        else {
            panic!("the discriminant dependency should connect to a constant node");
        };
        let discriminant_constant = rvsdg[discriminant_node].expect_constant().constant();
        let ConstantKind::ByteData(discriminant_alloc, discriminant_offset) =
            *module.constants[discriminant_constant].kind()
        else {
            panic!("the discriminant constant should contain byte data");
        };

        assert_eq!(module.constants[discriminant_constant].ty(), TY_U32);
        assert_eq!(
            &module.allocations[discriminant_alloc].bytes[discriminant_offset..],
            &[42, 0, 0, 0],
        );

        let ValueOrigin::Output {
            producer: active_node,
            output: 0,
        } = dependencies[2].origin
        else {
            panic!("the active payload dependency should connect to a constant node");
        };
        let active_constant = rvsdg[active_node].expect_constant().constant();

        assert_eq!(module.constants[active_constant].ty(), active_ty);
        assert!(matches!(
            module.constants[active_constant].kind(),
            ConstantKind::ByteData(id, 0) if *id == alloc_id
        ));

        let discriminant_store = *stores
            .iter()
            .find(|node| rvsdg[**node].expect_op_store().value_input().ty == TY_U32)
            .unwrap();
        let active_store = *stores
            .iter()
            .find(|node| rvsdg[**node].expect_op_store().value_input().ty == active_ty)
            .unwrap();

        assert_eq!(
            *rvsdg[discriminant_store].expect_op_store().value_input(),
            ValueInput::output(TY_U32, discriminant_load, 0),
        );
        assert_eq!(
            *rvsdg[active_store].expect_op_store().value_input(),
            ValueInput::output(active_ty, active_load, 0),
        );

        let ValueOrigin::Output {
            producer: discriminant_destination,
            output: 0,
        } = rvsdg[discriminant_store]
            .expect_op_store()
            .ptr_input()
            .origin
        else {
            panic!("the discriminant store should target an alloca");
        };
        let ValueOrigin::Output {
            producer: active_destination,
            output: 0,
        } = rvsdg[active_store].expect_op_store().ptr_input().origin
        else {
            panic!("the active payload store should target an alloca");
        };

        assert_eq!(
            rvsdg[discriminant_destination].expect_op_alloca().ty(),
            TY_U32
        );
        assert_eq!(rvsdg[active_destination].expect_op_alloca().ty(), active_ty);

        // Check the state chain
        assert_eq!(
            rvsdg[discriminant_load].state().unwrap().origin,
            StateOrigin::Argument
        );
        assert_eq!(
            rvsdg[active_load].state().unwrap().origin,
            StateOrigin::Node(discriminant_load)
        );
        assert_eq!(
            rvsdg[discriminant_store].state().unwrap().origin,
            StateOrigin::Node(active_load)
        );
        assert_eq!(
            rvsdg[active_store].state().unwrap().origin,
            StateOrigin::Node(discriminant_store)
        );
        assert_eq!(
            *rvsdg[region].state_result(),
            StateOrigin::Node(active_store)
        );

        assert!(rvsdg[region].value_arguments()[0].users.is_empty());

        assert!(!rvsdg.is_live_node(load));
        assert!(!rvsdg.is_live_node(store));
        assert!(!rvsdg.is_live_node(destination));
    }
}
