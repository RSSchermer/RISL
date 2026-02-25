
use crate::rvsdg::{
    Connectivity, Node, NodeKind, Region, Rvsdg, SimpleNode, ValueInput, ValueOrigin, ValueUser,
};
use crate::ty::{TY_PTR_U32, TY_U32, Type, TypeKind, TypeRegistry};

pub struct EnumAllocaReplacer<'a> {
    rvsdg: &'a mut Rvsdg,
    node: Node,
    ty: TypeRegistry,
    enum_ty: Type,
}

impl<'a> EnumAllocaReplacer<'a> {
    pub fn new(rvsdg: &'a mut Rvsdg, node: Node) -> Self {
        let enum_ty = rvsdg[node].expect_op_alloca().ty();
        let ty_registry = rvsdg.ty().clone();

        assert!(ty_registry.kind(enum_ty).is_enum());

        EnumAllocaReplacer {
            ty: rvsdg.ty().clone(),
            rvsdg,
            enum_ty,
            node,
        }
    }

    pub fn replace_alloca<F>(&mut self, mut with_replacements: F)
    where
        F: FnMut(Node, Type),
    {
        let node_data = &self.rvsdg[self.node];
        let region = node_data.region();
        let discriminant_node = self.rvsdg.add_op_alloca(region, TY_U32);

        let mut replacements = Vec::new();

        replacements.push(ValueInput {
            ty: TY_PTR_U32,
            origin: ValueOrigin::Output {
                producer: discriminant_node,
                output: 0,
            },
        });

        let ty_kind = self.ty.kind(self.enum_ty);

        replacements.extend(ty_kind.expect_enum().variants.iter().copied().map(|ty| {
            let ptr_ty = self.ty.register(TypeKind::Ptr(ty));
            let variant_node = self.rvsdg.add_op_alloca(region, ty);

            with_replacements(variant_node, ty);

            ValueInput {
                ty: ptr_ty,
                origin: ValueOrigin::Output {
                    producer: variant_node,
                    output: 0,
                },
            }
        }));

        self.visit_users(self.node, 0, &replacements);

        // The OpAlloca node now should not have any users left, so we can remove it
        self.rvsdg.remove_node(self.node);
    }

    fn visit_users(&mut self, node: Node, output: u32, split_inputs: &[ValueInput]) {
        let region = self.rvsdg[node].region();
        let user_count = self.rvsdg[node].value_outputs()[output as usize]
            .users
            .len();

        // We iterate over users in reverse order, so that users may more themselves from the user
        // set, without disrupting iteration
        for i in (0..user_count).rev() {
            let user = self.rvsdg[node].value_outputs()[output as usize].users[i];

            self.visit_user(region, user, split_inputs);
        }
    }

    fn visit_user(&mut self, region: Region, user: ValueUser, split_input: &[ValueInput]) {
        match user {
            ValueUser::Result(result) => self.split_region_result(region, result, split_input),
            ValueUser::Input { consumer, input } => {
                self.visit_node_input(consumer, input, split_input)
            }
        }
    }

    fn split_region_result(&mut self, region: Region, result: u32, split_input: &[ValueInput]) {
        let owner = self.rvsdg[region].owner();

        match self.rvsdg[owner].kind() {
            NodeKind::Switch(_) => self.split_switch_result(region, result, split_input),
            NodeKind::Loop(_) => self.split_loop_result(region, result, split_input),
            NodeKind::Function(_) => panic!(
                "cannot split function result; \
            non-local-use analyses should have rejected the alloca"
            ),
            _ => unreachable!("node kind cannot by a region owner"),
        }
    }

    fn visit_node_input(&mut self, node: Node, input: u32, split_input: &[ValueInput]) {
        use NodeKind::*;
        use SimpleNode::*;

        match self.rvsdg[node].kind() {
            Switch(_) => self.split_switch_input(node, input, split_input),
            Loop(_) => self.split_loop_input(node, input, split_input),
            Simple(OpGetDiscriminant(_)) => self.replace_op_get_discriminant(node, split_input),
            Simple(OpSetDiscriminant(_)) => self.replace_op_set_discriminant(node, split_input),
            Simple(OpDiscriminantPtr(_)) => self.elide_op_discriminant_ptr(node, split_input),
            Simple(OpVariantPtr(_)) => self.elide_op_variant_ptr(node, split_input),
            Simple(ValueProxy(_)) => self.visit_value_proxy(node, split_input),
            _ => unreachable!("node kind cannot take a pointer to an enum as input"),
        }
    }

    fn elide_op_discriminant_ptr(&mut self, node: Node, split_input: &[ValueInput]) {
        let node_data = &self.rvsdg[node];
        let region = node_data.region();
        let node_data = node_data.expect_op_discriminant_ptr();
        let new_user_origin = split_input[0].origin;
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

    fn elide_op_variant_ptr(&mut self, node: Node, split_input: &[ValueInput]) {
        let node_data = &self.rvsdg[node];
        let region = node_data.region();
        let node_data = node_data.expect_op_variant_ptr();
        let variant_index = node_data.variant_index();
        let new_user_origin = split_input[variant_index as usize + 1].origin;
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

    fn replace_op_get_discriminant(&mut self, node: Node, split_input: &[ValueInput]) {
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

        let load_node = self.rvsdg.add_op_load(region, split_input[0], state_origin);

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

    fn replace_op_set_discriminant(&mut self, node: Node, split_input: &[ValueInput]) {
        let region = self.rvsdg[node].region();
        let state_origin = self.rvsdg[node]
            .state()
            .expect("set-discriminant operation should part of state chain")
            .origin;
        let node_data = self.rvsdg[node].expect_op_set_discriminant();
        let variant_index = node_data.variant_index();

        let variant_node = self.rvsdg.add_const_u32(region, variant_index);

        self.rvsdg.add_op_store(
            region,
            split_input[0],
            ValueInput::output(TY_U32, variant_node, 0),
            state_origin,
        );
        self.rvsdg.remove_node(node);
    }

    fn split_switch_input(&mut self, node: Node, input: u32, split_input: &[ValueInput]) {
        assert_ne!(input, 0, "the branch selector input is never an aggregate");

        let arg_index = input as usize - 1;
        let node_data = self.rvsdg[node].expect_switch();
        let branch_count = node_data.branches().len();
        let split_args_base = node_data.value_inputs().len() - 1;

        let split_args = split_input
            .iter()
            .copied()
            .enumerate()
            .map(|(i, input)| {
                self.rvsdg.add_switch_input(node, input);

                ValueInput {
                    ty: input.ty,
                    origin: ValueOrigin::Argument((split_args_base + i) as u32),
                }
            })
            .collect::<Vec<_>>();

        for branch_index in 0..branch_count {
            let branch = self.rvsdg[node].expect_switch().branches()[branch_index];

            self.redirect_region_argument(branch, arg_index as u32, &split_args);
        }

        self.rvsdg.remove_switch_input(node, input);
    }

    fn split_switch_result(&mut self, branch: Region, result: u32, split_input: &[ValueInput]) {
        let node = self.rvsdg[branch].owner();
        let node_data = self.rvsdg[node].expect_switch();
        let branch_count = node_data.branches().len();
        let base_index = node_data.value_outputs().len();

        let mut split_output = Vec::with_capacity(split_input.len());

        // First split the output/results and connect the new results for the provoking branch. Also
        // record a mapping for the split output
        for (i, input) in split_input.iter().enumerate() {
            let index = base_index + i;

            self.rvsdg.add_switch_output(node, input.ty);
            self.rvsdg
                .reconnect_region_result(branch, index as u32, input.origin);

            split_output.push(ValueInput {
                ty: input.ty,
                origin: ValueOrigin::Output {
                    producer: node,
                    output: index as u32,
                },
            })
        }

        // Now reconnect the results for the other branches
        for i in 0..branch_count {
            let current_branch = self.rvsdg[node].expect_switch().branches()[i];

            if current_branch != branch {
                self.redirect_region_result(current_branch, result as usize, base_index);
            }
        }

        // Finally, split the output
        self.visit_users(node, result, &split_output);

        // Disconnect the original region result for the provoking branch, as its users have been
        // replaced by the split results
        self.rvsdg.disconnect_region_result(branch, result);
    }

    fn split_loop_input(&mut self, node: Node, input: u32, split_input: &[ValueInput]) {
        let node_data = self.rvsdg[node].expect_loop();
        let loop_region = node_data.loop_region();
        let prior_input_count = node_data.value_inputs().len();
        let prior_result_count = prior_input_count + 1;

        let mut split_args = Vec::with_capacity(split_input.len());
        let mut split_outputs = Vec::with_capacity(split_input.len());

        // Add inputs/outputs/arguments/results for each element of the original aggregate input
        // and record mappings for both the arguments and the outputs.
        for (i, input) in split_input.iter().enumerate() {
            self.rvsdg.add_loop_input(node, *input);

            split_args.push(ValueInput {
                ty: input.ty,
                origin: ValueOrigin::Argument((prior_input_count + i) as u32),
            });
            split_outputs.push(ValueInput {
                ty: input.ty,
                origin: ValueOrigin::Output {
                    producer: node,
                    output: (prior_input_count + i) as u32,
                },
            });
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
        self.redirect_region_argument(loop_region, input, &split_args);

        // Finally, redirect the value output using the output mapping
        self.visit_users(node, input, &split_outputs);

        // Now neither the argument nor the output should have any remaining users, so we can remove
        // the original input/output/argument/result and disconnect the input from its origin.
        self.rvsdg.remove_loop_input(node, input);
    }

    fn split_loop_result(&mut self, region: Region, result: u32, split_input: &[ValueInput]) {
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

        let mut split_args = Vec::with_capacity(split_input.len());
        let mut split_outputs = Vec::with_capacity(split_input.len());

        // Add an input for the discriminant and for each variant and connect them to the proxy via
        // an OpPtrDiscriminantPtr node for the discriminant input or an OpPtrVariantPtr node for
        // the variant inputs. Also record an argument mapping in `split_args` and an output mapping
        // in `split_outputs`.

        let discriminant_node = self
            .rvsdg
            .add_op_discriminant_ptr(outer_region, proxy_input);

        self.rvsdg
            .add_loop_input(owner, ValueInput::output(TY_PTR_U32, discriminant_node, 0));

        split_args.push(ValueInput::argument(TY_PTR_U32, prior_input_count as u32));
        split_outputs.push(ValueInput::output(
            TY_PTR_U32,
            owner,
            prior_input_count as u32,
        ));

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

            split_args.push(ValueInput::argument(variant_ptr_ty, input_index));
            split_outputs.push(ValueInput::output(variant_ptr_ty, owner, input_index));
        }

        // Reconnect the results we just created to the `split_input`.
        for (i, input) in split_input.iter().enumerate() {
            let result_index = prior_result_count + i;

            self.rvsdg
                .reconnect_region_result(region, result_index as u32, input.origin);
        }

        // Redirect the argument using the argument mapping
        self.redirect_region_argument(region, input_index, &split_args);

        // Redirect the value output using the output mapping
        self.visit_users(owner, input_index, &split_outputs);

        // Now neither the argument nor the output should have any remaining users, so we can remove
        // the original input/output/argument/result and disconnect the input from its origin. We
        // don't have to worry about the input's removal affecting the user traversal of an upstream
        // node, because we proxied the input earlier.
        self.rvsdg.remove_loop_input(owner, input_index);
    }

    fn visit_value_proxy(&mut self, node: Node, split_input: &[ValueInput]) {
        self.visit_users(node, 0, split_input);
        self.rvsdg.remove_node(node);
    }

    /// Redirects all users of the `region`'s given `argument` to the `split_input` nodes.
    ///
    /// Leaves the `argument` without any users.
    fn redirect_region_argument(
        &mut self,
        region: Region,
        argument: u32,
        split_input: &[ValueInput],
    ) {
        let arg_index = argument as usize;
        let user_count = self.rvsdg[region].value_arguments()[arg_index].users.len();

        // We iterate over users in reverse order, so that users may more themselves from the user
        // set, without disrupting iteration
        for user_index in (0..user_count).rev() {
            let user = self.rvsdg[region].value_arguments()[arg_index].users[user_index];

            self.visit_user(region, user, split_input)
        }
    }

    /// Redirects the origin for the `region`'s given `result` to a set of "split" results that
    /// start at `split_results_start`, via either [OpPtrDiscriminant] for the discriminant or
    /// [OpPtrVariantPtr] nodes for the variants.
    ///
    /// Leaves the original result connected to the "placeholder" origin.
    fn redirect_region_result(
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
    EnumAllocaReplacer::new(rvsdg, node).replace_alloca(with_replacements)
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::rvsdg::{StateOrigin, ValueOutput};
    use crate::ty::{Enum, Struct, StructField, TY_DUMMY, TY_PREDICATE};
    use crate::{FnArg, FnSig, Symbol, thin_set, Module, Function};

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
            variants: vec![variant_0_ty, variant_1_ty],
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
        let switch_1_predicate = rvsdg.add_op_u32_to_branch_selector(
            region,
            2,
            ValueInput::output(TY_U32, get_discr_node, 0),
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
            variants: vec![variant_0_ty, variant_1_ty],
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
}
