//! Replaces invalid pointer-refinement nodes with [ConstFallback][0] nodes.
//!
//! This transformation identifies "invalid" pointer-refinement nodes (e.g., [OpElementPtr][1],
//! [OpFieldPtr][2]) and replaces them with [ConstFallback][0] nodes of the appropriate pointer
//! type.
//!
//! A pointer-refinement node is considered invalid if:
//!
//! 1. Its pointer input traces back to a [ConstFallback][0] node.
//! 2. It is an [OpElementPtr][1] operation on a zero-length array.
//!
//! While we don't consider the creation of invalid pointers to be Undefined Behavior, using such
//! pointers in memory load or store operations is Undefined Behavior; this is a preparatory pass
//! that makes it easier for later transforms to identify such load and store operations.
//!
//! [0]: crate::rvsdg::ConstFallback
//! [1]: crate::rvsdg::OpElementPtr
//! [2]: crate::rvsdg::OpFieldPtr

use rustc_hash::FxHashSet;

use crate::Module;
use crate::rvsdg::visit::region_nodes::{RegionNodesVisitor, visit_node};
use crate::rvsdg::{Connectivity, Node, NodeKind, Region, Rvsdg, SimpleNode, ValueOrigin};
use crate::ty::{Type, TypeKind};

/// Replaces invalid pointer-refinement nodes with [ConstFallback][0] nodes.
///
/// See the [module-level documentation](crate::rvsdg::transform::invalid_ptr_replacement) for
/// details.
pub struct InvalidPtrReplacer {
    finder: PointerOriginFinder,
    jobs: Vec<Job>,
}

struct Job {
    node: Node,
    ty: Type,
}

impl InvalidPtrReplacer {
    pub fn new() -> Self {
        Self {
            finder: PointerOriginFinder::new(),
            jobs: Vec::new(),
        }
    }

    pub fn process_region(&mut self, rvsdg: &mut Rvsdg, region: Region) -> bool {
        let mut changed = false;
        let mut do_iteration = true;

        self.jobs.clear();

        while do_iteration {
            do_iteration = false;

            JobCollector {
                jobs: &mut self.jobs,
                finder: &mut self.finder,
            }
            .visit_region(rvsdg, region);

            for job in self.jobs.drain(..) {
                let node_region = rvsdg[job.node].region();

                let fallback = rvsdg.add_const_fallback(node_region, job.ty);
                rvsdg.reconnect_value_users(
                    node_region,
                    ValueOrigin::Output {
                        producer: job.node,
                        output: 0,
                    },
                    ValueOrigin::Output {
                        producer: fallback,
                        output: 0,
                    },
                );

                rvsdg.remove_node(job.node);

                do_iteration = true;
                changed = true;
            }
        }

        changed
    }
}

struct JobCollector<'a> {
    jobs: &'a mut Vec<Job>,
    finder: &'a mut PointerOriginFinder,
}

impl<'a> RegionNodesVisitor for JobCollector<'a> {
    fn visit_node(&mut self, rvsdg: &Rvsdg, node: Node) {
        let node_data = &rvsdg[node];

        if let NodeKind::Simple(simple_node) = node_data.kind() {
            match simple_node {
                SimpleNode::OpElementPtr(op_element_ptr) => {
                    let ptr_input = &op_element_ptr.value_inputs()[0];
                    let ptr_ty = ptr_input.ty;

                    // Check if the pointee is a zero-length array first, as that may allow us to
                    // skip the more expensive trace.
                    let pointee_ty = rvsdg.ty().kind(ptr_ty).expect_ptr();
                    let is_invalid =
                        if let TypeKind::Array { count: 0, .. } = *rvsdg.ty().kind(pointee_ty) {
                            true
                        } else {
                            self.finder
                                .trace(rvsdg, node_data.region(), ptr_input.origin)
                        };

                    if is_invalid {
                        let output_ty = op_element_ptr.value_outputs()[0].ty;

                        self.jobs.push(Job {
                            node,
                            ty: output_ty,
                        });
                    }
                }
                SimpleNode::OpFieldPtr(op_field_ptr) => {
                    let ptr_input = &op_field_ptr.value_inputs()[0];

                    if self
                        .finder
                        .trace(rvsdg, node_data.region(), ptr_input.origin)
                    {
                        let output_ty = op_field_ptr.value_outputs()[0].ty;

                        self.jobs.push(Job {
                            node,
                            ty: output_ty,
                        });
                    }
                }
                _ => {}
            }
        }

        // Continue visiting subregions
        visit_node(self, rvsdg, node);
    }
}

struct PointerOriginFinder {
    visited: FxHashSet<(Region, ValueOrigin)>,
}

impl PointerOriginFinder {
    fn new() -> Self {
        Self {
            visited: FxHashSet::default(),
        }
    }

    fn trace(&mut self, rvsdg: &Rvsdg, region: Region, origin: ValueOrigin) -> bool {
        self.visited.clear();
        self.trace_recursive(rvsdg, region, origin)
    }

    fn trace_recursive(&mut self, rvsdg: &Rvsdg, region: Region, origin: ValueOrigin) -> bool {
        if !self.visited.insert((region, origin)) {
            return false;
        }

        match origin {
            ValueOrigin::Output { producer, .. } => {
                let producer_data = &rvsdg[producer];
                if let NodeKind::Simple(simple_node) = producer_data.kind() {
                    match simple_node {
                        SimpleNode::ConstFallback(_) => return true,
                        SimpleNode::ValueProxy(value_proxy) => {
                            return self.trace_recursive(
                                rvsdg,
                                producer_data.region(),
                                value_proxy.value_inputs()[0].origin,
                            );
                        }
                        _ => {}
                    }
                }
                false
            }
            ValueOrigin::Argument(arg_index) => {
                if region == rvsdg.global_region() {
                    return false;
                }

                let owner_node = rvsdg[region].owner();
                let owner_data = &rvsdg[owner_node];
                let outer_region = owner_data.region();

                match owner_data.kind() {
                    NodeKind::Switch(switch_node) => {
                        let input_index = arg_index + 1;
                        self.trace_recursive(
                            rvsdg,
                            outer_region,
                            switch_node.value_inputs()[input_index as usize].origin,
                        )
                    }
                    NodeKind::Loop(loop_node) => {
                        // Argument i is a loop-constant if the loop-region value result i + 1
                        // has an origin that is the same region-argument i.
                        let loop_region = loop_node.loop_region();
                        let result_origin =
                            rvsdg[loop_region].value_results()[arg_index as usize + 1].origin;

                        if let ValueOrigin::Argument(res_arg_index) = result_origin {
                            if res_arg_index == arg_index {
                                return self.trace_recursive(
                                    rvsdg,
                                    outer_region,
                                    loop_node.value_inputs()[arg_index as usize].origin,
                                );
                            }
                        }
                        false
                    }
                    _ => false,
                }
            }
        }
    }
}

/// Replaces invalid pointer-refinement nodes with [ConstFallback][0] nodes in all entry-point
/// functions.
///
/// See the [module-level documentation](crate::rvsdg::transform::invalid_ptr_replacement) for
/// details.
pub fn transform_entry_points(module: &Module, rvsdg: &mut Rvsdg) {
    let mut replacer = InvalidPtrReplacer::new();

    for (entry_point, _) in module.entry_points.iter() {
        let fn_node = rvsdg
            .get_function_node(entry_point)
            .expect("function not registered");
        let body_region = rvsdg[fn_node].expect_function().body_region();

        replacer.process_region(rvsdg, body_region);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rvsdg::{ValueInput, ValueOrigin};
    use crate::ty::{Struct, StructField, TY_DUMMY, TY_PREDICATE, TY_PTR_U32, TY_U32, TypeKind};
    use crate::{FnArg, FnSig, Function, Module, Symbol};

    #[test]
    fn test_invalid_ptr_replacement_zero_length_array() {
        let mut module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let pointee_ty = rvsdg.ty().register(TypeKind::Array {
            element_ty: TY_U32,
            count: 0,
            stride: 4,
        });
        let ptr_ty = rvsdg.ty().register(TypeKind::Ptr(pointee_ty));

        let function = Function {
            name: Symbol::from_ref("f"),
            module: Symbol::from_ref("m"),
        };

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

        let index = rvsdg.add_const_u32(region, 0);
        let op_element_ptr = rvsdg.add_op_element_ptr(
            region,
            ValueInput::argument(ptr_ty, 0),
            ValueInput::output(TY_U32, index, 0),
        );

        let element_ptr_ty = rvsdg.ty().register(TypeKind::Ptr(TY_U32));
        let user_node = rvsdg.add_value_proxy(
            region,
            ValueInput::output(element_ptr_ty, op_element_ptr, 0),
        );

        let mut replacer = InvalidPtrReplacer::new();

        assert!(replacer.process_region(&mut rvsdg, region));

        let ValueOrigin::Output { producer, .. } = rvsdg[user_node].value_inputs()[0].origin else {
            panic!("expected output origin");
        };

        let fallback = rvsdg[producer].expect_const_fallback();
        assert_eq!(fallback.ty(), TY_PTR_U32);

        assert!(!rvsdg.is_live_node(op_element_ptr));
    }

    #[test]
    fn test_invalid_ptr_replacement_trace_to_fallback() {
        let mut module = Module::new(Symbol::from_ref(""));
        let mut rvsdg = Rvsdg::new(module.ty.clone());

        let struct_ty = rvsdg.ty().register(TypeKind::Struct(Struct {
            fields: vec![StructField {
                offset: 0,
                ty: TY_U32,
                io_binding: None,
            }],
        }));
        let ptr_ty = rvsdg.ty().register(TypeKind::Ptr(struct_ty));

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

        let fallback = rvsdg.add_const_fallback(region, ptr_ty);
        let proxy = rvsdg.add_value_proxy(region, ValueInput::output(ptr_ty, fallback, 0));
        let op_field_ptr = rvsdg.add_op_field_ptr(region, ValueInput::output(ptr_ty, proxy, 0), 0);
        let user_node =
            rvsdg.add_value_proxy(region, ValueInput::output(TY_PTR_U32, op_field_ptr, 0));

        let mut replacer = InvalidPtrReplacer::new();

        assert!(replacer.process_region(&mut rvsdg, region));

        let ValueOrigin::Output { producer, .. } = rvsdg[user_node].value_inputs()[0].origin else {
            panic!("expected output origin");
        };

        let fallback = rvsdg[producer].expect_const_fallback();
        assert_eq!(fallback.ty(), TY_PTR_U32);

        assert!(!rvsdg.is_live_node(op_field_ptr));
    }

    #[test]
    fn test_invalid_ptr_replacement_trace_through_switch() {
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
                args: vec![FnArg {
                    ty: TY_PREDICATE,
                    shader_io_binding: None,
                }],
                ret_ty: None,
            },
        );

        let (_, region) = rvsdg.register_function(&module, function, Vec::new());

        let struct_ty = rvsdg.ty().register(TypeKind::Struct(Struct {
            fields: vec![StructField {
                offset: 0,
                ty: TY_U32,
                io_binding: None,
            }],
        }));
        let ptr_ty = rvsdg.ty().register(TypeKind::Ptr(struct_ty));

        let fallback = rvsdg.add_const_fallback(region, ptr_ty);

        let switch_node = rvsdg.add_switch(
            region,
            vec![
                ValueInput::argument(TY_PREDICATE, 0), // selector
                ValueInput::output(ptr_ty, fallback, 0),
            ],
            vec![],
            None,
        );

        let branch_region = rvsdg.add_switch_branch(switch_node);

        let op_field_ptr =
            rvsdg.add_op_field_ptr(branch_region, ValueInput::argument(ptr_ty, 0), 0);
        let user_node = rvsdg.add_value_proxy(
            branch_region,
            ValueInput::output(TY_PTR_U32, op_field_ptr, 0),
        );

        let mut replacer = InvalidPtrReplacer::new();

        assert!(replacer.process_region(&mut rvsdg, region));

        let ValueOrigin::Output { producer, .. } = rvsdg[user_node].value_inputs()[0].origin else {
            panic!("expected output origin");
        };

        let fallback = rvsdg[producer].expect_const_fallback();
        assert_eq!(fallback.ty(), TY_PTR_U32);

        assert!(!rvsdg.is_live_node(op_field_ptr));
    }

    #[test]
    fn test_invalid_ptr_replacement_trace_through_loop_constant() {
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
                args: Vec::new(),
                ret_ty: None,
            },
        );

        let (_, region) = rvsdg.register_function(&module, function, Vec::new());

        let struct_ty = rvsdg.ty().register(TypeKind::Struct(Struct {
            fields: vec![StructField {
                offset: 0,
                ty: TY_U32,
                io_binding: None,
            }],
        }));
        let ptr_ty = rvsdg.ty().register(TypeKind::Ptr(struct_ty));

        let fallback = rvsdg.add_const_fallback(region, ptr_ty);

        let (loop_node, loop_region) =
            rvsdg.add_loop(region, vec![ValueInput::output(ptr_ty, fallback, 0)], None);

        let op_field_ptr = rvsdg.add_op_field_ptr(loop_region, ValueInput::argument(ptr_ty, 0), 0);
        let user_node =
            rvsdg.add_value_proxy(loop_region, ValueInput::output(TY_PTR_U32, op_field_ptr, 0));

        let bool_true = rvsdg.add_const_bool(loop_region, true);
        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: bool_true,
                output: 0,
            },
        );

        // Loop result 1 must be the same argument 0 for it to be a loop-constant
        rvsdg.reconnect_region_result(loop_region, 1, ValueOrigin::Argument(0));

        let mut replacer = InvalidPtrReplacer::new();

        let _loop_node = loop_node;
        replacer.process_region(&mut rvsdg, region);

        let ValueOrigin::Output { producer, .. } = rvsdg[user_node].value_inputs()[0].origin else {
            panic!("expected output origin");
        };

        let fallback = rvsdg[producer].expect_const_fallback();
        assert_eq!(fallback.ty(), TY_PTR_U32);

        assert!(!rvsdg.is_live_node(op_field_ptr));
    }
}
