use rustc_hash::FxHashSet;

use crate::rvsdg::NodeKind::Simple;
use crate::rvsdg::SimpleNode::OpElementPtr;
use crate::rvsdg::visit::bottom_up::{BottomUpVisitor, visit_node_bottom_up};
use crate::rvsdg::{Node, Region, Rvsdg, ValueInput, ValueOrigin};
use crate::ty::{TY_U32, Type, TypeKind, TypeRegistry};
use crate::{BinaryOperator, Function, Module};

fn is_slice_ptr_ty(type_registry: &TypeRegistry, ty: Type) -> bool {
    if let TypeKind::Ptr(pointee_ty) = *type_registry.kind(ty) {
        type_registry.kind(pointee_ty).is_slice()
    } else {
        false
    }
}

fn elaborate_element_ptr_input(rvsdg: &mut Rvsdg, node: Node) {
    let region = rvsdg[node].region();
    let data = rvsdg[node].expect_op_element_ptr();
    let ptr_input = *data.ptr_input();
    let index_input = *data.index_input();

    let get_offset_node = rvsdg.add_op_get_ptr_offset(region, ptr_input);
    let add_node = rvsdg.add_op_binary(
        region,
        BinaryOperator::Add,
        ValueInput::output(TY_U32, get_offset_node, 0),
        index_input,
    );

    rvsdg.reconnect_value_input(
        node,
        1,
        ValueOrigin::Output {
            producer: add_node,
            output: 0,
        },
    );
}

struct Collector {
    seen: FxHashSet<Node>,
    queue: Vec<Node>,
}

impl Collector {
    pub fn new() -> Self {
        Self {
            seen: Default::default(),
            queue: Vec::new(),
        }
    }

    pub fn collect(&mut self, rvsdg: &Rvsdg, region: Region) {
        self.queue.clear();
        self.seen.clear();
        self.visit_region(rvsdg, region);
    }
}

impl BottomUpVisitor for Collector {
    fn visit_node(&mut self, rvsdg: &Rvsdg, node: Node) {
        if let Simple(OpElementPtr(op)) = rvsdg[node].kind()
            && is_slice_ptr_ty(rvsdg.ty(), op.ptr_input().ty)
            && self.seen.insert(node)
        {
            self.queue.push(node);
        }

        visit_node_bottom_up(self, rvsdg, node);
    }
}

pub struct PtrOffsetElaborator {
    collector: Collector,
}

impl PtrOffsetElaborator {
    pub fn new() -> Self {
        PtrOffsetElaborator {
            collector: Collector::new(),
        }
    }

    pub fn elaborate_offset_slice_in_fn(&mut self, rvsdg: &mut Rvsdg, function: Function) {
        let body_region = rvsdg
            .get_function_node(function)
            .map(|n| rvsdg[n].expect_function().body_region());

        if let Some(body_region) = body_region {
            self.collector.collect(rvsdg, body_region);

            while let Some(node) = self.collector.queue.pop() {
                elaborate_element_ptr_input(rvsdg, node);
            }
        }
    }
}

pub fn transform_entry_points(module: &Module, rvsdg: &mut Rvsdg) {
    let mut elaborator = PtrOffsetElaborator::new();

    for (entry_point, _) in module.entry_points.iter() {
        elaborator.elaborate_offset_slice_in_fn(rvsdg, entry_point);
    }
}
