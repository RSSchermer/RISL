use crate::rvsdg::{Connectivity, Node, NodeKind, Region, Rvsdg, ValueOrigin};

pub trait ReverseValueFlowVisitor: Sized {
    /// Returns a boolean that indicates whether to visit the given `user`.
    fn should_visit(&mut self, region: Region, origin: ValueOrigin) -> bool;

    fn visit_value_input(&mut self, rvsdg: &Rvsdg, node: Node, input: u32) {
        visit_value_input(self, rvsdg, node, input)
    }

    fn visit_value_origin(&mut self, rvsdg: &Rvsdg, region: Region, origin: ValueOrigin) {
        visit_value_origin(self, rvsdg, region, origin)
    }

    fn visit_region_argument(&mut self, rvsdg: &Rvsdg, region: Region, argument: u32) {
        visit_region_argument(self, rvsdg, region, argument)
    }

    fn visit_value_output(&mut self, rvsdg: &Rvsdg, node: Node, output: u32) {
        visit_value_output(self, rvsdg, node, output)
    }

    fn visit_region_result(&mut self, rvsdg: &Rvsdg, region: Region, result: u32) {
        visit_region_result(self, rvsdg, region, result)
    }
}

pub fn visit_value_input<V: ReverseValueFlowVisitor>(
    visitor: &mut V,
    rvsdg: &Rvsdg,
    node: Node,
    input: u32,
) {
    let region = rvsdg[node].region();
    let origin = rvsdg[node].value_inputs()[input as usize].origin;

    visitor.visit_value_origin(rvsdg, region, origin);
}

pub fn visit_value_origin<V: ReverseValueFlowVisitor>(
    visitor: &mut V,
    rvsdg: &Rvsdg,
    region: Region,
    origin: ValueOrigin,
) {
    if visitor.should_visit(region, origin) {
        match origin {
            ValueOrigin::Argument(argument) => {
                visitor.visit_region_argument(rvsdg, region, argument)
            }
            ValueOrigin::Output { producer, output } => {
                visitor.visit_value_output(rvsdg, producer, output)
            }
        }
    }
}

pub fn visit_region_argument<V: ReverseValueFlowVisitor>(
    visitor: &mut V,
    rvsdg: &Rvsdg,
    region: Region,
    argument: u32,
) {
    let owner = rvsdg[region].owner();

    match rvsdg[owner].kind() {
        NodeKind::Switch(_) => {
            let input = argument + 1;

            visitor.visit_value_input(rvsdg, owner, input);
        }
        NodeKind::Loop(_) => visitor.visit_value_input(rvsdg, owner, argument),
        NodeKind::Function(_) => {
            // Do nothing, we're done...
        }
        _ => unreachable!("node kind cannot own a region"),
    }
}

pub fn visit_value_output<V: ReverseValueFlowVisitor>(
    visitor: &mut V,
    rvsdg: &Rvsdg,
    node: Node,
    output: u32,
) {
    match rvsdg[node].kind() {
        NodeKind::Switch(n) => {
            for branch in n.branches() {
                visitor.visit_region_result(rvsdg, *branch, output);
            }
        }
        NodeKind::Loop(n) => {
            let result = output + 1;

            visitor.visit_region_result(rvsdg, n.loop_region(), result);
        }
        NodeKind::Simple(n) => {
            let value_input_count = n.value_inputs().len() as u32;

            for i in 0..value_input_count {
                visitor.visit_value_input(rvsdg, node, i);
            }
        }
        _ => panic!("reverse value-flow visitor should not be used in the global region"),
    }
}

pub fn visit_region_result<V: ReverseValueFlowVisitor>(
    visitor: &mut V,
    rvsdg: &Rvsdg,
    region: Region,
    result: u32,
) {
    let origin = rvsdg[region].value_results()[result as usize].origin;

    visitor.visit_value_origin(rvsdg, region, origin);
}
