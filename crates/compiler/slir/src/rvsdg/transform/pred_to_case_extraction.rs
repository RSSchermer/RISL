use crate::rvsdg::visit::region_nodes::RegionNodesVisitor;
use crate::rvsdg::{Connectivity, Node, NodeKind, Rvsdg, SimpleNode, ValueOrigin, visit};
use crate::{Function, Module};

struct NodeCollector<'a> {
    candidates: &'a mut Vec<Node>,
}

impl RegionNodesVisitor for NodeCollector<'_> {
    fn visit_node(&mut self, rvsdg: &Rvsdg, node: Node) {
        use NodeKind::*;

        match rvsdg[node].kind() {
            Switch(_) => self.candidates.push(node),
            _ => (),
        }

        visit::region_nodes::visit_node(self, rvsdg, node);
    }
}

fn try_extract_pred_to_case(rvsdg: &mut Rvsdg, switch_node: Node) {
    use NodeKind::*;
    use SimpleNode::*;

    let region = rvsdg[switch_node].region();
    let data = rvsdg[switch_node].expect_switch();
    let predicate_input = data.value_inputs()[0];
    let output_count = data.value_outputs().len();
    let branch_count = data.branches().len();

    let mut cases = Vec::with_capacity(branch_count);

    for output in 0..output_count {
        cases.clear();

        let data = rvsdg[switch_node].expect_switch();

        for branch in data.branches() {
            if let ValueOrigin::Output {
                producer,
                output: 0,
            } = rvsdg[*branch].value_results()[output].origin
                && let Simple(ConstU32(n)) = rvsdg[producer].kind()
            {
                cases.push(n.value());
            }
        }

        if cases.len() == branch_count {
            let cases = cases.clone();
            let pred_to_case = rvsdg.add_op_branch_selector_to_case(region, predicate_input, cases);

            rvsdg.reconnect_value_users(
                region,
                ValueOrigin::Output {
                    producer: switch_node,
                    output: output as u32,
                },
                ValueOrigin::Output {
                    producer: pred_to_case,
                    output: 0,
                },
            );
        }
    }
}

pub struct PredToCaseExtractor {
    candidates: Vec<Node>,
}

impl PredToCaseExtractor {
    pub fn new() -> Self {
        Self {
            candidates: Vec::new(),
        }
    }

    pub fn extract_in_fn(&mut self, rvsdg: &mut Rvsdg, function: Function) {
        let fn_node = rvsdg
            .get_function_node(function)
            .expect("function not registered");
        let mut collector = NodeCollector {
            candidates: &mut self.candidates,
        };

        collector.visit_node(rvsdg, fn_node);

        for node in self.candidates.drain(..) {
            try_extract_pred_to_case(rvsdg, node);
        }
    }
}

pub fn transform_entry_points(module: &Module, rvsdg: &mut Rvsdg) {
    let mut extractor = PredToCaseExtractor::new();

    for (function, _) in module.entry_points.iter() {
        extractor.extract_in_fn(rvsdg, function);
    }
}
