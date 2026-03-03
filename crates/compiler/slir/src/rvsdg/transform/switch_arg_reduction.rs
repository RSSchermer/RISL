use crate::rvsdg::visit::region_nodes::RegionNodesVisitor;
use crate::rvsdg::{Node, NodeKind, Rvsdg, ValueOrigin, visit};
use crate::{Function, Module};

// TODO: for the time being, this pass only does a very simple thing: it tests if any of a switch
// node's entry inputs are the same as its predicate input. If that's the case, then it replaces
// the branch arguments that correspond to the entry input with constant predicates. This handles
// a very common code pattern that earlier transforms produce for the legalization of "enum" values.
// However, this pass can potentially be extended to perform a "virtual" constant
// folding/propagation pass, where the predicate value is first up-propagated, then down-propagated
// to the switch entry inputs; we don't modify any nodes outside the switch (as the predicate is
// indeterminate outside the switch), but we can propagate constant values the switch's branch
// arguments, as inside each branch the predicate value can be determined (e.g.: in the first
// branch the predicate value must be `0`, in the second branch the predicate value must be `1`,
// etc.).

struct Job {
    switch_node: Node,
    entry_input: u32,
}

struct JobCollector<'a> {
    jobs: &'a mut Vec<Job>,
}

impl RegionNodesVisitor for JobCollector<'_> {
    fn visit_node(&mut self, rvsdg: &Rvsdg, node: Node) {
        if let NodeKind::Switch(switch_node) = rvsdg[node].kind() {
            let predicate_origin = switch_node.branch_selector().origin;

            for (i, entry) in switch_node.entry_inputs().iter().enumerate() {
                if entry.origin == predicate_origin {
                    self.jobs.push(Job {
                        switch_node: node,
                        entry_input: i as u32,
                    });
                }
            }
        }

        visit::region_nodes::visit_node(self, rvsdg, node);
    }
}

fn replace_switch_arg(rvsdg: &mut Rvsdg, switch_node: Node, entry_input: u32) {
    let branch_count = rvsdg[switch_node].expect_switch().branches().len();

    for b in 0..branch_count {
        let branch = rvsdg[switch_node].expect_switch().branches()[b];
        let replacement = rvsdg.add_const_predicate(branch, b as u32);

        rvsdg.reconnect_value_users(
            branch,
            ValueOrigin::Argument(entry_input),
            ValueOrigin::Output {
                producer: replacement,
                output: 0,
            },
        );
    }
}

pub struct SwitchArgReducer {
    queue: Vec<Job>,
}

impl SwitchArgReducer {
    pub fn new() -> Self {
        Self { queue: Vec::new() }
    }

    pub fn reduce_in_fn(&mut self, rvsdg: &mut Rvsdg, function: Function) {
        let fn_node = rvsdg
            .get_function_node(function)
            .expect("function not registered");

        let mut collector = JobCollector {
            jobs: &mut self.queue,
        };

        collector.visit_node(rvsdg, fn_node);

        while let Some(job) = self.queue.pop() {
            replace_switch_arg(rvsdg, job.switch_node, job.entry_input);
        }
    }
}

pub fn transform_entry_points(module: &Module, rvsdg: &mut Rvsdg) {
    let mut reducer = SwitchArgReducer::new();

    for (entry_point, _) in module.entry_points.iter() {
        reducer.reduce_in_fn(rvsdg, entry_point);
    }
}
