//! Replaces fallback switch-branch results with a constant value when every other branch returns
//! the same scalar constant.
//!
//! The compiler is allowed to replace fallback values with any value of a matching type. We inspect
//! the region results that correspond to the same switch output. If:
//!
//! - At least one branch returns a scalar constant
//! - At least one branch returns a fallback value
//! - All other branches return either an identical constant value or a fallback value
//!
//! Then we add that same constant value to all branches in which the region result connects to a
//! fallback value and reconnect the region result to this new value.
//!
//! This allows other transforms to treat the switch output as constant and propagate the value for
//! simplification.

use crate::Module;
use crate::rvsdg::visit::region_nodes::RegionNodesVisitor;
use crate::rvsdg::{Connectivity, Node, NodeKind, Region, Rvsdg, SimpleNode, ValueOrigin, visit};

#[derive(Clone, Copy, PartialEq, Eq)]
enum ScalarConstant {
    U32(u32),
    I32(i32),
    F32(u32),
    Bool(bool),
    Predicate(u32),
}

impl ScalarConstant {
    fn from_node(rvsdg: &Rvsdg, node: Node) -> Option<Self> {
        match rvsdg[node].kind() {
            NodeKind::Simple(SimpleNode::ConstU32(value)) => Some(Self::U32(value.value())),
            NodeKind::Simple(SimpleNode::ConstI32(value)) => Some(Self::I32(value.value())),
            NodeKind::Simple(SimpleNode::ConstF32(value)) => {
                Some(Self::F32(value.value().to_bits()))
            }
            NodeKind::Simple(SimpleNode::ConstBool(value)) => Some(Self::Bool(value.value())),
            NodeKind::Simple(SimpleNode::ConstPredicate(value)) => {
                Some(Self::Predicate(value.value()))
            }
            _ => None,
        }
    }

    fn add_to_region(self, rvsdg: &mut Rvsdg, region: Region) -> Node {
        match self {
            Self::U32(value) => rvsdg.add_const_u32(region, value),
            Self::I32(value) => rvsdg.add_const_i32(region, value),
            Self::F32(value) => rvsdg.add_const_f32(region, f32::from_bits(value)),
            Self::Bool(value) => rvsdg.add_const_bool(region, value),
            Self::Predicate(value) => rvsdg.add_const_predicate(region, value),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BranchResult {
    Constant(ScalarConstant),
    Fallback,
}

impl BranchResult {
    fn classify(rvsdg: &Rvsdg, origin: ValueOrigin) -> Option<(Self, Node)> {
        let ValueOrigin::Output {
            producer,
            output: 0,
        } = origin
        else {
            return None;
        };

        if matches!(
            rvsdg[producer].kind(),
            NodeKind::Simple(SimpleNode::ConstFallback(_))
        ) {
            Some((Self::Fallback, producer))
        } else {
            ScalarConstant::from_node(rvsdg, producer)
                .map(Self::Constant)
                .map(|result| (result, producer))
        }
    }

    fn try_unify(&mut self, result: Self) -> bool {
        match (*self, result) {
            (Self::Fallback, result) => {
                *self = result;
                true
            }
            (Self::Constant(_), Self::Fallback) => true,
            (Self::Constant(common), Self::Constant(value)) => common == value,
        }
    }
}

struct SwitchNodeCollector<'a> {
    queue: &'a mut Vec<Node>,
}

impl RegionNodesVisitor for SwitchNodeCollector<'_> {
    fn visit_node(&mut self, rvsdg: &Rvsdg, node: Node) {
        if let NodeKind::Switch(_) = rvsdg[node].kind() {
            self.queue.push(node);
        }

        visit::region_nodes::visit_node(self, rvsdg, node);
    }
}

pub struct SwitchFallbackUnifier {
    switch_node_queue: Vec<Node>,
    switch_branches: Vec<Region>,
    fallback_results: Vec<(Region, Node)>,
}

impl SwitchFallbackUnifier {
    pub fn new() -> Self {
        Self {
            switch_node_queue: Vec::new(),
            switch_branches: Vec::new(),
            fallback_results: Vec::new(),
        }
    }

    pub fn process_region(&mut self, rvsdg: &mut Rvsdg, region: Region) -> bool {
        self.switch_node_queue.clear();

        SwitchNodeCollector {
            queue: &mut self.switch_node_queue,
        }
        .visit_region(rvsdg, region);

        let mut changed = false;

        while let Some(switch_node) = self.switch_node_queue.pop() {
            changed |= self.process_switch_node(rvsdg, switch_node);
        }

        changed
    }

    fn process_switch_node(&mut self, rvsdg: &mut Rvsdg, switch_node: Node) -> bool {
        self.switch_branches.clear();

        let switch = rvsdg[switch_node].expect_switch();
        let output_count = switch.value_outputs().len();

        self.switch_branches.extend_from_slice(switch.branches());

        let mut changed = false;

        for output in 0..output_count {
            self.fallback_results.clear();

            let mut common_result = BranchResult::Fallback;
            let mut unifiable = true;

            for &branch in &self.switch_branches {
                let origin = rvsdg[branch].value_results()[output].origin;

                let Some((branch_result, producer)) = BranchResult::classify(rvsdg, origin) else {
                    unifiable = false;

                    break;
                };

                if branch_result == BranchResult::Fallback {
                    self.fallback_results.push((branch, producer));
                }

                if !common_result.try_unify(branch_result) {
                    unifiable = false;

                    break;
                }
            }

            let BranchResult::Constant(common_constant) = common_result else {
                continue;
            };

            if !unifiable || self.fallback_results.is_empty() {
                continue;
            }

            for &(branch, fallback_node) in &self.fallback_results {
                let replacement_node = common_constant.add_to_region(rvsdg, branch);

                rvsdg.reconnect_region_result(
                    branch,
                    output as u32,
                    ValueOrigin::Output {
                        producer: replacement_node,
                        output: 0,
                    },
                );

                if rvsdg[fallback_node].value_outputs()[0].users.is_empty() {
                    rvsdg.remove_node(fallback_node);
                }
            }

            changed = true;
        }

        changed
    }
}

pub fn transform_entry_points(module: &Module, rvsdg: &mut Rvsdg) -> bool {
    let mut unifier = SwitchFallbackUnifier::new();
    let mut changed = false;

    for (entry_point, _) in module.entry_points.iter() {
        let fn_node = rvsdg
            .get_function_node(entry_point)
            .expect("function not registered");
        let body_region = rvsdg[fn_node].expect_function().body_region();

        changed |= unifier.process_region(rvsdg, body_region);
    }

    changed
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::rvsdg::{ValueInput, ValueOutput};
    use crate::ty::{TY_DUMMY, TY_PREDICATE, TY_U32};
    use crate::{FnSig, Function, Symbol};

    #[test]
    fn unifies_branch_results() {
        let mut result = BranchResult::Fallback;

        assert!(result.try_unify(BranchResult::Fallback));
        assert!(result.try_unify(BranchResult::Constant(ScalarConstant::U32(1))));
        assert!(result.try_unify(BranchResult::Fallback));
        assert!(result.try_unify(BranchResult::Constant(ScalarConstant::U32(1))));
        assert!(!result.try_unify(BranchResult::Constant(ScalarConstant::U32(2))));

        assert!(matches!(
            result,
            BranchResult::Constant(ScalarConstant::U32(1))
        ));
    }

    #[test]
    fn unifies_constant_and_fallback_results() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref("test"),
            module: Symbol::from_ref(""),
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
        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, body) = rvsdg.register_function(&module, function, iter::empty());

        let selector = rvsdg.add_const_predicate(body, 0);
        let switch = rvsdg.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, selector, 0)],
            vec![ValueOutput::new(TY_U32)],
            None,
        );
        let branch_0 = rvsdg.add_switch_branch(switch);
        let branch_1 = rvsdg.add_switch_branch(switch);

        let constant = rvsdg.add_const_u32(branch_0, 1);
        let fallback = rvsdg.add_const_fallback(branch_1, TY_U32);

        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: constant,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            branch_1,
            0,
            ValueOrigin::Output {
                producer: fallback,
                output: 0,
            },
        );

        let mut unifier = SwitchFallbackUnifier::new();

        assert!(unifier.process_region(&mut rvsdg, body));

        let ValueOrigin::Output {
            producer: replacement,
            output: 0,
        } = rvsdg[branch_1].value_results()[0].origin
        else {
            panic!("expected the fallback result to connect to a constant node");
        };

        assert_eq!(rvsdg[replacement].expect_const_u32().value(), 1);
        assert!(!rvsdg.is_live_node(fallback));
        assert!(!unifier.process_region(&mut rvsdg, body));
    }

    #[test]
    fn preserves_shared_fallback_node() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref("test"),
            module: Symbol::from_ref(""),
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
        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, body) = rvsdg.register_function(&module, function, iter::empty());

        let selector = rvsdg.add_const_predicate(body, 0);
        let switch = rvsdg.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, selector, 0)],
            vec![ValueOutput::new(TY_U32)],
            None,
        );
        let branch_0 = rvsdg.add_switch_branch(switch);
        let branch_1 = rvsdg.add_switch_branch(switch);

        let constant = rvsdg.add_const_u32(branch_0, 7);
        let fallback = rvsdg.add_const_fallback(branch_1, TY_U32);
        let proxy = rvsdg.add_value_proxy(branch_1, ValueInput::output(TY_U32, fallback, 0));

        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: constant,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            branch_1,
            0,
            ValueOrigin::Output {
                producer: fallback,
                output: 0,
            },
        );

        let mut unifier = SwitchFallbackUnifier::new();

        assert!(unifier.process_region(&mut rvsdg, body));

        assert!(rvsdg.is_live_node(fallback));
        assert_eq!(
            rvsdg[proxy].value_inputs()[0].origin,
            ValueOrigin::Output {
                producer: fallback,
                output: 0,
            }
        );

        let ValueOrigin::Output {
            producer: replacement,
            output: 0,
        } = rvsdg[branch_1].value_results()[0].origin
        else {
            panic!("expected the fallback result to connect to a constant node");
        };
        assert_eq!(rvsdg[replacement].expect_const_u32().value(), 7);
    }

    #[test]
    fn rejects_non_unifiable_result_sets() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref("test"),
            module: Symbol::from_ref(""),
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
        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, body) = rvsdg.register_function(&module, function, iter::empty());

        let selector = rvsdg.add_const_predicate(body, 0);
        let switch = rvsdg.add_switch(
            body,
            vec![ValueInput::output(TY_PREDICATE, selector, 0)],
            vec![ValueOutput::new(TY_U32)],
            None,
        );
        let branch_0 = rvsdg.add_switch_branch(switch);
        let branch_1 = rvsdg.add_switch_branch(switch);
        let branch_2 = rvsdg.add_switch_branch(switch);

        let zero = rvsdg.add_const_u32(branch_0, 0);
        let one = rvsdg.add_const_u32(branch_1, 1);
        let fallback = rvsdg.add_const_fallback(branch_2, TY_U32);

        rvsdg.reconnect_region_result(
            branch_0,
            0,
            ValueOrigin::Output {
                producer: zero,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            branch_1,
            0,
            ValueOrigin::Output {
                producer: one,
                output: 0,
            },
        );
        rvsdg.reconnect_region_result(
            branch_2,
            0,
            ValueOrigin::Output {
                producer: fallback,
                output: 0,
            },
        );

        let mut unifier = SwitchFallbackUnifier::new();

        assert!(!unifier.process_region(&mut rvsdg, body));

        assert!(rvsdg.is_live_node(fallback));
        assert_eq!(
            rvsdg[branch_2].value_results()[0].origin,
            ValueOrigin::Output {
                producer: fallback,
                output: 0,
            }
        );
    }
}
