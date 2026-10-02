use crate::rvsdg::{Connectivity, Node, NodeKind, Region, Rvsdg, SimpleNode, ValueOrigin};
use crate::ty::Type;

pub const MAX_CANONICALIZATION_STEPS: usize = 1024;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ValueKey {
    pub region: Region,
    pub origin: ValueOrigin,
}

impl ValueKey {
    pub fn new(region: Region, origin: ValueOrigin) -> Self {
        Self { region, origin }
    }

    pub fn for_node_input(rvsdg: &Rvsdg, node: Node, input: usize) -> Self {
        Self::new(
            rvsdg[node].region(),
            rvsdg[node].value_inputs()[input].origin,
        )
    }

    pub fn for_branch_result(rvsdg: &Rvsdg, switch: Node, output: u32, branch: usize) -> Self {
        let region = rvsdg[switch].expect_switch().branches()[branch];

        Self::new(
            region,
            rvsdg[region].value_results()[output as usize].origin,
        )
    }

    pub fn ty(self, rvsdg: &Rvsdg) -> Type {
        rvsdg.value_origin_ty(self.region, self.origin)
    }

    /// Trace a value up through switch arguments, loop-invariant loop-region argument, and through
    /// proxy nodes, to its "canonical" origin.
    pub fn canonicalize(mut self, rvsdg: &Rvsdg) -> Option<Self> {
        let mut remaining_steps = MAX_CANONICALIZATION_STEPS;

        while let Some(next) = route(rvsdg, self) {
            if remaining_steps == 0 {
                return None;
            }

            remaining_steps -= 1;
            self = next;
        }

        Some(self)
    }
}

/// One identity-preserving step. Never follows loop feedback or loop outputs.
/// Callers retain their query context when following this route.
pub fn route(rvsdg: &Rvsdg, key: ValueKey) -> Option<ValueKey> {
    match key.origin {
        ValueOrigin::Output {
            producer,
            output: 0,
        } => {
            if let NodeKind::Simple(SimpleNode::ValueProxy(proxy)) = rvsdg[producer].kind() {
                return Some(ValueKey::new(key.region, proxy.value_inputs()[0].origin));
            }
        }
        ValueOrigin::Argument(argument) if key.region != rvsdg.global_region() => {
            let owner = rvsdg[key.region].owner();
            let input = match rvsdg[owner].kind() {
                NodeKind::Switch(switch) => switch.value_inputs()[argument as usize + 1],
                NodeKind::Loop(loop_node)
                    if rvsdg[key.region].value_results()[argument as usize + 1].origin
                        == key.origin =>
                {
                    loop_node.value_inputs()[argument as usize]
                }
                _ => return None,
            };

            return Some(ValueKey::new(rvsdg[owner].region(), input.origin));
        }
        _ => {}
    }

    None
}

pub fn contains_region(rvsdg: &Rvsdg, ancestor: Region, mut region: Region) -> bool {
    loop {
        if region == ancestor {
            return true;
        }

        if region == rvsdg.global_region() {
            return false;
        }

        region = rvsdg[rvsdg[region].owner()].region();
    }
}
