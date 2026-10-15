#[cfg(debug_assertions)]
use rustc_hash::FxHashSet;

use crate::rvsdg::{Connectivity, Node, NodeKind, Region, Rvsdg, SimpleNode, ValueOrigin};
use crate::ty::Type;

/// A value origin qualified by the region in which it is used.
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

    /// Traces identity-preserving connections to the value's canonical origin.
    ///
    /// Follows proxies, switch arguments, and directly invariant loop arguments, but never loop
    /// feedback or structural-node outputs. The returned value may belong to an ancestor region;
    /// it is not necessarily available for direct rewiring in this value's region.
    ///
    /// Uses `usize::MAX` as the transition budget. Debug builds panic if a cycle is detected,
    /// since this indicates an invalid RVSDG.
    pub fn canonicalize(self, rvsdg: &Rvsdg) -> Self {
        let mut remaining_steps = usize::MAX;

        self.canonicalize_with_budget(rvsdg, &mut remaining_steps)
            .expect("canonicalization should not exhaust usize::MAX transitions")
    }

    /// Canonicalizes using a caller-owned budget, consuming one unit per identity transition.
    ///
    /// Returns `None` if another transition is needed after the budget is exhausted. Debug builds
    /// use a visited-value set and panic if a cycle is detected before the budget is exhausted,
    /// since this indicates an invalid RVSDG.
    /// A terminal value requires no budget, so reaching it with exactly zero remaining succeeds.
    pub fn canonicalize_with_budget(
        mut self,
        rvsdg: &Rvsdg,
        remaining_steps: &mut usize,
    ) -> Option<Self> {
        #[cfg(debug_assertions)]
        let mut visited = FxHashSet::default();

        loop {
            let next = match self.origin {
                ValueOrigin::Output {
                    producer,
                    output: 0,
                } => {
                    if let NodeKind::Simple(SimpleNode::ValueProxy(proxy)) = rvsdg[producer].kind()
                    {
                        Self::new(self.region, proxy.value_inputs()[0].origin)
                    } else {
                        return Some(self);
                    }
                }
                ValueOrigin::Argument(argument) if self.region != rvsdg.global_region() => {
                    let owner = rvsdg[self.region].owner();
                    let res = argument as usize + 1;
                    let input = match rvsdg[owner].kind() {
                        NodeKind::Switch(switch) => switch.entry_inputs()[argument as usize],
                        NodeKind::Loop(loop_node)
                            if rvsdg[self.region].value_results()[res].origin == self.origin =>
                        {
                            loop_node.value_inputs()[argument as usize]
                        }
                        _ => return Some(self),
                    };

                    Self::new(rvsdg[owner].region(), input.origin)
                }
                _ => return Some(self),
            };

            if *remaining_steps == 0 {
                return None;
            }

            #[cfg(debug_assertions)]
            assert!(visited.insert(self), "RVSDG should not contain cycles");

            *remaining_steps -= 1;
            self = next;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::rvsdg::{ValueInput, ValueOutput};
    use crate::ty::{TY_DUMMY, TY_PREDICATE, TY_U32};
    use crate::{FnArg, FnSig, Function, Module, Symbol};

    #[test]
    fn canonicalization_with_budget() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref("test"),
            module: module.name,
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: function.name,
                ty: TY_DUMMY,
                args: vec![FnArg {
                    ty: TY_U32,
                    shader_io_binding: None,
                }],
                ret_ty: None,
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, body) = rvsdg.register_function(&module, function, iter::empty());

        let proxy = rvsdg.add_value_proxy(body, ValueInput::argument(TY_U32, 0));
        let proxy_2 = rvsdg.add_value_proxy(body, ValueInput::output(TY_U32, proxy, 0));

        let base_key = ValueKey::new(body, ValueOrigin::Argument(0));
        let proxy_2_key = ValueKey::new(
            body,
            ValueOrigin::Output {
                producer: proxy_2,
                output: 0,
            },
        );

        assert_eq!(proxy_2_key.ty(&rvsdg), TY_U32);
        assert_eq!(ValueKey::for_node_input(&rvsdg, proxy, 0), base_key);
        assert_eq!(proxy_2_key.canonicalize_with_budget(&rvsdg, &mut 0), None);
        assert_eq!(proxy_2_key.canonicalize_with_budget(&rvsdg, &mut 1), None);

        let mut budget = 2;

        assert_eq!(
            proxy_2_key.canonicalize_with_budget(&rvsdg, &mut budget),
            Some(base_key)
        );
        assert_eq!(budget, 0);
        assert_eq!(
            base_key.canonicalize_with_budget(&rvsdg, &mut budget),
            Some(base_key)
        );
        assert_eq!(budget, 0);
        assert_eq!(base_key.canonicalize(&rvsdg), base_key);
        assert_eq!(proxy_2_key.canonicalize(&rvsdg), base_key);
        assert_eq!(
            proxy_2_key.canonicalize_with_budget(&rvsdg, &mut 3),
            Some(base_key)
        );
    }

    #[test]
    fn resolves_through_switch_arguments() {
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
        let constant = rvsdg.add_const_u32(body, 7);
        let fallback = rvsdg.add_const_fallback(body, TY_U32);

        let switch = rvsdg.add_switch(
            body,
            vec![
                ValueInput::output(TY_PREDICATE, selector, 0),
                ValueInput::output(TY_U32, constant, 0),
                ValueInput::output(TY_U32, fallback, 0),
            ],
            vec![ValueOutput::new(TY_U32)],
            None,
        );
        let branch = rvsdg.add_switch_branch(switch);

        assert_eq!(
            ValueKey::new(branch, ValueOrigin::Argument(0)).canonicalize(&rvsdg),
            ValueKey::new(
                body,
                ValueOrigin::Output {
                    producer: constant,
                    output: 0,
                },
            )
        );
        assert_eq!(
            ValueKey::new(branch, ValueOrigin::Argument(1)).canonicalize(&rvsdg),
            ValueKey::new(
                body,
                ValueOrigin::Output {
                    producer: fallback,
                    output: 0,
                },
            )
        );
    }

    #[test]
    fn resolves_only_through_loop_invariant_arguments() {
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

        let constant = rvsdg.add_const_u32(body, 7);
        let fallback = rvsdg.add_const_fallback(body, TY_U32);

        let (_, loop_region) = rvsdg.add_loop(
            body,
            vec![
                ValueInput::output(TY_U32, constant, 0),
                ValueInput::output(TY_U32, fallback, 0),
            ],
            None,
        );
        let reentry = rvsdg.add_const_bool(loop_region, false);

        rvsdg.reconnect_region_result(
            loop_region,
            0,
            ValueOrigin::Output {
                producer: reentry,
                output: 0,
            },
        );

        // Both loop-values start out loop-invariant
        rvsdg.reconnect_region_result(loop_region, 1, ValueOrigin::Argument(0));
        rvsdg.reconnect_region_result(loop_region, 2, ValueOrigin::Argument(1));

        assert_eq!(
            ValueKey::new(loop_region, ValueOrigin::Argument(0)).canonicalize(&rvsdg),
            ValueKey::new(
                body,
                ValueOrigin::Output {
                    producer: constant,
                    output: 0,
                },
            )
        );
        assert_eq!(
            ValueKey::new(loop_region, ValueOrigin::Argument(1)).canonicalize(&rvsdg),
            ValueKey::new(
                body,
                ValueOrigin::Output {
                    producer: fallback,
                    output: 0,
                },
            )
        );

        // Change the second loop-value to no longer be loop-invariant
        rvsdg.reconnect_region_result(loop_region, 2, ValueOrigin::Argument(0));

        assert_eq!(
            ValueKey::new(loop_region, ValueOrigin::Argument(1)).canonicalize(&rvsdg),
            ValueKey::new(loop_region, ValueOrigin::Argument(1))
        );
    }
}
