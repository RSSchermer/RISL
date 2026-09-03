use rustc_hash::FxHashSet;

use crate::rvsdg::analyse::scalar_constant::ScalarConstant;
use crate::rvsdg::{Connectivity, NodeKind, Region, Rvsdg, SimpleNode, ValueOrigin};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ResolvedValue {
    Constant(ScalarConstant),
    Fallback,
    Opaque { region: Region, origin: ValueOrigin },
}

pub struct ValueResolver {
    visited: FxHashSet<(Region, ValueOrigin)>,
}

impl ValueResolver {
    pub fn new() -> Self {
        Self {
            visited: FxHashSet::default(),
        }
    }

    pub fn resolve(
        &mut self,
        rvsdg: &Rvsdg,
        mut region: Region,
        mut origin: ValueOrigin,
    ) -> ResolvedValue {
        self.visited.clear();

        loop {
            if !self.visited.insert((region, origin)) {
                panic!("RVSDG should not contain cycles");
            }

            match origin {
                ValueOrigin::Output { producer, output } => {
                    if let Some(constant) = ScalarConstant::from_node(rvsdg, producer, output) {
                        return ResolvedValue::Constant(constant);
                    }

                    if output == 0 {
                        match rvsdg[producer].kind() {
                            NodeKind::Simple(SimpleNode::ConstFallback(_)) => {
                                return ResolvedValue::Fallback;
                            }

                            // If we encounter a value-proxy node, we trace through it to the
                            // proxied value.
                            NodeKind::Simple(SimpleNode::ValueProxy(proxy)) => {
                                region = rvsdg[producer].region();
                                origin = proxy.value_inputs()[0].origin;

                                continue;
                            }
                            _ => {}
                        }
                    }

                    // Value is not a scalar constant or fallback, so resolve it as "opaque".
                    return ResolvedValue::Opaque { region, origin };
                }
                ValueOrigin::Argument(argument) if region != rvsdg.global_region() => {
                    let owner = rvsdg[region].owner();
                    let outer_region = rvsdg[owner].region();

                    match rvsdg[owner].kind() {
                        NodeKind::Switch(switch) => {
                            origin = switch.value_inputs()[argument as usize + 1].origin;
                            region = outer_region;
                        }
                        NodeKind::Loop(loop_node) => {
                            let loop_region = loop_node.loop_region();
                            let result = argument as usize + 1;

                            if rvsdg[loop_region].value_results()[result].origin
                                != ValueOrigin::Argument(argument)
                            {
                                // Loop value is not loop-invariant, so conservatively resolve it as
                                // "opaque".
                                return ResolvedValue::Opaque { region, origin };
                            }

                            origin = loop_node.value_inputs()[argument as usize].origin;
                            region = outer_region;
                        }
                        NodeKind::Function(_) => return ResolvedValue::Opaque { region, origin },
                        _ => panic!("node kind cannot own a region"),
                    }
                }
                _ => return ResolvedValue::Opaque { region, origin },
            }
        }
    }
}

impl Default for ValueResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::*;
    use crate::rvsdg::{ValueInput, ValueOutput};
    use crate::ty::{TY_DUMMY, TY_PREDICATE, TY_U32};
    use crate::{FnSig, Function, Module, Symbol};

    #[test]
    fn resolves_scalar_constants_and_fallback() {
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

        let u32_constant = rvsdg.add_const_u32(body, 7);
        let i32_constant = rvsdg.add_const_i32(body, -3);
        let f32_constant = rvsdg.add_const_f32(body, 1.25);
        let bool_constant = rvsdg.add_const_bool(body, true);
        let predicate_constant = rvsdg.add_const_predicate(body, 2);
        let fallback = rvsdg.add_const_fallback(body, TY_U32);

        let mut resolver = ValueResolver::new();

        assert_eq!(
            resolver.resolve(
                &rvsdg,
                body,
                ValueOrigin::Output {
                    producer: u32_constant,
                    output: 0,
                },
            ),
            ResolvedValue::Constant(ScalarConstant::U32(7))
        );
        assert_eq!(
            resolver.resolve(
                &rvsdg,
                body,
                ValueOrigin::Output {
                    producer: i32_constant,
                    output: 0,
                },
            ),
            ResolvedValue::Constant(ScalarConstant::I32(-3))
        );
        assert_eq!(
            resolver.resolve(
                &rvsdg,
                body,
                ValueOrigin::Output {
                    producer: f32_constant,
                    output: 0,
                },
            ),
            ResolvedValue::Constant(ScalarConstant::F32(1.25_f32.to_bits()))
        );
        assert_eq!(
            resolver.resolve(
                &rvsdg,
                body,
                ValueOrigin::Output {
                    producer: bool_constant,
                    output: 0,
                },
            ),
            ResolvedValue::Constant(ScalarConstant::Bool(true))
        );
        assert_eq!(
            resolver.resolve(
                &rvsdg,
                body,
                ValueOrigin::Output {
                    producer: predicate_constant,
                    output: 0,
                },
            ),
            ResolvedValue::Constant(ScalarConstant::Predicate(2))
        );
        assert_eq!(
            resolver.resolve(
                &rvsdg,
                body,
                ValueOrigin::Output {
                    producer: fallback,
                    output: 0,
                },
            ),
            ResolvedValue::Fallback
        );
    }

    #[test]
    fn resolves_through_proxies() {
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
        let constant_proxy = rvsdg.add_value_proxy(body, ValueInput::output(TY_U32, constant, 0));
        let constant_proxy_2 =
            rvsdg.add_value_proxy(body, ValueInput::output(TY_U32, constant_proxy, 0));
        let fallback = rvsdg.add_const_fallback(body, TY_U32);
        let fallback_proxy = rvsdg.add_value_proxy(body, ValueInput::output(TY_U32, fallback, 0));

        let mut resolver = ValueResolver::new();

        assert_eq!(
            resolver.resolve(
                &rvsdg,
                body,
                ValueOrigin::Output {
                    producer: constant_proxy_2,
                    output: 0,
                },
            ),
            ResolvedValue::Constant(ScalarConstant::U32(7))
        );
        assert_eq!(
            resolver.resolve(
                &rvsdg,
                body,
                ValueOrigin::Output {
                    producer: fallback_proxy,
                    output: 0,
                },
            ),
            ResolvedValue::Fallback
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

        let mut resolver = ValueResolver::new();

        assert_eq!(
            resolver.resolve(&rvsdg, branch, ValueOrigin::Argument(0)),
            ResolvedValue::Constant(ScalarConstant::U32(7))
        );
        assert_eq!(
            resolver.resolve(&rvsdg, branch, ValueOrigin::Argument(1)),
            ResolvedValue::Fallback
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

        let mut resolver = ValueResolver::new();

        assert_eq!(
            resolver.resolve(&rvsdg, loop_region, ValueOrigin::Argument(0)),
            ResolvedValue::Constant(ScalarConstant::U32(7))
        );
        assert_eq!(
            resolver.resolve(&rvsdg, loop_region, ValueOrigin::Argument(1)),
            ResolvedValue::Fallback
        );

        // Change the second loop-value to no longer be loop-invariant
        rvsdg.reconnect_region_result(loop_region, 2, ValueOrigin::Argument(0));

        assert_eq!(
            resolver.resolve(&rvsdg, loop_region, ValueOrigin::Argument(1)),
            ResolvedValue::Opaque {
                region: loop_region,
                origin: ValueOrigin::Argument(1),
            }
        );
    }

    #[test]
    #[should_panic(expected = "RVSDG should not contain cycles")]
    fn panics_on_rvsdg_cycle() {
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
        let cycle_0 = rvsdg.add_value_proxy(body, ValueInput::output(TY_U32, constant, 0));
        let cycle_1 = rvsdg.add_value_proxy(body, ValueInput::output(TY_U32, cycle_0, 0));

        // Construct a cycle between cycle_0 and cycle_1
        rvsdg.reconnect_value_input(
            cycle_0,
            0,
            ValueOrigin::Output {
                producer: cycle_1,
                output: 0,
            },
        );

        let mut resolver = ValueResolver::new();

        resolver.resolve(
            &rvsdg,
            body,
            ValueOrigin::Output {
                producer: cycle_0,
                output: 0,
            },
        );
    }
}
