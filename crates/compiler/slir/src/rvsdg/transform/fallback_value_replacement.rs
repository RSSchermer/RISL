use crate::Module;
use crate::rvsdg::visit::region_nodes::{self, RegionNodesVisitor, visit_region};
use crate::rvsdg::{Node, NodeKind, Region, Rvsdg, SimpleNode, ValueOrigin};
use crate::ty::{TY_BOOL, TY_F32, TY_I32, TY_U32};

pub struct FallbackValueReplacer {
    queue: Vec<(Node, Region)>,
}

impl FallbackValueReplacer {
    pub fn new() -> Self {
        Self { queue: Vec::new() }
    }

    pub fn transform_region(&mut self, rvsdg: &mut Rvsdg, region: Region) {
        self.queue.clear();

        visit_region(self, rvsdg, region);

        for (node, region) in self.queue.drain(..) {
            let ty = match rvsdg[node].expect_simple() {
                SimpleNode::ConstFallback(c) => c.ty(),
                _ => unreachable!(),
            };

            let replacement_node = match ty {
                TY_U32 => rvsdg.add_const_u32(region, 0),
                TY_I32 => rvsdg.add_const_i32(region, 0),
                TY_F32 => rvsdg.add_const_f32(region, 0.0),
                TY_BOOL => rvsdg.add_const_bool(region, false),
                _ => panic!(
                    "unsupported ConstFallback type `{:?}` for node `{:?}",
                    ty, node
                ),
            };

            rvsdg.reconnect_value_users(
                region,
                ValueOrigin::Output {
                    producer: node,
                    output: 0,
                },
                ValueOrigin::Output {
                    producer: replacement_node,
                    output: 0,
                },
            );

            rvsdg.remove_node(node);
        }
    }
}

impl RegionNodesVisitor for FallbackValueReplacer {
    fn visit_node(&mut self, rvsdg: &Rvsdg, node: Node) {
        let node_data = &rvsdg[node];

        if let NodeKind::Simple(SimpleNode::ConstFallback(_)) = node_data.kind() {
            self.queue.push((node, node_data.region()));
        }

        region_nodes::visit_node(self, rvsdg, node);
    }
}

pub fn transform_entry_points(module: &Module, rvsdg: &mut Rvsdg) {
    let mut transformer = FallbackValueReplacer::new();

    for (entry_point, _) in module.entry_points.iter() {
        let fn_node = rvsdg
            .get_function_node(entry_point)
            .expect("function not registered");
        let body_region = rvsdg[fn_node].expect_function().body_region();

        transformer.transform_region(rvsdg, body_region);
    }
}

#[cfg(test)]
mod tests {
    use std::iter;

    use super::FallbackValueReplacer;
    use crate::rvsdg::{Connectivity, Rvsdg, ValueInput, ValueOrigin};
    use crate::ty::{TY_BOOL, TY_DUMMY, TY_F32, TY_I32, TY_U32};
    use crate::{FnArg, FnSig, Function, Module, Symbol};

    #[test]
    fn test_fallback_value_replacement() {
        let mut module = Module::new(Symbol::from_ref(""));
        let function = Function {
            name: Symbol::from_ref("test"),
            module: Symbol::from_ref(""),
        };

        module.fn_sigs.register(
            function,
            FnSig {
                name: Default::default(),
                ty: TY_DUMMY,
                args: vec![FnArg {
                    ty: TY_U32,
                    shader_io_binding: None,
                }],
                ret_ty: Some(TY_U32),
            },
        );

        let mut rvsdg = Rvsdg::new(module.ty.clone());
        let (_, region) = rvsdg.register_function(&module, function, iter::empty());

        let fb_u32 = rvsdg.add_const_fallback(region, TY_U32);
        let proxy_u32 = rvsdg.add_value_proxy(region, ValueInput::output(TY_U32, fb_u32, 0));

        let fb_i32 = rvsdg.add_const_fallback(region, TY_I32);
        let proxy_i32 = rvsdg.add_value_proxy(region, ValueInput::output(TY_I32, fb_i32, 0));

        let fb_f32 = rvsdg.add_const_fallback(region, TY_F32);
        let proxy_f32 = rvsdg.add_value_proxy(region, ValueInput::output(TY_F32, fb_f32, 0));

        let fb_bool = rvsdg.add_const_fallback(region, TY_BOOL);
        let proxy_bool = rvsdg.add_value_proxy(region, ValueInput::output(TY_BOOL, fb_bool, 0));

        let mut transformer = FallbackValueReplacer::new();
        transformer.transform_region(&mut rvsdg, region);

        // Verify that ValueProxy node inputs connect to the new constant nodes.
        let ValueOrigin::Output {
            producer: prod_u32,
            output: 0,
        } = rvsdg[proxy_u32].value_inputs()[0].origin
        else {
            panic!("proxy_u32 should connect to a node output");
        };
        let prod_u32_data = rvsdg[prod_u32].expect_const_u32();
        assert_eq!(prod_u32_data.value(), 0);

        let ValueOrigin::Output {
            producer: prod_i32,
            output: 0,
        } = rvsdg[proxy_i32].value_inputs()[0].origin
        else {
            panic!("proxy_i32 should connect to a node output");
        };
        let prod_i32_data = rvsdg[prod_i32].expect_const_i32();
        assert_eq!(prod_i32_data.value(), 0);

        let ValueOrigin::Output {
            producer: prod_f32,
            output: 0,
        } = rvsdg[proxy_f32].value_inputs()[0].origin
        else {
            panic!("proxy_f32 should connect to a node output");
        };
        let prod_f32_data = rvsdg[prod_f32].expect_const_f32();
        assert_eq!(prod_f32_data.value(), 0.0);

        let ValueOrigin::Output {
            producer: prod_bool,
            output: 0,
        } = rvsdg[proxy_bool].value_inputs()[0].origin
        else {
            panic!("proxy_bool should connect to a node output");
        };
        let prod_bool_data = rvsdg[prod_bool].expect_const_bool();
        assert_eq!(prod_bool_data.value(), false);

        // Verify that original ConstFallback nodes are removed
        assert!(!rvsdg.is_live_node(fb_u32));
        assert!(!rvsdg.is_live_node(fb_i32));
        assert!(!rvsdg.is_live_node(fb_f32));
        assert!(!rvsdg.is_live_node(fb_bool));
    }
}
