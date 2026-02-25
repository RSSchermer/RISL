use crate::rvsdg::Rvsdg;
use crate::rvsdg::transform::memory_promotion_and_legalization::MemoryPromoterLegalizer;
use crate::rvsdg::transform::proxy_node_elimination::region_eliminate_proxy_nodes;
use crate::rvsdg::transform::scalar_replacement::AggregateReplacementContext;
use crate::rvsdg::transform::store_coalescing::region_coalesce_store_ops;
use crate::{Function, Module};

pub struct MemoryTransformer {
    aggregate_replacement_cx: AggregateReplacementContext,
    promoter_legalizer: MemoryPromoterLegalizer,
}

impl MemoryTransformer {
    pub fn new() -> Self {
        MemoryTransformer {
            aggregate_replacement_cx: AggregateReplacementContext::new(),
            promoter_legalizer: MemoryPromoterLegalizer::new(),
        }
    }

    pub fn transform_fn(&mut self, rvsdg: &mut Rvsdg, function: Function) {
        let node = rvsdg
            .get_function_node(function)
            .expect("function should have RVSDG body");
        let body_region = rvsdg[node].expect_function().body_region();

        let mut region_replacement_cx =
            self.aggregate_replacement_cx.for_region(rvsdg, body_region);

        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 32;
        let mut did_transform = false;

        loop {
            if iterations >= MAX_ITERATIONS {
                break;
            }

            // We run the promoter-legalizer at least once (on the first iteration) and then as
            // often as we keep replacing aggregate alloca nodes.
            if iterations == 0 || did_transform {
                did_transform = false;

                did_transform |= region_replacement_cx.replace(rvsdg);

                region_eliminate_proxy_nodes(rvsdg, body_region);

                did_transform |= region_coalesce_store_ops(rvsdg, body_region);

                self.promoter_legalizer
                    .promote_and_legalize(rvsdg, body_region);

                iterations += 1;
            } else {
                break;
            }
        }
    }
}

pub fn transform_entry_points(module: &Module, rvsdg: &mut Rvsdg) {
    let mut transformer = MemoryTransformer::new();

    for (entry_point, _) in module.entry_points.iter() {
        transformer.transform_fn(rvsdg, entry_point);
    }
}
