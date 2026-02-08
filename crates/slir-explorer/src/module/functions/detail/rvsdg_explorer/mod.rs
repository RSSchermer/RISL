use std::sync::Arc;

use leptos::prelude::*;
use slir::rvsdg::{Region as RvsdgRegion, Rvsdg};
use slir::Function;

use crate::module::functions::detail::detail::use_function;
use crate::module::functions::detail::rvsdg_explorer::layout::{Config, RegionLayout};
use crate::module::functions::detail::rvsdg_explorer::region::Region;
use crate::module::url::function_url;
use crate::module::use_module_data;

mod connector;
mod layout;
mod node;
mod region;

#[derive(Clone)]
pub struct RegionViewerContext {
    pub rvsdg: StoredValue<Rvsdg>,
    pub gen_function_url: Arc<dyn Fn(Function) -> String + Send + Sync>,
}

pub fn use_region_viewer_context() -> RegionViewerContext {
    use_context::<RegionViewerContext>().expect("RegionViewerContext not provided")
}

#[derive(Copy, Clone)]
pub enum RvsdgStage {
    Initial,
    Transformed,
}

#[component]
pub fn RvsdgRegionViewer(
    rvsdg: StoredValue<Rvsdg>,
    region: RvsdgRegion,
    gen_function_url: impl Fn(Function) -> String + Send + Sync + 'static,
) -> impl IntoView {
    let gen_function_url = Arc::new(gen_function_url);

    provide_context(RegionViewerContext {
        rvsdg,
        gen_function_url: gen_function_url.clone(),
    });

    let region_layout = RegionLayout::generate(&Config::default(), &rvsdg.read_value(), region);

    let width = region_layout.rect().size[0] + 10.0;
    let height = region_layout.rect().size[1] + 30.0;

    view! {
        <svg xmlns="http://www.w3.org/2000/svg" width=width height=height>
            <style type="text/css">
                r#"
                text {
                    font-family: Courier New,Courier,Lucida Sans Typewriter,Lucida Typewriter,monospace;
                    font-size: 15px;
                    line-height: 15px;
                }
                
                a text {
                    fill: #4085f5;
                    font-weight: bold;
                }
                
                .region-rect {
                    stroke-width: 2px;
                    stroke: black;
                    fill: white;
                }
                
                .node-rect {
                    stroke-width: 2px;
                    stroke: black;
                }

                .node-rect.simple {
                    fill: #faf5bb;
                }

                .node-rect.loop {
                    fill: #edd3da;
                }

                .node-rect.switch {
                    fill: #baf7c9;
                }

                .node-content-container .node-tooltip rect {
                    stroke_width: 1px;
                    stroke: black;
                    fill: white;
                }

                .node-content-container .node-tooltip {
                    display: none;
                }

                .node-container:hover > .node-content-container > .node-tooltip {
                    display: block;
                    z-index: 100;
                }

                .connector .tooltip rect {
                    stroke_width: 1px;
                    stroke: black;
                    fill: white;
                }
                
                .connector .tooltip {
                    display: none;
                }
                
                .connector:hover .tooltip {
                    display: block;
                    z-index: 100;
                }

                .connector-rect {
                    fill: black;
                }

                .edge-lines {
                    stroke-linecap: round;
                }

                .edge-lines.state-edge {
                    stroke-dasharray: 5, 5;
                }
                
                .edge-lines .visible-line {
                    stroke-width: 2.0px;
                    stroke: black;
                    fill: none;
                }
                
                .edge-lines .hover-target {
                    stroke-width: 10.0px;
                    stroke: transparent;
                    fill: none;
                }

                .edge-lines:hover {
                    cursor: pointer;
                }
                
                .edge-lines:hover .visible-line {
                    stroke-width: 4.0px;
                    stroke: #fcba03;
                }

                .edge-lines.highlighted .visible-line {
                    stroke-width: 4.0px;
                    stroke: #fae17d;
                }
                "#
            </style>
            <g transform="translate(5, 5)">
                <Region region_layout />
            </g>
        </svg>
    }
}

#[component]
pub fn RvsdgExplorer(stage: RvsdgStage) -> impl IntoView {
    let module_data = use_module_data();
    let function = use_function();
    let rvsdg = match stage {
        RvsdgStage::Initial => module_data.rvsdg_initial,
        RvsdgStage::Transformed => module_data.rvsdg_transformed,
    };

    let region = Memo::new(move |_| {
        rvsdg.and_then(|rvsdg| {
            let rvsdg = rvsdg.read_value();

            rvsdg
                .get_function_node(function)
                .map(|node| rvsdg[node].expect_function().body_region())
        })
    });

    view! {
        {move || {
            if let (Some(rvsdg), Some(region)) = (rvsdg, region()) {
                let module_name = module_data.module.read_value().name;
                let gen_function_url = move |f| function_url(module_name, f);

                view! {
                    <RvsdgRegionViewer
                        rvsdg
                        region
                        gen_function_url
                    />
                }.into_any()
            } else {
                view! {
                    <div class="info-page-container">
                        <p>"No RVSDG for the current function."</p>
                    </div>
                }.into_any()
            }
        }}
    }
}
