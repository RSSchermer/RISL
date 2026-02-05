use leptos::prelude::*;

use crate::module::functions::detail::detail::use_function;
use crate::module::functions::detail::rvsdg_explorer::layout::{Config, RegionLayout};
use crate::module::functions::detail::rvsdg_explorer::region::Region;
use crate::module::use_module_data;

mod connector;
mod layout;
mod node;
mod region;

pub enum RvsdgStage {
    Initial,
    Transformed,
}

#[component]
pub fn RvsdgExplorer(stage: RvsdgStage) -> impl IntoView {
    let module_data = use_module_data();
    let function = use_function();

    let region_layout = Memo::new(move |_| {
        let module_data = module_data.read_value();

        let rvsdg = match stage {
            RvsdgStage::Initial => module_data.rvsdg_initial.as_ref(),
            RvsdgStage::Transformed => module_data.rvsdg_transformed.as_ref(),
        };

        rvsdg.and_then(|rvsdg| {
            rvsdg.get_function_node(function).map(|node| {
                let region = rvsdg[node].expect_function().body_region();

                RegionLayout::generate(&Config::default(), rvsdg, region)
            })
        })
    });

    view! {
        {move || {
            if let Some(region_layout) = region_layout() {
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
                            <Region region=region_layout />
                        </g>
                    </svg>
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
