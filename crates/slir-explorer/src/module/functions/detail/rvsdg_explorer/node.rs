use leptos::prelude::*;
use leptos::{component, IntoView};

use crate::module::functions::detail::rvsdg_explorer::connector::{Connector, ToolTipPosition};
use crate::module::functions::detail::rvsdg_explorer::layout::{NodeContent, NodeLayout};
use crate::module::functions::detail::rvsdg_explorer::region::Region;
use crate::module::functions::detail::rvsdg_explorer::use_region_viewer_context;

// It seems that if we set the font-size to e.g. 15px, the text element's height will get some
// top and bottom padding, even if we try to nullify that in CSS or try setting the CSS line-height
// to 15px also. If anyone reading this knows what's going on here please reach out. For now I'm
// dirty-fixing this with a hard-coded adjustment
const TEXT_ADJUST: f32 = -3.0;

const TOOLTIP_FONT_HEIGHT: f32 = 15.0;
const TOOLTIP_FONT_RATIO: f32 = 0.6;
const TOOLTIP_PADDING: f32 = 5.0;

#[component]
pub fn Node(node: NodeLayout) -> impl IntoView {
    let node = StoredValue::new(node);
    let rect = move || node.read_value().rect();

    let rect_class = match node.read_value().content() {
        NodeContent::PlainText(_) => "node-rect simple",
        NodeContent::FnCall(_, _) => "node-rect simple",
        NodeContent::Loop(_, _) => "node-rect loop",
        NodeContent::Switch(_, _) => "node-rect switch",
    };

    view! {
        {move || {
            node.read_value().input_connectors().iter().cloned().map(|connector| {
                view! { <Connector connector tooltip_position=ToolTipPosition::Top /> }
            }).collect_view()
        }}

        {move || {
            node.read_value().output_connectors().iter().cloned().map(|connector| {
                view! { <Connector connector tooltip_position=ToolTipPosition::Bottom /> }
            }).collect_view()
        }}

        <g class="node-container">
            <rect class=rect_class x=rect().origin[0] y=rect().origin[1] width=rect().size[0] height=rect().size[1] />

            <g class="node-content-container">
                {move || {
                    let text = format!("{:?}", node.read_value().node());
                    let text_width = text.len() as f32 * TOOLTIP_FONT_HEIGHT * TOOLTIP_FONT_RATIO;
                    let width = text_width + TOOLTIP_PADDING * 2.0;
                    let height = TOOLTIP_FONT_HEIGHT + TOOLTIP_PADDING * 2.0;
                    let text_base = TOOLTIP_PADDING + TOOLTIP_FONT_HEIGHT;
                    let x = rect().origin[0];
                    let y = rect().origin[1] - height;

                    view! {
                        <g class="node-tooltip" transform=format!("translate({}, {})", x, y)>
                            <rect x=0 y=0 width=width height=height />
                            <text x=TOOLTIP_PADDING y=text_base>{text}</text>
                        </g>
                    }
                }}

                {move || {
                    let node = node.read_value();

                    match node.content() {
                        NodeContent::PlainText(text) => {
                            let [x, y] = text.translation();
                            let tooltip = text.tooltip().map(|t| t.to_string());

                            view! {
                                <g transform=format!("translate({}, {})", x, y)>
                                    <text y=TEXT_ADJUST>{text.text().to_owned()}</text>
                                </g>

                                {move || {
                                    if let Some(tooltip) = tooltip.clone() {
                                        let text_width = tooltip.len() as f32 * TOOLTIP_FONT_HEIGHT * TOOLTIP_FONT_RATIO;
                                        let width = text_width + TOOLTIP_PADDING * 2.0;
                                        let height = TOOLTIP_FONT_HEIGHT + TOOLTIP_PADDING * 2.0;
                                        let text_base = TOOLTIP_PADDING + TOOLTIP_FONT_HEIGHT;

                                        view! {
                                            <g class="node-tooltip" transform="translate(0, 5)">
                                                <rect x=0 y=0 width=width height=height />
                                                <text x=TOOLTIP_PADDING y=text_base>{tooltip}</text>
                                            </g>
                                        }.into_any()
                                    } else {
                                        view! {}.into_any()
                                    }
                                }}
                            }.into_any()
                        }
                        NodeContent::FnCall(text, f) => {
                            let [x, y] = text.translation();
                            let ctx = use_region_viewer_context();
                            let f = *f;

                            view! {
                                <g>
                                    <a href=move || (ctx.gen_function_url)(f)>
                                        <text x=x y=y+TEXT_ADJUST>
                                            {text.text().to_owned()}
                                        </text>
                                    </a>
                                </g>
                            }.into_any()
                        }
                        NodeContent::Loop(text, region_layout) => {
                            let [x, y] = text.translation();
                            let region_layout = region_layout.clone();

                            view! {
                                <text x=x y=y+TEXT_ADJUST>{text.text().to_owned()}</text>

                                <Region region_layout />
                            }.into_any()
                        }
                        NodeContent::Switch(text, regions) => {
                            let [x, y] = text.translation();
                            let regions = regions.clone();

                            view! {
                                <text x=x y=y+TEXT_ADJUST>{text.text().to_owned()}</text>

                                {move || {
                                    regions.clone().into_iter().map(|region_layout| {
                                        view! { <Region region_layout /> }
                                    }).collect_view()
                                }}
                            }.into_any()
                        }
                    }
                }}
            </g>
        </g>
    }
}
