use leptos::prelude::*;
use leptos::{component, IntoView};

use crate::module::functions::detail::rvsdg_explorer::connector::{Connector, ToolTipPosition};
use crate::module::functions::detail::rvsdg_explorer::layout::RegionLayout;
use crate::module::functions::detail::rvsdg_explorer::node::Node;

#[component]
pub fn Region(region_layout: RegionLayout) -> impl IntoView {
    let translation = region_layout.translation();
    let rect = region_layout.rect();

    let region = StoredValue::new(region_layout);

    view! {
        <g transform=format!("translate({} {})", translation[0], translation[1])>
            <rect class="region-rect" x=rect.origin[0] y=rect.origin[1] width=rect.size[0] height=rect.size[1]/>

            {move || {
                let region = region.read_value();
                let edge_count = region.edge_count();

                (0..edge_count).map(|i| {
                    let path = region.edge_vertices(i).iter().map(|v| format!("{},{}", v[0], v[1])).collect::<Vec<_>>().join(" ");
                    let is_state_edge = region.is_state_edge(i);
                    let (highlight, set_highlight) = signal(false);

                    // We want to do some edge-highlighting on hover (see the <style> block in the
                    // root svg element in mod.rs) to make it easier to visually track individual
                    // edges. However, mousing over the actual thin edge lines is somewhat finicky,
                    // so to make that easier we place the hover effect on an enclosing <g> element
                    // and then draw a second line as part of that group that is styled to be
                    // transparent (invisible) and has a greater stroke-width. This creates some
                    // margin around the "visible line" that makes it a bit easier to mouse over
                    // the element group and trigger the hover effect.
                    view! {
                        <g class="edge-lines"
                            class=("state-edge", move || is_state_edge)
                            class=("highlighted", move || highlight.get())
                            on:click=move |_| set_highlight.update(|v| *v = !*v)
                        >
                            <polyline class="visible-line" points=path.clone() />
                            <polyline class="hover-target" points=path />
                        </g>
                    }
                }).collect_view()
            }}

            {move || {
                region.read_value().argument_connectors().iter().cloned().map(|connector| {
                    view! { <Connector connector tooltip_position=ToolTipPosition::Bottom /> }
                }).collect_view()
            }}

            {move || {
                // Reverse the iteration order so we draw from right to left: this makes the
                // tooltips correctly appear on top if they are big enough to overlap a node to
                // the right.
                region.read_value().node_layouts().iter().rev().cloned().map(|node| {
                    view! { <Node node /> }
                }).collect_view()
            }}

            {move || {
                region.read_value().result_connectors().iter().cloned().map(|connector| {
                    view! { <Connector connector tooltip_position=ToolTipPosition::Top /> }
                }).collect_view()
            }}
        </g>
    }
}
