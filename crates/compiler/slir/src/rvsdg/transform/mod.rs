pub mod const_switch_inlining;
pub mod dead_connectible_elimination;
pub mod enum_replacement;
pub mod function_inlining;
pub mod memory_promotion_and_legalization;
pub mod memory_transform;
pub mod offset_slice_elaboration;
pub mod offset_slice_replacement;
pub mod pred_to_case_extraction;
pub mod pred_to_case_to_pred_merging;
pub mod proxy_node_elimination;
pub mod region_replication;
pub mod scalar_replacement;
pub mod store_coalescing;
pub mod switch_arg_reduction;
pub mod switch_merging;
pub mod switch_passthrough_elimination;
pub mod variable_pointer_emulation;

use crate::Module;
use crate::rvsdg::Rvsdg;

pub fn transform(module: &mut Module, rvsdg: &mut Rvsdg) {
    function_inlining::transform_entry_points(module, rvsdg);
    switch_passthrough_elimination::transform_entry_points(module, rvsdg);
    offset_slice_elaboration::transform_entry_points(module, rvsdg);
    memory_transform::transform_entry_points(module, rvsdg);
    offset_slice_replacement::transform_entry_points(module, rvsdg);
    pred_to_case_extraction::transform_entry_points(module, rvsdg);
    pred_to_case_to_pred_merging::transform_entry_points(module, rvsdg);
    switch_arg_reduction::transform_entry_points(module, rvsdg);
    const_switch_inlining::transform_entry_points(module, rvsdg);
    switch_merging::transform_entry_points(module, rvsdg);
    dead_connectible_elimination::transform_entry_points(module, rvsdg);
}
