pub mod branch_selector_normalization;
pub mod common_node_elimination;
pub mod common_value_elimination;
pub mod conditional_ub_elimination;
pub mod const_dependency_inlining;
pub mod const_switch_output_extraction;
pub mod contextual_node_reduction;
pub mod dead_loop_value_normalization;
pub mod dead_value_elimination;
pub mod duplicate_loop_value_elimination;
pub mod duplicate_switch_input_elimination;
pub mod duplicate_switch_output_elimination;
pub mod enum_replacement;
pub mod fallback_value_replacement;
pub mod function_inlining;
pub mod identical_branch_elimination;
pub mod instruction_combining;
pub mod invalid_ptr_replacement;
pub mod loop_pointer_normalization;
pub mod loop_slice_offset_normalization;
pub mod memory_promotion_and_legalization;
pub mod memory_transform;
pub mod offset_slice_elaboration;
pub mod offset_slice_replacement;
pub mod passthrough_elimination;
pub mod pointer_reconstruction;
pub mod pred_to_case_extraction;
pub mod pred_to_case_to_pred_merging;
pub mod proxy_node_elimination;
pub mod redundant_state_link_elimination;
pub mod region_replication;
pub mod scalar_replacement;
pub mod store_coalescing;
mod switch_branch_pruning;
pub mod switch_fallback_unification;
pub mod switch_merging;
pub mod switchify_pred_to_case;
pub mod variable_pointer_emulation;

use crate::rvsdg::Rvsdg;
use crate::rvsdg::transform::common_value_elimination::CommonValueEliminator;
use crate::rvsdg::transform::const_switch_output_extraction::ConstSwitchOutputExtractor;
use crate::rvsdg::transform::contextual_node_reduction::ContextualNodeReducer;
use crate::rvsdg::transform::dead_loop_value_normalization::DeadLoopValueNormalizer;
use crate::rvsdg::transform::identical_branch_elimination::IdenticalBranchElimination;
use crate::rvsdg::transform::instruction_combining::InstructionCombiner;
use crate::rvsdg::transform::passthrough_elimination::PassthroughEliminator;
use crate::rvsdg::transform::redundant_state_link_elimination::RedundantStateLinkEliminator;
use crate::rvsdg::transform::switch_fallback_unification::SwitchFallbackUnifier;
use crate::rvsdg::transform::switch_merging::SwitchMerger;
use crate::{Function, Module};

pub fn transform(module: &mut Module, rvsdg: &mut Rvsdg) {
    function_inlining::transform_entry_points(module, rvsdg);
    passthrough_elimination::transform_entry_points(module, rvsdg);
    offset_slice_elaboration::transform_entry_points(module, rvsdg);
    memory_transform::transform_entry_points(module, rvsdg);
    redundant_state_link_elimination::transform_entry_points(module, rvsdg);
    offset_slice_replacement::transform_entry_points(module, rvsdg);
    // After offset-slice-replacement, unsized "slice" array values should have been made redundant.
    // However, such "dead" values have not yet been removed. We do an early dead-value-elimination
    // pass here so that later passes may assume the absence of unsized "slice" array values.
    dead_value_elimination::transform_entry_points(module, rvsdg);
    pred_to_case_extraction::transform_entry_points(module, rvsdg);
    pred_to_case_to_pred_merging::transform_entry_points(module, rvsdg);
    switchify_pred_to_case::transform_entry_points(module, rvsdg);
    common_value_elimination::transform_entry_points(module, rvsdg);
    const_dependency_inlining::transform_entry_points(module, rvsdg);
    contextual_node_reduction::transform_entry_points(module, rvsdg);
    invalid_ptr_replacement::transform_entry_points(module, rvsdg);
    switch_merging::transform_entry_points(module, rvsdg);
    branch_selector_normalization::transform_entry_points(module, rvsdg);
    conditional_ub_elimination::transform_entry_points(module, rvsdg);
    common_value_elimination::transform_entry_points(module, rvsdg);

    // TODO: this is an expensive and optional optimization loop. We should add some config
    // to control whether this is run.
    optimize_entry_points(module, rvsdg);

    fallback_value_replacement::transform_entry_points(module, rvsdg);

    rvsdg.dump_to_file("final_transformed.dump").unwrap();
}

fn optimize_entry_points(module: &mut Module, rvsdg: &mut Rvsdg) {
    let mut optimizer = LoopingOptimizer::new();
    let entry_points = module.entry_points.iter().collect::<Vec<_>>();

    for (entry_point, _) in entry_points {
        optimizer.optimize_function(module, rvsdg, entry_point);
    }
}

struct LoopingOptimizer {
    contextual_value_optimizer: ContextualNodeReducer,
    instruction_combiner: InstructionCombiner,
    switch_merger: SwitchMerger,
    common_value_eliminator: CommonValueEliminator,
    switch_fallback_unifier: SwitchFallbackUnifier,
    const_switch_output_extractor: ConstSwitchOutputExtractor,
    passthrough_eliminator: PassthroughEliminator,
    redundant_state_link_eliminator: RedundantStateLinkEliminator,
    dead_loop_value_normalizer: DeadLoopValueNormalizer,
    identical_branch_eliminator: IdenticalBranchElimination,
}

impl LoopingOptimizer {
    fn new() -> Self {
        Self {
            contextual_value_optimizer: ContextualNodeReducer::new(),
            instruction_combiner: InstructionCombiner::new(),
            switch_merger: SwitchMerger::new(),
            common_value_eliminator: CommonValueEliminator::new(),
            switch_fallback_unifier: SwitchFallbackUnifier::new(),
            const_switch_output_extractor: ConstSwitchOutputExtractor::new(),
            passthrough_eliminator: PassthroughEliminator::new(),
            redundant_state_link_eliminator: RedundantStateLinkEliminator::new(),
            dead_loop_value_normalizer: DeadLoopValueNormalizer::new(),
            identical_branch_eliminator: IdenticalBranchElimination::new(),
        }
    }

    fn optimize_function(&mut self, module: &mut Module, rvsdg: &mut Rvsdg, function: Function) {
        let fn_node = rvsdg
            .get_function_node(function)
            .expect("function not registered");
        let body_region = rvsdg[fn_node].expect_function().body_region();

        let mut do_iteration = true;

        while do_iteration {
            do_iteration = false;

            do_iteration |= self
                .contextual_value_optimizer
                .optimize_function(module, rvsdg, function);
            do_iteration |= self.instruction_combiner.combine_in_fn(rvsdg, function);
            do_iteration |= self.switch_merger.merge_in_fn(module, rvsdg, function);
            do_iteration |= self
                .switch_fallback_unifier
                .process_region(rvsdg, body_region);
            do_iteration |= self
                .const_switch_output_extractor
                .process_region(rvsdg, body_region);
            do_iteration |= self.passthrough_eliminator.eliminate_in_fn(rvsdg, function);
            do_iteration |= self
                .dead_loop_value_normalizer
                .transform_region(rvsdg, body_region);
            do_iteration |=
                self.identical_branch_eliminator
                    .inline_in_region(module, rvsdg, body_region);
            do_iteration |= self
                .common_value_eliminator
                .process_region(rvsdg, body_region);
            do_iteration |= self
                .redundant_state_link_eliminator
                .eliminate_in_fn(rvsdg, function);
        }
    }
}
