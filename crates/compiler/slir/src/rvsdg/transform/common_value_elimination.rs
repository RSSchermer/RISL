use crate::Module;
use crate::rvsdg::transform::dead_connectible_elimination::DeadConnectibleEliminator;
use crate::rvsdg::transform::duplicate_switch_input_elimination::DuplicateSwitchInputEliminator;
use crate::rvsdg::transform::duplicate_switch_output_elimination::DuplicateSwitchOutputEliminator;
use crate::rvsdg::{Region, Rvsdg};

pub struct CommonValueEliminator {
    duplicate_switch_input_eliminator: DuplicateSwitchInputEliminator,
    duplicate_switch_output_eliminator: DuplicateSwitchOutputEliminator,
    dead_connectible_eliminator: DeadConnectibleEliminator,
}

impl CommonValueEliminator {
    pub fn new() -> Self {
        CommonValueEliminator {
            duplicate_switch_input_eliminator: DuplicateSwitchInputEliminator::new(),
            duplicate_switch_output_eliminator: DuplicateSwitchOutputEliminator::new(),
            dead_connectible_eliminator: DeadConnectibleEliminator::new(),
        }
    }

    pub fn process_region(&mut self, rvsdg: &mut Rvsdg, region: Region) {
        // Removing a duplicate value can uncover additional duplicate values. We therefore
        // remove duplicate values in a loop until a loop iteration finds no more duplicates.

        let mut do_iteration = true;

        while do_iteration {
            do_iteration = false;

            do_iteration |= self
                .duplicate_switch_input_eliminator
                .process_region(rvsdg, region);

            do_iteration |= self
                .duplicate_switch_output_eliminator
                .process_region(rvsdg, region);

            self.dead_connectible_eliminator
                .process_region(rvsdg, region)
        }
    }
}

pub fn transform_entry_points(module: &Module, rvsdg: &mut Rvsdg) {
    let mut eliminator = CommonValueEliminator::new();

    for (entry_point, _) in module.entry_points.iter() {
        let fn_node = rvsdg
            .get_function_node(entry_point)
            .expect("function not registered");
        let body_region = rvsdg[fn_node].expect_function().body_region();

        eliminator.process_region(rvsdg, body_region);
    }
}
