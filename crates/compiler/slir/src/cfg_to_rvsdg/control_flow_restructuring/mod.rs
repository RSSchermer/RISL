mod strongly_connected_components;

mod branch_restructuring;
mod exit_restructuring;
pub mod graph;
mod loop_restructuring;

pub use self::branch_restructuring::restructure_branches;
pub use self::graph::*;
pub use self::loop_restructuring::restructure_loops;
