pub mod builtin_function;
pub mod cfg;
pub mod cfg_to_rvsdg;
pub mod dependencies;
pub mod import;
pub mod intrinsic;
pub mod rvsdg;
pub mod rvsdg_to_scf;
pub mod scf;
pub mod smi;
pub mod ty;
pub mod write;

mod branch_case;
mod core;
mod serde;
mod util;

pub use branch_case::BranchCase;

pub use self::core::*;
