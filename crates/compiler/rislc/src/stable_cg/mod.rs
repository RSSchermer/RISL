// This module is modified from rustc_codegen_ssa

mod common;
mod layout;
mod mir;
mod mono_item;
mod scalar;

pub mod traits;

pub use self::common::*;
pub use self::layout::{ScalarExt, TyAndLayout};
pub use self::mir::operand::{OperandRef, OperandValue};
pub use self::mir::place::{PlaceRef, PlaceValue};
pub use self::mono_item::MonoItemExt;
pub use self::scalar::Scalar;
