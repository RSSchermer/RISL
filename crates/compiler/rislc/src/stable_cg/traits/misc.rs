use rustc_public::mir::mono::Instance;
use rustc_public::ty::{FnDef, GenericArgs};

use super::BackendTypes;
use crate::stable_cg::TyAndLayout;

pub trait MiscCodegenMethods: BackendTypes {
    fn get_fn(&self, instance: &Instance) -> Self::Function;
    fn get_fn_addr(&self, instance: &Instance) -> Self::Value;
    fn resolve_instance(&self, fn_def: FnDef, args: &GenericArgs) -> Instance;
    fn adjust_layout(&self, layout: TyAndLayout) -> TyAndLayout;
}
