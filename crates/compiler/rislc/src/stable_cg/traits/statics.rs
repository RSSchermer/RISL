use rustc_public::mir::mono::StaticDef;
use rustc_public::ty::Align;

use super::BackendTypes;

pub trait StaticCodegenMethods: BackendTypes {
    fn static_addr_of(&self, cv: Self::Value, align: Align, kind: Option<&str>) -> Self::Value;
    fn codegen_static(&self, def: StaticDef);
}

pub trait StaticBuilderMethods: BackendTypes {
    fn get_static(&mut self, def: StaticDef) -> Self::Value;
}
