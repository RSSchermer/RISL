use rustc_public::mir::mono::{Instance, StaticDef};

pub trait PreDefineCodegenMethods {
    fn predefine_static(&self, def: StaticDef, symbol_name: &str);
    fn predefine_fn(&self, instance: Instance, symbol_name: &str);
}
