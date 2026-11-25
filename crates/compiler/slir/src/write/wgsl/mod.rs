mod abi;
mod writer;

use rustc_hash::FxHashSet;

use self::abi::AbiLayouts;
use self::writer::{Context, WgslModuleWriter};
use crate::Module;
use crate::scf::Scf;
use crate::scf::analyze::global_use::collect_used_global_bindings;
use crate::scf::analyze::local_binding_use::count_local_binding_use;
use crate::scf::analyze::struct_use::collect_used_structs;

pub fn write_wgsl(module: &Module, scf: &Scf) -> String {
    let abi_layouts = AbiLayouts::build(module);
    let used_globals = collect_used_global_bindings(module, scf);
    let used_structs = collect_used_structs(module, scf, &used_globals);

    let local_binding_use_counts = count_local_binding_use(scf);

    let cx = Context {
        module,
        scf,
        local_binding_use_counts: &local_binding_use_counts,
        used_structs: &used_structs,
        used_workgroup_bindings: &used_globals.workgroup_bindings,
        used_constants: &used_globals.constants,
        abi_layouts: &abi_layouts,
    };

    let reserved_names = module
        .entry_points
        .iter()
        .map(|(_, e)| e.name.to_string())
        .collect::<FxHashSet<_>>();

    WgslModuleWriter::new(reserved_names).write(cx)
}
