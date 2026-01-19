use indexmap::IndexSet;
use rustc_middle::mir::mono::MonoItem;
use rustc_public::rustc_internal::stable;

use crate::context::RislContext;
use crate::slir_build::builder::Builder;
use crate::slir_build::context::CodegenContext;
use crate::slir_build::risl_intrinsic::maybe_rislc_intrinsic;
use crate::slir_build::shims::maybe_shim;
use crate::stable_cg::MonoItemExt;

pub mod builder;
pub mod context;
mod risl_intrinsic;
mod risl_primitive_ty;
mod shims;
pub mod ty;
pub mod value;

pub fn build_shader_module<'tcx>(
    rcx: &RislContext<'tcx>,
    name: slir::Symbol,
    items: &IndexSet<MonoItem<'tcx>>,
) -> (slir::Module, slir::cfg::Cfg) {
    let codegen_cx = CodegenContext::new(rcx, name);

    for item in items {
        let item = stable(item);

        item.predefine::<Builder>(&codegen_cx);
    }

    for item in items {
        let item = stable(item);
        let item =
            maybe_shim(item, &codegen_cx).and_then(|item| maybe_rislc_intrinsic(item, &codegen_cx));

        // Certain items in `core` that we want to support use instructions internally that we don't
        // support in user-defined RISL. Rather than trying to convert such instruction sequences,
        // we instead treat these as special case intrinsics.
        if let Some(item) = item {
            // No special case; send it down the regular codegen path.
            item.define::<Builder>(&codegen_cx);
        }
    }

    codegen_cx.finish()
}
