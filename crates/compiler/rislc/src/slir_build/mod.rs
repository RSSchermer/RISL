use indexmap::IndexSet;
use rustc_public::mir::mono::MonoItem;

use crate::context::RislContext;
use crate::slir_build::builder::Builder;
use crate::slir_build::context::CodegenContext;
use crate::slir_build::risl_intrinsic::maybe_rislc_intrinsic;
use crate::stable_cg::MonoItemExt;

pub mod builder;
pub mod context;
pub mod risl_intrinsic;
pub mod risl_primitive_ty;
pub mod ty;
pub mod value;

pub fn build_shader_module(
    rcx: &RislContext,
    name: slir::Symbol,
    items: &IndexSet<MonoItem>,
) -> (slir::Module, slir::cfg::Cfg) {
    let codegen_cx = CodegenContext::new(rcx, name);

    for item in items {
        item.predefine::<Builder>(&codegen_cx);
    }

    for item in items {
        // First check for an intrinsic, which needs special handling. Note that this function
        // returns `Some` with the original mono-item if the item is *not* an intrinsic.
        let item = maybe_rislc_intrinsic(item.clone(), &codegen_cx);

        if let Some(item) = item {
            // No special case; send it down the regular codegen path.
            item.define::<Builder>(&codegen_cx);
        }
    }

    codegen_cx.finish()
}
