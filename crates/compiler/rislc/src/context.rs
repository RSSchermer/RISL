use std::path::PathBuf;

use rustc_hash::FxHashMap;
use rustc_hir::{ItemId, Mod};
use rustc_middle::ty::TyCtxt;
use rustc_span::Symbol;
use rustc_span::def_id::{CrateNum, DefId, LocalModDefId};

use crate::hir_ext::{ExtendedItem, HirExt, ModExt};
use crate::hir_ext_build;

pub fn generate_crate_name_to_num(tcx: TyCtxt) -> FxHashMap<Symbol, CrateNum> {
    let mut crate_name_to_num = FxHashMap::default();

    for crate_num in tcx.crates(()) {
        let crate_name = tcx.crate_name(*crate_num);

        crate_name_to_num.insert(crate_name, *crate_num);
    }

    crate_name_to_num
}

pub struct RislContext<'tcx> {
    tcx: TyCtxt<'tcx>,
    hir_ext: HirExt,
    crate_name_to_num: FxHashMap<Symbol, CrateNum>,
}

impl<'tcx> RislContext<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>) -> Self {
        RislContext {
            tcx,
            hir_ext: HirExt::new(),
            crate_name_to_num: generate_crate_name_to_num(tcx),
        }
    }

    pub fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    pub fn crate_num_for_name(&self, name: Symbol) -> CrateNum {
        *self.crate_name_to_num.get(&name).expect("crate not found")
    }

    pub fn build_hir_ext(&mut self) {
        hir_ext_build::build(&mut self.hir_ext, self.tcx);
    }

    pub fn hir_ext(&self) -> &HirExt {
        &self.hir_ext
    }

    pub fn extended_item<'ext>(&'ext self, item_id: ItemId) -> Option<ExtendedItem<'tcx, 'ext>> {
        self.hir_ext.extend_item(self.tcx.hir_item(item_id))
    }

    pub fn extended_module<'ext>(
        &'ext self,
        id: LocalModDefId,
    ) -> Option<(&'tcx Mod<'tcx>, &'ext ModExt)> {
        self.hir_ext.get_mod_ext(id).map(|ext| {
            let (mod_, _, _) = self.tcx.hir_get_module(id);

            (mod_, ext)
        })
    }

    pub fn shader_module_name(&self, module_id: DefId) -> String {
        let crate_name = self.tcx().crate_name(module_id.krate);

        format!("{}-{}", crate_name, self.tcx().def_path_str(module_id))
    }

    pub fn shader_artifact_file_path(&self, module_id: DefId) -> PathBuf {
        let output_dir = self
            .tcx()
            .sess
            .io
            .output_dir
            .clone()
            .expect("no output directory specified");

        output_dir
            .join(format!("{}.slir", self.shader_module_name(module_id)))
            .canonicalize()
            .expect("should be a valid path")
    }
}
