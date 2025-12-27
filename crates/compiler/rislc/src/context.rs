use std::ffi::OsString;
use std::path::PathBuf;

use rustc_hash::FxHashMap;
use rustc_hir::{ItemId, Mod};
use rustc_middle::ty::TyCtxt;
use rustc_middle::ty::layout::MaybeResult;
use rustc_span::Symbol;
use rustc_span::def_id::{CrateNum, DefId, DefIndex, LocalDefId, LocalModDefId};
use serde::ser::SerializeTuple;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

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

    /// Generates a path for the codegen results for the shader module identified by the
    /// `module_id`.
    pub fn shader_artifact_file_path(&self, module_id: LocalDefId) -> PathBuf {
        let output_dir = self
            .tcx()
            .sess
            .io
            .output_dir
            .clone()
            .expect("no output directory specified");
        let filename = format!(
            "{}{}.slir",
            self.shader_module_name(module_id.to_def_id()),
            &self.tcx.sess.opts.cg.extra_filename
        );

        output_dir.join(filename)
    }

    /// Creates a shader-module-artifact-mapping (SMAM) for the local crate.
    ///
    /// Maps shader module IDs to their codegen result files. Dependencies may fulfil shader codegen
    /// requests by looking up codegen artifacts through a dependent crate's SMAM.
    pub fn local_smam(&self) -> FxHashMap<u32, OsString> {
        let mut mapping = FxHashMap::default();

        for (mod_id, mod_ext) in &self.hir_ext().mod_ext {
            if mod_ext.is_shader_module && !self.tcx.is_unreachable_local_definition(*mod_id) {
                let index = mod_id.to_def_id().index.as_u32();
                let path = self
                    .shader_artifact_file_path(mod_id.to_local_def_id())
                    .into_os_string();

                mapping.insert(index, path);
            }
        }

        mapping
    }
}
