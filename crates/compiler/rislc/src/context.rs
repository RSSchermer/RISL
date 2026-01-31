use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::{LazyLock, Once};

use rustc_hash::FxHashMap;
use rustc_hir::{ItemId, Mod};
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::{CrateNum, DefId, DefIndex, LOCAL_CRATE, LocalDefId, LocalModDefId};

use crate::core_shim::is_core_shim_crate;
use crate::hir_ext::{ExtendedItem, HirExt, ModExt};
use crate::hir_ext_build;

pub fn crate_slir_module_name(tcx: TyCtxt, crate_num: CrateNum) -> String {
    let crate_name = tcx.crate_name(crate_num);
    let extra = if crate_num == LOCAL_CRATE {
        &tcx.sess.opts.cg.extra_filename
    } else {
        tcx.extra_filename(crate_num)
    };

    format!("{}{}", crate_name, extra)
}

pub fn generate_crate_slir_module_name_to_crate_num(tcx: TyCtxt) -> FxHashMap<String, CrateNum> {
    let mut crate_name_to_num = FxHashMap::default();

    for crate_num in tcx.crates(()) {
        crate_name_to_num.insert(crate_slir_module_name(tcx, *crate_num), *crate_num);
    }

    crate_name_to_num
}

pub struct RislContext<'tcx> {
    tcx: TyCtxt<'tcx>,
    hir_ext: HirExt,
    crate_slir_module_name_to_crate_num: FxHashMap<String, CrateNum>,
    current_crate_is_core_shim_crate: bool,
}

impl<'tcx> RislContext<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>) -> Self {
        RislContext {
            tcx,
            hir_ext: hir_ext_build::build(tcx),
            crate_slir_module_name_to_crate_num: generate_crate_slir_module_name_to_crate_num(tcx),
            current_crate_is_core_shim_crate: is_core_shim_crate(tcx),
        }
    }

    pub fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    /// Whether the current crate is the core-shim crate.
    pub fn current_crate_is_core_shim_crate(&self) -> bool {
        self.current_crate_is_core_shim_crate
    }

    pub fn crate_slir_module_name(&self, crate_num: CrateNum) -> String {
        crate_slir_module_name(self.tcx, crate_num)
    }

    pub fn crate_num_for_crate_slir_module_name(&self, name: &str) -> CrateNum {
        *self
            .crate_slir_module_name_to_crate_num
            .get(name)
            .expect("crate not found")
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
        let extra = if module_id.is_local() {
            self.tcx().sess.opts.cg.extra_filename.clone()
        } else {
            self.tcx().extra_filename(module_id.krate).clone()
        };

        format!(
            "{}-{}{}",
            crate_name,
            self.tcx().def_path_str(module_id),
            extra
        )
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
        let filename = format!("{}.slir", self.shader_module_name(module_id.to_def_id()));

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
