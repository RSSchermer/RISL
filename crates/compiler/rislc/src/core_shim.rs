use std::fs::File;
use std::io::Read;

use ar::Archive;
use rustc_hash::FxHashMap;
use rustc_hir::def_id::{CrateNum, DefId, DefIndex, LOCAL_CRATE};
use rustc_middle::ty::TyCtxt;
use rustc_public::CrateDef;
use rustc_public::mir::mono::Instance;
use rustc_public::rustc_internal::{internal, stable};
use rustc_span::Symbol;
use serde::{Deserialize, Serialize};

use crate::compiler::SHIM_LOOKUP_HEADER;
use crate::context::RislContext;
use crate::hir_ext::{FnExt, GpuFnExt, HirExt};

#[derive(Serialize, Deserialize, Debug)]
pub struct ShimDefLookup {
    shim_crate_num: usize,
    mapping: FxHashMap<String, usize>,
}

impl ShimDefLookup {
    pub fn new(shim_crate_num: CrateNum) -> Self {
        ShimDefLookup {
            shim_crate_num: shim_crate_num.as_usize(),
            mapping: FxHashMap::default(),
        }
    }

    pub fn maybe_shimmed(&self, tcx: TyCtxt, instance: Instance) -> Instance {
        let def_path = instance
            .ty()
            .kind()
            .fn_def()
            .expect("instance should be a function")
            .0
            .name();

        self.mapping
            .get(&def_path)
            .copied()
            .map(|index| {
                let shimmed_def_id = DefId {
                    krate: CrateNum::from_usize(self.shim_crate_num),
                    index: DefIndex::from_usize(index),
                };

                let instance = internal(tcx, instance);
                let shimmed_instance = rustc_middle::ty::Instance {
                    def: rustc_middle::ty::InstanceKind::Item(shimmed_def_id),
                    args: instance.args,
                };

                stable(&shimmed_instance)
            })
            .unwrap_or(instance)
    }

    pub fn register_gpu_fn_ext(&mut self, def_id: DefId, gpu_fn_ext: &GpuFnExt) {
        if let Some(target) = gpu_fn_ext.core_shim_for {
            self.mapping
                .insert(target.to_string(), def_id.index.as_usize());
        }
    }
}

pub fn is_core_shim_crate(tcx: TyCtxt) -> bool {
    tcx.hir_krate_attrs().iter().any(|attr| {
        attr.path_matches(&[Symbol::intern("rislc"), Symbol::intern("core_shim_crate")])
    })
}

pub fn resolve_shim_def_lookup(tcx: TyCtxt, hir_ext: &HirExt) -> ShimDefLookup {
    if is_core_shim_crate(tcx) {
        return build_shim_def_lookup(hir_ext);
    }

    for &crate_num in tcx.crates(()) {
        let is_shim_crate = tcx
            .get_attrs(crate_num.as_def_id(), Symbol::intern("rislc"))
            .any(|attr| {
                attr.path_matches(&[Symbol::intern("rislc"), Symbol::intern("core_shim_crate")])
            });

        if is_shim_crate {
            return load_shim_def_lookup(tcx, crate_num);
        }
    }

    ShimDefLookup {
        shim_crate_num: 0,
        mapping: Default::default(),
    }
}

fn load_shim_def_lookup(tcx: TyCtxt, dependency: CrateNum) -> ShimDefLookup {
    let filename = &tcx.used_crate_source(dependency).rlib.as_ref().unwrap().0;

    let mut archive = Archive::new(File::open(filename).unwrap());

    while let Some(Ok(mut entry)) = archive.next_entry() {
        if entry.header().identifier() == SHIM_LOOKUP_HEADER.as_bytes() {
            let mut bytes = Vec::with_capacity(entry.header().size() as usize + 1);

            entry.read_to_end(&mut bytes).unwrap();

            let (lookup, _) = bincode::serde::decode_from_slice::<ShimDefLookup, _>(
                bytes.as_slice(),
                bincode::config::standard(),
            )
            .unwrap();

            return lookup;
        }
    }

    panic!(
        "failed to load shim def lookup from crate `{}`",
        tcx.crate_name(dependency)
    );
}

pub fn build_shim_def_lookup(hir_ext: &HirExt) -> ShimDefLookup {
    let mut lookup = ShimDefLookup::new(LOCAL_CRATE);

    for (&def_id, fn_ext) in &hir_ext.fn_ext {
        if let FnExt::GpuFn(gpu_fn_ext) = fn_ext {
            lookup.register_gpu_fn_ext(def_id.to_def_id(), gpu_fn_ext);
        }
    }

    for (&hir_id, gpu_fn_ext) in &hir_ext.impl_fn_ext {
        let def_id = hir_id.expect_owner().def_id;

        lookup.register_gpu_fn_ext(def_id.to_def_id(), gpu_fn_ext);
    }

    for (&hir_id, gpu_fn_ext) in &hir_ext.trait_fn_ext {
        let def_id = hir_id.expect_owner().def_id;

        lookup.register_gpu_fn_ext(def_id.to_def_id(), gpu_fn_ext);
    }

    lookup
}
