use rustc_hash::FxHashMap;
use rustc_hir::def_id::{CrateNum, DefId, DefIndex};
use rustc_middle::ty::TyCtxt;
use rustc_span::Symbol;
use serde::{Deserialize, Serialize};

use crate::context::RislContext as Cx;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
struct SerializableDefId {
    krate: usize,
    index: usize,
}

impl From<DefId> for SerializableDefId {
    fn from(def_id: DefId) -> Self {
        SerializableDefId {
            krate: def_id.krate.index(),
            index: def_id.index.index(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ShimDefLookup {
    shim_crate_num: usize,
    mapping: FxHashMap<SerializableDefId, usize>,
}

impl ShimDefLookup {
    pub fn maybe_shimmed(&self, def_id: DefId) -> DefId {
        self.mapping
            .get(&SerializableDefId::from(def_id))
            .copied()
            .map(|index| DefId {
                krate: CrateNum::from_usize(self.shim_crate_num),
                index: DefIndex::from_usize(index),
            })
            .unwrap_or(def_id)
    }
}

pub fn is_core_shim_crate(tcx: TyCtxt) -> bool {
    tcx.hir_krate_attrs().iter().any(|attr| {
        attr.path_matches(&[Symbol::intern("rislc"), Symbol::intern("core_shim_crate")])
    })
}

pub fn resolve_shim_def_lookup(cx: &RislContext) -> ShimDefLookup {
    if cx.current_crate_is_core_shim_crate() {}

    todo!()
}

fn build_shim_def_lookup(cx: &Cx) -> ShimDefLookup {
    todo!()
}
