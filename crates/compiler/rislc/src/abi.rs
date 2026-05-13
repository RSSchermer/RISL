use rustc_abi::{
    AbiAlign, Align, BackendRepr, FieldsShape, LayoutData, Size, VariantIdx, Variants,
};
use rustc_hir::definitions::DefPath;
use rustc_index::Idx;
use rustc_middle::infer::canonical::ir::inherent::AdtDef;
use rustc_middle::ty;
use rustc_middle::ty::layout::{LayoutError, TyAndLayout};
use rustc_middle::ty::{Ty, TyCtxt, TyKind};
use rustc_middle::util::Providers;
use rustc_target::callconv::{ArgAbi, ArgAttributes, FnAbi, PassMode};

pub fn provide(providers: &mut Providers) {
    // rustc generally passes ADT's indirectly, but makes an exception for enums where all variants
    // have at most one scalar field and the enum uses a niche rather than a separate discriminant
    // (i.e. `Option<&T>`); in these cases the default `FnAbi` will want to pass such a value in
    // "direct mode" and as a scalar. However, for our implementation strategy of enums (without
    // type casting, null pointers or variable pointers) we want all enums in the function address
    // space to be behind an alloca and all interactions to be through pointers. We therefore adjust
    // such `FnAbi`s to always pass enum types in "indirect" mode.

    fn adjust_fn_abi<'tcx>(
        tcx: TyCtxt<'tcx>,
        fn_abi: &'tcx FnAbi<'tcx, Ty<'tcx>>,
    ) -> &'tcx FnAbi<'tcx, Ty<'tcx>> {
        let adjusted_arg_abi = |arg: &ArgAbi<'tcx, Ty<'tcx>>| {
            let mut arg = arg.clone();

            if matches!(arg.mode, PassMode::Cast { .. }) {
                arg.mode = PassMode::Direct(ArgAttributes::new());
            }

            if matches!(arg.layout.layout.variants(), Variants::Multiple { .. })
                && !arg.layout.is_1zst()
                && !matches!(arg.mode, PassMode::Indirect { .. } | PassMode::Ignore)
            {
                arg.mode = PassMode::Indirect {
                    attrs: ArgAttributes::new(),
                    meta_attrs: None,
                    on_stack: false,
                };
            }

            arg
        };

        tcx.arena.alloc(FnAbi {
            args: fn_abi.args.iter().map(adjusted_arg_abi).collect(),
            ret: adjusted_arg_abi(&fn_abi.ret),
            c_variadic: fn_abi.c_variadic,
            fixed_count: fn_abi.fixed_count,
            conv: fn_abi.conv,
            can_unwind: fn_abi.can_unwind,
        })
    }

    providers.fn_abi_of_fn_ptr = |tcx, query| {
        let result = (rustc_interface::DEFAULT_QUERY_PROVIDERS.fn_abi_of_fn_ptr)(tcx, query);

        Ok(adjust_fn_abi(tcx, result?))
    };

    providers.fn_abi_of_instance = |tcx, query| {
        let result = (rustc_interface::DEFAULT_QUERY_PROVIDERS.fn_abi_of_instance)(tcx, query);

        Ok(adjust_fn_abi(tcx, result?))
    };

    fn is_slice_iter_def(tcx: TyCtxt, def_path: DefPath) -> bool {
        let crate_sym = tcx.crate_name(def_path.krate);
        let crate_str = crate_sym.as_str();

        if crate_str != "core" && crate_str != "std" {
            return false;
        }

        let Some(slice_mod_path) = def_path.data.get(0) else {
            return false;
        };

        if slice_mod_path.as_sym(false).as_str() != "slice" {
            return false;
        }

        let Some(iter_mod_path) = def_path.data.get(1) else {
            return false;
        };

        if iter_mod_path.as_sym(false).as_str() != "iter" {
            return false;
        }

        let Some(ty_path) = def_path.data.get(2) else {
            return false;
        };

        let ty_sym = ty_path.as_sym(false);
        let ty_str = ty_sym.as_str();

        ty_str == "Iter" || ty_str == "IterMut"
    }

    fn layout_of<'tcx>(
        tcx: TyCtxt<'tcx>,
        query: ty::PseudoCanonicalInput<'tcx, Ty<'tcx>>,
    ) -> Result<TyAndLayout<'tcx>, &'tcx LayoutError<'tcx>> {
        let ty::PseudoCanonicalInput {
            typing_env,
            value: ty,
        } = query;

        if let TyKind::Adt(def, generic_args) = ty.kind() {
            let def_path = tcx.def_path(def.def_id());

            if is_slice_iter_def(tcx, def_path) {
                let layout = LayoutData {
                    fields: FieldsShape::Arbitrary {
                        offsets: rustc_index::IndexVec::from_raw(vec![
                            Size::from_bytes(0),
                            Size::from_bytes(16),
                            Size::from_bytes(20),
                        ]),
                        memory_index: rustc_index::IndexVec::from_raw(vec![0u32, 1u32, 2u32]),
                    },
                    variants: Variants::Single {
                        index: VariantIdx::new(0),
                    },
                    backend_repr: BackendRepr::Memory { sized: true },
                    largest_niche: None,
                    uninhabited: false,
                    align: AbiAlign::new(Align::EIGHT),
                    size: Size::from_bytes(24),
                    max_repr_align: None,
                    unadjusted_abi_align: Align::EIGHT,
                    randomization_seed: Default::default(),
                };
                let layout = tcx.mk_layout(layout);

                return Ok(TyAndLayout { ty, layout });
            }
        }

        (rustc_interface::DEFAULT_QUERY_PROVIDERS.layout_of)(tcx, query)
    }

    providers.layout_of = layout_of;
}
