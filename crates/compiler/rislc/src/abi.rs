use rustc_abi::Variants;
use rustc_middle::ty::{Ty, TyCtxt};
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
}
