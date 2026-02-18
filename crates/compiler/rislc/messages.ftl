rislc_monomorphize_encountered_error_while_instantiating =
    the above error was encountered while instantiating `{$kind} {$instance}`

rislc_monomorphize_recursion_limit =
    reached the recursion limit while instantiating `{$instance}`
    .note = `{$def_path_str}` defined here

rislc_monomorphize_no_optimized_mir =
    missing optimized MIR for `{$instance}` in the crate `{$crate_name}`
    .note = missing optimized MIR for this item (was the crate `{$crate_name}` compiled with `--emit=metadata`?)
