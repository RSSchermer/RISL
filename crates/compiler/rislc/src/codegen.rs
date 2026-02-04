use std::fs::File;
use std::io::Read;

use ar::Archive;
use rustc_public::rustc_internal::run;
use rustc_span::Symbol;
use rustc_span::def_id::LOCAL_CRATE;
use slir::{rvsdg, scf};

use crate::artifact::{SlirArtifactBuilder, SlirArtifactBuilderConfig};
use crate::compiler::LIB_MODULE_HEADER;
use crate::context::RislContext as Cx;
use crate::monomorphize::collect_shader_module_codegen_units;
use crate::slir_build::build_shader_module;

struct RlibDependencyLoader<'a, 'tcx> {
    rcx: &'a Cx<'tcx>,
}

impl slir::dependencies::DependencyLoader for RlibDependencyLoader<'_, '_> {
    type ModuleData<'a>
        = slir::Module
    where
        Self: 'a;
    type CfgData<'a>
        = slir::cfg::Cfg
    where
        Self: 'a;

    fn load<'a>(&'a mut self, module: slir::Symbol) -> (Self::ModuleData<'a>, Self::CfgData<'a>) {
        let crate_num = self
            .rcx
            .crate_num_for_crate_slir_module_name(module.as_str());
        let filename = &self
            .rcx
            .tcx()
            .used_crate_source(crate_num)
            .rlib
            .as_ref()
            .unwrap()
            .0;

        let mut archive = Archive::new(File::open(filename).unwrap());

        while let Some(Ok(mut entry)) = archive.next_entry() {
            if entry.header().identifier() == LIB_MODULE_HEADER.as_bytes() {
                let mut bytes = Vec::with_capacity(entry.header().size() as usize + 1);

                entry.read_to_end(&mut bytes).unwrap();

                let ((module, cfg_data), _) =
                    bincode::serde::decode_from_slice::<(slir::Module, slir::cfg::CfgData), _>(
                        bytes.as_slice(),
                        bincode::config::standard(),
                    )
                    .unwrap();

                let cfg = slir::cfg::Cfg::from_ty_and_data(module.ty.clone(), cfg_data);

                return (module, cfg);
            }
        }

        panic!("failed to load module dependency: {}", module);
    }
}

pub fn codegen_shader_modules(cx: &Cx) -> (slir::Module, slir::cfg::Cfg) {
    run(cx.tcx(), || {
        let (free_items, shader_modules) = collect_shader_module_codegen_units(cx);
        let mut dependency_loader = RlibDependencyLoader { rcx: cx };

        // We create a separate SLIR artifact for every shader module in the current crate (every
        // `mod` item with a `#[risl::shader_module]` attribute). These artifacts are the basis for the
        // final compilation step (e.g. to WGSL, SPIRV, HLSL, etc.), which is typically done by a macro
        // in the second compilation phase (when the actual non-RISL Rust code gets compiled).
        for shader_module in shader_modules {
            let name = slir::Symbol::new(cx.shader_module_name(shader_module.def_id.to_def_id()));
            let mut artifact_builder = SlirArtifactBuilder::new(
                cx,
                SlirArtifactBuilderConfig {
                    module_id: shader_module.def_id,
                    include_rvsdg_initial: true,
                    include_rvsdg_transformed: true,
                    include_wgsl: true,
                    include_smi: true,
                },
            );

            let (mut module, mut cfg) = build_shader_module(cx, name, &shader_module.items);

            slir::dependencies::import_dependencies(&mut module, &mut cfg, &mut dependency_loader);

            artifact_builder.add_cfg(&cfg);

            let mut rvsdg = slir::cfg_to_rvsdg::cfg_to_rvsdg(&mut module, &mut cfg);

            artifact_builder.maybe_add_rvsdg_initial(&rvsdg);

            rvsdg::transform::transform(&mut module, &mut rvsdg);

            artifact_builder.maybe_add_rvsdg_transformed(&rvsdg);

            // let mut scf = slir::rvsdg_to_scf::rvsdg_entry_points_to_scf(&module, &rvsdg);
            //
            // scf::transform::transform(&mut module, &mut scf);
            //
            // artifact_builder.add_scf(&scf);
            //
            // let wgsl = slir::write::wgsl::write_wgsl(&module, &scf);
            //
            // artifact_builder.maybe_add_wgsl(&wgsl);

            let smi = slir::smi::build_smi(&module, &cfg);

            artifact_builder.maybe_add_smi(&smi);

            artifact_builder.finish(&module);
        }

        // We also create one additional module for the whole crate for the SLIR of all "free functions"
        // (functions that are not part of a `mod` item with a `#[risl::shader_module]` attribute). This
        // will get stored as part of the crates `rlib`; it is used by `rislc` when compiling dependent
        // crates to import dependencies.
        let lib_name = slir::Symbol::new(cx.crate_slir_module_name(LOCAL_CRATE));

        build_shader_module(cx, lib_name, &free_items)
    })
    .unwrap()
}
