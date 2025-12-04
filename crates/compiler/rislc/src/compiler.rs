use std::env;
use std::fs::File;
use std::path::PathBuf;

use ar::{GnuBuilder, Header};
use rustc_driver::{Callbacks, Compilation, run_compiler};
use rustc_interface::interface;
use rustc_metadata::METADATA_FILENAME;
use rustc_metadata::fs::encode_and_write_metadata;
use rustc_middle::ty::TyCtxt;
use rustc_public::target;
use rustc_session::config::{CrateType, DebugInfo};
use rustc_session::output::out_filename;
use rustc_span::def_id::LOCAL_CRATE;
use rustc_target::spec::TargetTuple;

use crate::abi;
use crate::codegen::codegen_shader_modules;
use crate::context::RislContext;
use crate::hir_ext::{ShaderRequest, ShaderRequestKind};

pub const LIB_MODULE_FILENAME: &str = "lib.slir";
pub const ATTRIBUTE_NAMESPACE: &'static str = "rislc";

pub fn run(args: Vec<String>) {
    let mut compiler = RislCompiler {
        compile_risl: false,
    };

    run_compiler(&args, &mut compiler);
}

pub struct RislCompiler {
    compile_risl: bool,
}

impl Callbacks for RislCompiler {
    fn config(&mut self, config: &mut interface::Config) {
        self.compile_risl = !config
            .opts
            .crate_types
            .iter()
            .any(|ct| ct == &CrateType::ProcMacro);

        if self.compile_risl {
            // SAFETY: rustc should not be running concurrent threads during Callbacks::config
            unsafe {
                env::set_var("IS_RISLC_PASS", "1");
            }

            // Add a `rislc` cfg condition. We use this condition control the tool attributes in the
            // RISL standard library.
            config.crate_cfg.push("rislc".to_string());

            // Use the `register_tool` feature to register the `risl` tool to make rustc pass through
            // `#[risl::...]` attributes.
            let crate_attr = &mut config.opts.unstable_opts.crate_attr;
            crate_attr.push("feature(register_tool)".to_string());
            crate_attr.push(format!("register_tool({})", ATTRIBUTE_NAMESPACE));

            // With `DebugInfo::Full`, stable_cg generates a bunch of extra alloca statements for locals
            // into which the function argument values get copied; it does this as a work-around to be
            // able to attach debug-info. For our purposes, this just adds noise, so set the debug-info
            // level to `DebugInfo::Limited`.
            config.opts.debuginfo = DebugInfo::Limited;

            // We never want to generate overflow panics code for the GPU, not even in debug mode, so
            // when compiling RISL we always disable these checks.
            config.opts.cg.overflow_checks = Some(false);

            // Disable certain MIR transformations
            config.opts.unstable_opts.inline_mir = Some(false);
            config.opts.unstable_opts.mir_enable_passes.extend([
                ("GVN".to_string(), false),
                // Disable raw pointer UB checks. We don't currently allow user code to use raw
                // pointers (only references), so this type UB should not be possible. The checks
                // also involve pointer casting operations that the backend cannot support, and if
                // the check were to fail, we would not be able to unwind/abort.
                ("CheckAlignment".to_string(), false),
                ("CheckNull".to_string(), false),
            ]);

            config.override_queries = Some(|_, providers| {
                abi::provide(providers);
            });
        }
    }

    fn after_analysis<'tcx>(
        &mut self,
        _compiler: &interface::Compiler,
        tcx: TyCtxt<'tcx>,
    ) -> Compilation {
        if self.compile_risl {
            compile_risl(tcx);

            Compilation::Stop
        } else {
            Compilation::Continue
        }
    }
}

fn create_rlib(tcx: TyCtxt, lib_module: &(slir::Module, slir::cfg::Cfg)) {
    let filename = out_filename(
        tcx.sess,
        CrateType::Rlib,
        tcx.output_filenames(()),
        tcx.crate_name(LOCAL_CRATE),
    );

    println!("writing archive: {:?}", filename.as_path());

    let out_file = File::create(filename.as_path()).unwrap();

    let mut builder = GnuBuilder::new(
        out_file,
        [METADATA_FILENAME, LIB_MODULE_FILENAME]
            .iter()
            .map(|name| name.as_bytes().to_vec())
            .collect(),
    );

    let metadata = encode_and_write_metadata(tcx);
    let raw_metadata = metadata.full();

    builder
        .append(
            &Header::new(
                METADATA_FILENAME.as_bytes().to_vec(),
                raw_metadata.len() as u64,
            ),
            raw_metadata,
        )
        .unwrap();

    let raw_lib = bincode::serde::encode_to_vec(lib_module, bincode::config::standard()).unwrap();

    builder
        .append(
            &Header::new(
                LIB_MODULE_FILENAME.as_bytes().to_vec(),
                raw_lib.len() as u64,
            ),
            raw_lib.as_slice(),
        )
        .unwrap();

    builder.into_inner().unwrap();
}

fn fulfill_shader_request(cx: &RislContext, shader_request: &ShaderRequest) {
    let target_dir = env::var("CARGO_TARGET_DIR").unwrap_or_default();
    let target_dir = PathBuf::from(target_dir);
    let artifact_file_path = cx.shader_artifact_file_path(shader_request.shader_mod);

    let request_db_path = match shader_request.kind {
        ShaderRequestKind::Wgsl => target_dir.join("shader_wgsl_requests"),
        ShaderRequestKind::ShaderModuleInterface => todo!(),
    };

    let request_db = sled::open(request_db_path).unwrap();

    request_db
        .insert(
            shader_request.request_id.as_str(),
            artifact_file_path.as_os_str().as_encoded_bytes(),
        )
        .unwrap();
}

fn compile_risl(tcx: TyCtxt) {
    let mut cx = RislContext::new(tcx);

    cx.build_hir_ext();

    let lib_module = codegen_shader_modules(&cx);

    create_rlib(tcx, &lib_module);

    for shader_request in &cx.hir_ext().shader_requests {
        fulfill_shader_request(&cx, shader_request);
    }
}
