use std::env;
use std::fs::{File, create_dir_all};
use std::path::{Path, PathBuf};

use ar::{GnuBuilder, Header};
use rustc_driver::{Callbacks, Compilation, run_compiler};
use rustc_interface::interface;
use rustc_metadata::METADATA_FILENAME;
use rustc_metadata::fs::encode_and_write_metadata;
use rustc_middle::ty::TyCtxt;
use rustc_session::config::{CrateType, DebugInfo, ErrorOutputType, PrintKind};
use rustc_session::output::out_filename;
use rustc_session::search_paths::{PathKind, SearchPath};
use rustc_session::{EarlyDiagCtxt, Session};
use rustc_span::def_id::LOCAL_CRATE;
use rustc_target::spec::TargetTuple;

use crate::abi;
use crate::codegen::codegen_shader_modules;
use crate::context::RislContext;

pub const LIB_MODULE_FILENAME: &str = "lib.slir";
pub const ATTRIBUTE_NAMESPACE: &'static str = "rislc";
pub const RISL_INCREMENTAL_SUB_DIR: &'static str = "risl";
pub const RISL_OUTPUT_SUB_DIR: &'static str = "risl";

pub fn run(args: Vec<String>) {
    // RISL uses a two-pass compilation process.
    //
    // The first pass is the special RISL-pass. This pass registers a special RISL tool (see the
    // [register_tool crate attribute][1]). It also signals to the various RISL metadata macros
    // defined in the `risl-macros` crate (and re-exported by the `risl` standard library crate)
    // that they should output their RISL-specific metadata in the form of RISL tool-attributes. It
    // also signals to the various `shader_...` code-generation macros (e.g. `shader_wgsl!(...)`)
    // that they should operate in "request" mode and output tokens from which the RISL compiler can
    // derive a codegen request. The RISL compiler will collect the RISL-specific metadata and will
    // run an additional set of RISL-specific checks to verify that the attributes and the code they
    // annotate adhere to rules of RISL (TODO: the checks are not yet implemented). If there are no
    // errors, it will then fulfill the requests made by the code-generation macros by codegen-ing
    // the requested output and storing it in the filesystem (in the output directory).
    //
    // The second pass is a regular rustc pass. This pass sets no special signal for the various
    // macros. This causes the various metadata macros to not output anything at all. It also causes
    // the code-generation macros to operate in "retrieve" mode, in which they attempt to retrieve
    // the result of their request in the first pass. In the second pass the code-generation macros
    // essentially behave like the `include_str` and `include_bytes` macros in Rust's standard
    // library and inline the codegen results into the Rust code.
    //
    // [1]: https://doc.rust-lang.org/beta/unstable-book/language-features/register-tool.htm

    if args.iter().all(|arg| {
        // The compiler may have been invoked to print information. In all cases except for a
        // print-kind of "link-args" or "native-static-libs", the compiler will exit early. We don't
        // want to run the RISL passes for such early-exit print invocations, as we'll end up
        // double-printing in the second pass. For the "linked-args" and "native-static-libs" cases,
        // we'll run the RISL pass, but during `Callbacks::config` setup we remove all print
        // requests to avoid double-printing.
        if let Some(print_kind) = arg.strip_prefix("--print=") {
            print_kind == "link-args" || print_kind == "native-static-libs"
        } else {
            true
        }
    }) {
        // Run the first "RISL" pass
        run_compiler(&args, &mut RislPassCallbacks);
    }

    // TODO: we don't want to run the second pass if there were errors in the first pass.
    // `run_compiler` has no return value, but there has to be an elegant way to check for errors?

    // Run the second "regular Rust" pass.
    run_compiler(&args, &mut RustPassCallbacks);
}

pub struct RustPassCallbacks;

impl Callbacks for RustPassCallbacks {}

pub struct RislPassCallbacks;

impl Callbacks for RislPassCallbacks {
    fn config(&mut self, config: &mut interface::Config) {
        // Don't print anything during the RISL pass, we'd end up double-printing in the second
        // pass.
        config.opts.prints = Vec::new();

        // Do not emit JSON artifact notifications during the RISL pass. Build systems (Cargo) use
        // these notifications to detect when artifacts have been written; we only want to notify
        // the build system once, after the second pass.
        config.opts.json_artifact_notifications = false;

        config.opts.incremental = config.opts.incremental.as_ref().map(risl_incremental_dir);

        // Output RISL artifacts to an RISL-specific output directory.
        config.output_dir = Some(risl_output_dir(
            config.output_dir.as_ref().unwrap_or(&PathBuf::new()),
        ));

        // Add RISL-specific search paths.
        config.opts.search_paths = risl_search_paths(&config.opts.search_paths);

        // Add a `rislc` cfg condition. We use this condition on to turn the tool attributes
        // generated by risl_macros.
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

    fn after_analysis<'tcx>(
        &mut self,
        _compiler: &interface::Compiler,
        tcx: TyCtxt<'tcx>,
    ) -> Compilation {
        compile_risl(tcx);

        Compilation::Stop
    }
}

/// Returns a new output path for RISL output artifacts based on the output directory for regular
/// rustc output artifacts.
fn risl_incremental_dir(base: &PathBuf) -> PathBuf {
    let path = base.join(RISL_INCREMENTAL_SUB_DIR);

    ensure_dir(&path);

    path
}

/// Returns a new output path for RISL output artifacts based on the output directory for regular
/// rustc output artifacts.
fn risl_output_dir(base: &PathBuf) -> PathBuf {
    let path = base.join(RISL_OUTPUT_SUB_DIR);

    ensure_dir(&path);

    path
}

fn ensure_dir(path: &Path) {
    if !path.exists() {
        if let Err(err) = create_dir_all(&path) {
            let dcx = EarlyDiagCtxt::new(ErrorOutputType::default());

            dcx.early_fatal(format!("failed to create directory: {}", err));
        }
    }
}

/// Returns a new set of [SearchPath]s that includes the RISL-specific search paths.
fn risl_search_paths(base: &[SearchPath]) -> Vec<SearchPath> {
    // Creates a new list of search paths that first includes higher priority RISL-specific search
    // paths, followed by the original search paths.
    //
    // Note that the convention here must match the convention used by [risl_output_dir] for RISL
    // dependency artifacts to resolve correctly.

    let mut risl_search_paths = base
        .iter()
        .filter(|p| p.kind == PathKind::Dependency)
        .map(|p| {
            let risl_dir = p.dir.clone().join(RISL_OUTPUT_SUB_DIR);

            SearchPath::new(PathKind::Dependency, risl_dir)
        })
        .collect::<Vec<_>>();

    risl_search_paths.extend(base.iter().cloned());

    risl_search_paths
}

fn create_rlib(tcx: TyCtxt, lib_module: &(slir::Module, slir::cfg::Cfg)) {
    let filename = out_filename(
        tcx.sess,
        CrateType::Rlib,
        tcx.output_filenames(()),
        tcx.crate_name(LOCAL_CRATE),
    );

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

fn compile_risl(tcx: TyCtxt) {
    let mut cx = RislContext::new(tcx);

    cx.build_hir_ext();

    let lib_module = codegen_shader_modules(&cx);

    create_rlib(tcx, &lib_module);
}
