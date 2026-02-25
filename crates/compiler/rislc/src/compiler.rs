use std::collections::HashMap;
use std::ffi::OsString;
use std::fs::{File, create_dir_all};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::{env, fs, io};

use ar::{Archive, GnuBuilder, Header};
use rustc_driver::{Callbacks, Compilation, run_compiler};
use rustc_hash::FxHashMap;
use rustc_interface::interface::Compiler;
use rustc_interface::{Config, interface};
use rustc_metadata::METADATA_FILENAME;
use rustc_metadata::fs::encode_and_write_metadata;
use rustc_middle::bug;
use rustc_middle::ty::TyCtxt;
use rustc_session::EarlyDiagCtxt;
use rustc_session::config::{CrateType, DebugInfo, ErrorOutputType, OutFileName};
use rustc_session::output::out_filename;
use rustc_session::search_paths::{PathKind, SearchPath};
use rustc_span::def_id::{CrateNum, LOCAL_CRATE};

use crate::codegen::codegen_shader_modules;
use crate::context::RislContext;
use crate::core_shim::ShimDefLookup;
use crate::{abi, monomorphize};

/// The header used in an `rlib` archive for the crate's "free items" SLIR-CFG module.
///
/// The SLIR-CFG module contains the SLIR control-flow-graph IR for the crate's reachable `gpu`
/// items that may be used by dependent crates. Such dependent crate's may use this module for
/// linking.
pub const LIB_MODULE_HEADER: &str = "lib.slir";

/// The header used in an `rlib` archive for the crate's shader-module-artifact-mapping.
pub const SMAM_HEADER: &str = "lib.smam";

/// The header used in an `rlib` archive for the std/core shim lookup table.
///
/// This is only present in the `rlib` for the `risl` standard library crate.
pub const SHIM_LOOKUP_HEADER: &str = "lib.shim";

/// The attribute tool namespace used to attach RISL metadata to Rust nodes.
pub const ATTRIBUTE_NAMESPACE: &'static str = "rislc";

/// The name of the subdirectory into which the RISLs pass' output artifacts are stored.
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
    // the requested output and storing the resulting artifacts in the filesystem.
    //
    // The second pass is a regular rustc pass. This pass sets no special signal for the various
    // macros. This causes the various metadata macros to not output anything at all. It also causes
    // the code-generation macros to operate in "retrieve" mode, in which they attempt to retrieve
    // the result of their request in the first pass. In the second pass the code-generation macros
    // essentially behave like the `include_str` and `include_bytes` macros in Rust's standard
    // library: they look up the correct artifact file and inline the requested codegen results into
    // the Rust code.
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
        let mut risl_args = args.clone();

        // Adjust the --extern arguments to point to a matching RISL-path (if one exists).
        //
        // It feels somewhat iffy modifying the raw arguments to achieve this. I would prefer to
        // adjust this in the `Callbacks::config` hook by adjusting the `config.opts.externs` value
        // directly, but that does not seem easily adjustable.
        adjust_extern_args(&mut risl_args);

        // Run the first "RISL" pass
        run_compiler(&risl_args, &mut RislPassCallbacks);
    }

    // TODO: we don't want to run the second pass if there were errors in the first pass.
    // `run_compiler` has no return value, but there has to be an elegant way to check for errors?

    let mut rust_pass_callbacks = RustPassCallbacks {
        share_proc_macro_lib: None,
    };

    // Run the second "regular Rust" pass.
    run_compiler(&args, &mut rust_pass_callbacks);

    if let Some(share_proc_macro_lib) = rust_pass_callbacks.share_proc_macro_lib {
        if let Err(_) = share_proc_macro_lib.apply() {
            panic!("{:#?}", share_proc_macro_lib);
        }
    }
}

/// Adjust the --extern arguments to point to a matching RISL-path (if one exists).
fn adjust_extern_args(args: &mut Vec<String>) {
    for i in 0..args.len() {
        if &args[i] == "--extern"
            && let Some(arg) = args.get_mut(i + 1)
        {
            let Some((crate_name, path)) = arg.split_once("=") else {
                continue;
            };

            let adjusted_path = risl_file(&PathBuf::from(path));

            if adjusted_path.exists() {
                let mut adjusted = crate_name.to_string();

                adjusted.push('=');
                adjusted.push_str(adjusted_path.to_str().unwrap_or_default());

                *arg = adjusted;
            }
        }
    }
}

#[derive(Debug)]
struct OpShareProcMacroLib {
    pub src: PathBuf,
    pub dst: PathBuf,
}

impl OpShareProcMacroLib {
    fn apply(&self) -> io::Result<()> {
        // Based on rustc_fs_util::link_or_copy

        let OpShareProcMacroLib { src, dst } = self;

        let err = match fs::hard_link(src, dst) {
            Ok(()) => return Ok(()),
            Err(err) => err,
        };

        if err.kind() == io::ErrorKind::AlreadyExists {
            fs::remove_file(dst)?;

            if fs::hard_link(src, dst).is_ok() {
                return Ok(());
            }
        }

        fs::copy(src, dst).map(|_| ())
    }
}

struct RustPassCallbacks {
    pub share_proc_macro_lib: Option<OpShareProcMacroLib>,
}

impl Callbacks for RustPassCallbacks {
    fn config(&mut self, config: &mut Config) {
        let risl_output_dir = risl_dir(config.output_dir.as_ref().unwrap_or(&PathBuf::new()));
        let shader_request_lookup = shader_request_lookup_path(
            &risl_output_dir,
            config
                .opts
                .crate_name
                .as_ref()
                .map(|s| s.as_str())
                .unwrap_or("crate"),
            &config.opts.cg.extra_filename,
        );

        // SAFETY: rustc should not be running concurrent threads during Callbacks::config
        unsafe {
            // Unset the `IS_RISLC_PASS` variable to signal to RISL's proc-macros that we're in the
            // second "regular Rust" compiler-pass.
            env::remove_var("IS_RISLC_PASS");

            // Set the `RISL_SHADER_REQUEST_LOOKUP` to the file path of this crate's
            // shader-request-lookup file. In this second "regular Rust" compiler-pass, RISL's
            // codegen request macros (e.g. `risl_macros::shader_wgsl`) may load this file and use
            // to resolve their shader request ID to the compiled shader artifact produced during
            // the first "RISL" compiler-pass.
            env::set_var(
                "RISL_SHADER_REQUEST_LOOKUP",
                shader_request_lookup.as_os_str(),
            );
        }
    }

    fn after_analysis<'tcx>(&mut self, _compiler: &Compiler, tcx: TyCtxt<'tcx>) -> Compilation {
        let is_proc_macro_crate = tcx
            .sess
            .opts
            .crate_types
            .iter()
            .any(|ty| *ty == CrateType::ProcMacro);

        // For proc-macro crates, we need to make the compiled proc-macro library available to the
        // RISL-pass as well. We cannot simply add the second pass's output dir to the search-paths
        // of the RISL-pass: transitive dependency proc-macro libraries are only resolved from
        // `PathKind::Dependency` search-path kinds. This unfortunately messes with the resolution
        // of non-proc-macro dependencies: in this case the RISL-pass may resolve regular
        // dependencies from the rustc-pass's output directory, rather than from the RISL-pass's
        // output directory (and search-path order cannot be used to control this). Therefore, the
        // solution for the time being is to (hard)link (or copy) the proc-macro library file into
        // the RISL-pass's output directory.
        //
        // However, at the time this hook runs, the proc-macro library file has not been written
        // yet. Unfortunately, there is currently not a callback hook after all output files have
        // been written, so instead we'll encode a future operation that we'll apply after the rustc
        // compilation pass has completed (see the end of the [run] function).
        //
        // TODO: it feels like there has to be a cleaner way to do this, but it would need to be
        // accommodated by the Rust project. Any of the following solutions would be an improvement:
        //
        // - Adjust the way rustc uses search-paths for file resolution, by making the search-path
        //   order relevant. If it finds two identical candidates in different search-paths, it
        //   would select the one from the search-path that is ordered earlier in the session
        //   config. We would add both the RISL output directory and the regular Rust output
        //   directory as "dependency" search-paths for the RISL-pass, the RISL dependency directory
        //   being the earlier search-path. This would mean `rlib` and `rmeta` artifacts would be
        //   resolved from the RISL output directory, and proc-macro libs would be resolved from
        //   the regular Rust output directory.
        // - Adjust the way rustc uses search paths for file resolution, by adding a extra
        //   `PathKind::ProcMacro` search-path kind, for paths that are only used for proc-macro
        //   resolution. We would add the regular Rust output directory as such a "proc-macro"
        //   search-path for the RISL-pass, and the RISL-output directory as the only "dependency"
        //   search-path. This would again mean `rlib` and `rmeta` artifacts would be resolved from
        //   the RISL output directory, and proc-macro libs would be resolved from the regular Rust
        //   output directory.
        // - Add an additional hook to the `Callbacks` trait that gets run after all of rustc's
        //   output has been written (e.g. `after_output`). We'd use the same link/copy mechanism we
        //   use now, but we would not have to do the song-and-dance of encoding the operation, then
        //   applying it later, as now we'd have a hook that runs after the proc-macro library file
        //   has already been written.
        //
        if is_proc_macro_crate {
            let src = out_filename(
                tcx.sess,
                CrateType::ProcMacro,
                tcx.output_filenames(()),
                tcx.crate_name(LOCAL_CRATE),
            );

            if let OutFileName::Real(src) = src {
                let parent = src.parent().map(|p| p.to_path_buf()).unwrap_or_default();
                let dst = risl_dir(&parent).join(src.file_name().unwrap_or_default());

                self.share_proc_macro_lib = Some(OpShareProcMacroLib { src, dst });
            }
        }

        Compilation::Continue
    }
}

struct RislPassCallbacks;

impl Callbacks for RislPassCallbacks {
    fn config(&mut self, config: &mut interface::Config) {
        // Don't print anything during the RISL pass, we'd end up double-printing in the second
        // pass.
        config.opts.prints = Vec::new();

        // Do not emit JSON artifact notifications during the RISL pass. Build systems (Cargo) use
        // these notifications to detect when artifacts have been written; we only want to notify
        // the build system once, after the second pass.
        config.opts.json_artifact_notifications = false;

        // Since we use a different compiler configuration for the RISL-pass (in particular w.r.t.
        // MIR generation), we can't share incremental compilation artifacts with the second
        // "regular rustc" pass. Therefore, if incremental compilation is enabled, use an
        // RISL-specific output directory.
        config.opts.incremental = config.opts.incremental.as_ref().map(risl_dir);

        // Output RISL artifacts to an RISL-specific output directory.
        config.output_dir = Some(risl_dir(
            config.output_dir.as_ref().unwrap_or(&PathBuf::new()),
        ));

        // Since we're outputting RISL artifacts to an RISL-specific output directory, we also need
        // to adjust the search_paths to look for dependencies in that RISL-specific output
        // directory.
        for search_path in &mut config.opts.search_paths {
            if search_path.kind == PathKind::Dependency {
                let dir = risl_dir(&search_path.dir);

                *search_path = SearchPath::new(PathKind::Dependency, dir);
            }
        }

        // SAFETY: rustc should not be running concurrent threads during Callbacks::config
        unsafe {
            // Set the `IS_RISLC_PASS` variable to signal to RISL's proc-macros that we're in the
            // first "RISL" compiler-pass.
            env::set_var("IS_RISLC_PASS", "1");
        }

        // Add a `rislc` cfg condition. We use this condition to control the tool attributes in the
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
        config.opts.cg.debug_assertions = Some(false);

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
fn risl_dir<P: AsRef<Path>>(base: P) -> PathBuf {
    let path = base.as_ref().join(RISL_OUTPUT_SUB_DIR);

    if !path.exists() {
        if let Err(err) = create_dir_all(&path) {
            let dcx = EarlyDiagCtxt::new(ErrorOutputType::default());

            dcx.early_fatal(format!("failed to create directory: {}", err));
        }
    }

    path
}

/// Returns a file path for an RISL-pass artifact based on the given file path for a Rust-pass
/// artifact.
fn risl_file<P: AsRef<Path>>(base: P) -> PathBuf {
    let filename = base.as_ref().file_name().expect("should be a file");
    let parent = base
        .as_ref()
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_default();

    parent.join(RISL_OUTPUT_SUB_DIR).join(filename)
}

/// Generates the file path for this crate's shader-request-lookup (SRL) file.
///
/// See also [create_shader_request_lookup].
fn shader_request_lookup_path<P: AsRef<Path>>(
    risl_output_dir: P,
    crate_name: &str,
    extra_filename: &str,
) -> PathBuf {
    let filename = format!("{}{}.srl", crate_name, extra_filename);

    risl_output_dir.as_ref().join(filename)
}

/// A shader-module-artifact-mapping for a dependency.
///
/// See also [RislContext::shader_module_artifact_mapping].
#[derive(Debug)]
struct Smam {
    krate: CrateNum,
    mapping: FxHashMap<u32, OsString>,
}

/// Loads the shader-module-artifact-mapping (SMAM) for the given `dependency`.
fn load_smam(cx: &RislContext, dependency: CrateNum) -> Smam {
    let filename = cx.resolve_risl_rlib_path(dependency);

    let mut archive = Archive::new(File::open(filename).unwrap());

    while let Some(Ok(mut entry)) = archive.next_entry() {
        if entry.header().identifier() == SMAM_HEADER.as_bytes() {
            let mut bytes = Vec::with_capacity(entry.header().size() as usize + 1);

            entry.read_to_end(&mut bytes).unwrap();

            let (mapping, _) = bincode::serde::decode_from_slice::<FxHashMap<u32, OsString>, _>(
                bytes.as_slice(),
                bincode::config::standard(),
            )
            .unwrap();

            return Smam {
                krate: dependency,
                mapping,
            };
        }
    }

    bug!(
        "failed to load shader module artifact mapping for crate `{}`",
        cx.tcx().crate_name(dependency)
    );
}

/// Creates an `rlib` for the current crate that includes additional RISL-specific metadata.
///
/// The `rlib` contains.
///
/// - The regular `rlib` metadata rustc would generate under the regular archive headers.
/// - A SLIR module that contains the SLIR-CFG intermediate representation for all "free items" in
///   the current crate under the header `lib.slir`. Dependent crates may use the SLIR module for
///   SLIR dependency resolution.
/// - A mapping for all public shader modules to their compilation artifact under the header
///   `lib.smam`. Dependent crates may use this mapping to fulfil shader
///   codegen requests.
///
fn create_rlib(
    tcx: TyCtxt,
    lib_module: &(slir::Module, slir::cfg::Cfg),
    smam: FxHashMap<u32, OsString>,
    shim_lookup: Option<&ShimDefLookup>,
) {
    let filename = out_filename(
        tcx.sess,
        CrateType::Rlib,
        tcx.output_filenames(()),
        tcx.crate_name(LOCAL_CRATE),
    );

    let out_file = File::create(filename.as_path()).unwrap();

    let mut headers = vec![
        METADATA_FILENAME.as_bytes().to_vec(),
        LIB_MODULE_HEADER.as_bytes().to_vec(),
        SMAM_HEADER.as_bytes().to_vec(),
    ];

    if shim_lookup.is_some() {
        headers.push(SHIM_LOOKUP_HEADER.as_bytes().to_vec());
    }

    let mut builder = GnuBuilder::new(out_file, headers);

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
            &Header::new(LIB_MODULE_HEADER.as_bytes().to_vec(), raw_lib.len() as u64),
            raw_lib.as_slice(),
        )
        .unwrap();

    let raw_smam = bincode::serde::encode_to_vec(smam, bincode::config::standard()).unwrap();

    builder
        .append(
            &Header::new(SMAM_HEADER.as_bytes().to_vec(), raw_smam.len() as u64),
            raw_smam.as_slice(),
        )
        .unwrap();

    if let Some(shim_lookup) = shim_lookup {
        let raw_shim =
            bincode::serde::encode_to_vec(shim_lookup, bincode::config::standard()).unwrap();

        builder
            .append(
                &Header::new(
                    SHIM_LOOKUP_HEADER.as_bytes().to_vec(),
                    raw_shim.len() as u64,
                ),
                raw_shim.as_slice(),
            )
            .unwrap();
    }

    builder.into_inner().unwrap();
}

/// Creates a shader-request-lookup (SRL) file in the output directory.
///
/// The lookup file maps shader-codegen-request-IDs - generated by shader codegen macros (such as
/// `risl_macros::shader_wgsl` during the first compiler pass - to artifact file paths. In the
/// second compiler pass, the shader codegen macro reads this SRL file to look up the path of the
/// shader artifact it requested.
fn create_shader_request_lookup(cx: &RislContext) {
    let filename = shader_request_lookup_path(
        cx.tcx()
            .sess
            .io
            .output_dir
            .as_ref()
            .unwrap_or(&PathBuf::new()),
        cx.tcx().crate_name(LOCAL_CRATE).as_str(),
        &cx.tcx().sess.opts.cg.extra_filename,
    );

    let mut lookup: HashMap<String, OsString> = HashMap::default();

    let mut requests = cx.hir_ext().shader_requests.clone();

    // Sort the requests by crate so that we minimize the number of times we have to load
    // shader-module-artifact-mappings.
    requests.sort_by(|a, b| a.shader_mod.krate.cmp(&b.shader_mod.krate));

    let mut active_smam: Option<Smam> = None;

    for shader_request in requests {
        let artifact_path = if let Some(shader_mod) = shader_request.shader_mod.as_local() {
            cx.shader_artifact_file_path(shader_mod)
        } else {
            if Some(shader_request.shader_mod.krate) != active_smam.as_ref().map(|v| v.krate) {
                active_smam = Some(load_smam(cx, shader_request.shader_mod.krate));
            }

            let smam = active_smam.as_ref().unwrap();

            let Some(path) = smam.mapping.get(&shader_request.shader_mod.index.as_u32()) else {
                bug!("dependency should have an entry");
            };

            PathBuf::from(path.clone())
        };

        lookup.insert(
            shader_request.request_id.to_string(),
            artifact_path.into_os_string(),
        );
    }

    let mut out_file = File::create(filename.as_path()).unwrap();

    bincode::serde::encode_into_std_write(&lookup, &mut out_file, bincode::config::standard())
        .unwrap();
}

fn compile_risl(tcx: TyCtxt) {
    let cx = RislContext::new(tcx);
    let lib_module = codegen_shader_modules(&cx);
    let smam = cx.local_smam();
    let shim_lookup = cx
        .current_crate_is_core_shim_crate()
        .then_some(cx.shim_def_lookup());

    create_rlib(tcx, &lib_module, smam, shim_lookup);
    create_shader_request_lookup(&cx);
}
