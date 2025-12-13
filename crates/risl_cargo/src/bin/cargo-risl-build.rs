use cargo::core::Shell;
use cargo::core::compiler::UserIntent;
use cargo::util::command_prelude::{
    ArgMatches, ArgMatchesExt, Command, CommandExt, ProfileChecking, subcommand,
};
use cargo::{CliResult, GlobalContext, ops};

const SUBCOMMAND: &str = "risl-build";

pub fn cli() -> Command {
    let subcommand = Command::new(SUBCOMMAND)
        .about("Compile a local package and all of its dependencies with the RISL compiler")
        .arg_future_incompat_report()
        .arg_message_format()
        .arg_silent_suggestion()
        .arg_package_spec(
            "Package to build (see `cargo help pkgid`)",
            "Build all packages in the workspace",
            "Exclude packages from the build",
        )
        .arg_targets_all(
            "Build only this package's library",
            "Build only the specified binary",
            "Build all binaries",
            "Build only the specified example",
            "Build all examples",
            "Build only the specified test target",
            "Build all targets that have `test = true` set",
            "Build only the specified bench target",
            "Build all targets that have `bench = true` set",
            "Build all targets",
        )
        .arg_features()
        .arg_release("Build artifacts in release mode, with optimizations")
        .arg_redundant_default_mode("debug", "build", "release")
        .arg_profile("Build artifacts with the specified profile")
        .arg_parallel()
        .arg_target_triple("Build for the target triple")
        .arg_target_dir()
        .arg_artifact_dir()
        .arg_unit_graph()
        .arg_timings()
        .arg_compile_time_deps()
        .arg_manifest_path()
        .arg_lockfile_path()
        .arg_ignore_rust_version()
        .after_help(color_print::cstr!(
            "Run `<bright-cyan,bold>cargo help build</>` for more detailed information.\n"
        ));

    Command::new("cargo").subcommand(subcommand)
}

pub fn exec(gctx: &mut GlobalContext, args: &ArgMatches) -> CliResult {
    let ws = args.workspace(gctx)?;
    let mut compile_opts =
        args.compile_options(gctx, UserIntent::Build, Some(&ws), ProfileChecking::Custom)?;

    if let Some(artifact_dir) = args.value_of_path("artifact-dir", gctx) {
        // If the user specifies `--artifact-dir`, use that
        compile_opts.build_config.export_dir = Some(artifact_dir);
    } else if let Some(artifact_dir) = args.value_of_path("out-dir", gctx) {
        // `--out-dir` is deprecated, but still supported for now
        gctx.shell()
            .warn("the --out-dir flag has been changed to --artifact-dir")?;
        compile_opts.build_config.export_dir = Some(artifact_dir);
    } else if let Some(artifact_dir) = gctx.build_config()?.artifact_dir.as_ref() {
        // If a CLI option is not specified for choosing the artifact dir, use the `artifact-dir` from the build config, if
        // present
        let artifact_dir = artifact_dir.resolve_path(gctx);
        compile_opts.build_config.export_dir = Some(artifact_dir);
    } else if let Some(artifact_dir) = gctx.build_config()?.out_dir.as_ref() {
        // As a last priority, check `out-dir` in the build config
        gctx.shell()
            .warn("the out-dir config option has been changed to artifact-dir")?;
        let artifact_dir = artifact_dir.resolve_path(gctx);
        compile_opts.build_config.export_dir = Some(artifact_dir);
    }

    if compile_opts.build_config.export_dir.is_some() {
        gctx.cli_unstable()
            .fail_if_stable_opt("--artifact-dir", 6790)?;
    }

    ops::compile(&ws, &compile_opts)?;
    Ok(())
}

fn main() {
    let mut gctx = match GlobalContext::default() {
        Ok(gctx) => gctx,
        Err(e) => {
            let mut shell = Shell::new();
            cargo::exit_with_error(e.into(), &mut shell)
        }
    };

    run_cli(&mut gctx, cli(), SUBCOMMAND)
        .unwrap_or_else(|e| cargo::exit_with_error(e, &mut gctx.shell()));
}

fn run_cli(gctx: &mut GlobalContext, cli: Command, subcommand: &str) -> CliResult {
    let args = cli.try_get_matches()?;
    let subcommand_args = args.subcommand_matches(subcommand).unwrap();

    configure_gctx(gctx, &args, Some(subcommand_args), GlobalArgs::default())?;

    exec(gctx, subcommand_args)
}

#[derive(Default)]
struct GlobalArgs {
    verbose: u32,
    quiet: bool,
    color: Option<String>,
    frozen: bool,
    locked: bool,
    offline: bool,
    unstable_flags: Vec<String>,
    config_args: Vec<String>,
}

fn configure_gctx(
    gctx: &mut GlobalContext,
    args: &ArgMatches,
    subcommand_args: Option<&ArgMatches>,
    global_args: GlobalArgs,
) -> CliResult {
    let arg_target_dir = &subcommand_args.and_then(|a| a.value_of_path("target-dir", gctx));
    let verbose = global_args.verbose + args.verbose();
    // quiet is unusual because it is redefined in some subcommands in order
    // to provide custom help text.
    let quiet = args.flag("quiet")
        || subcommand_args.map(|a| a.flag("quiet")).unwrap_or_default()
        || global_args.quiet;
    let global_color = global_args.color; // Extract so it can take reference.
    let color = args
        .get_one::<String>("color")
        .map(String::as_str)
        .or_else(|| global_color.as_deref());
    let frozen = args.flag("frozen") || global_args.frozen;
    let locked = args.flag("locked") || global_args.locked;
    let offline = args.flag("offline") || global_args.offline;
    let mut unstable_flags = global_args.unstable_flags;
    if let Some(values) = args.get_many::<String>("unstable-features") {
        unstable_flags.extend(values.cloned());
    }
    let mut config_args = global_args.config_args;
    if let Some(values) = args.get_many::<String>("config") {
        config_args.extend(values.cloned());
    }
    gctx.configure(
        verbose,
        quiet,
        color,
        frozen,
        locked,
        offline,
        arg_target_dir,
        &unstable_flags,
        &config_args,
    )?;
    Ok(())
}
