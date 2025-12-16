use std::sync::Arc;

use cargo::core::compiler::{CompileMode, Executor, UserIntent};
use cargo::core::{PackageId, Shell, Target};
use cargo::util::command_prelude::{
    Arg, ArgAction, ArgMatches, ArgMatchesExt, Command, CommandExt, ProfileChecking, flag, heading,
    multi_opt, opt, subcommand,
};
use cargo::{CargoResult, CliResult, GlobalContext, ops};
use cargo_util::ProcessBuilder;

const SUBCOMMAND: &str = "risl-build";

pub fn base_cli() -> Command {
    // Sets up the base cargo command. We don't copy the full set of options of the actual cargo
    // command here, only the subset of options that are used to set up the GlobalContext properly.
    Command::new("cargo")
        .arg(
            opt(
                "verbose",
                "Use verbose output (-vv very verbose/build.rs output)",
            )
            .short('v')
            .action(ArgAction::Count)
            .global(true),
        )
        .arg(
            flag("quiet", "Do not print cargo log messages")
                .short('q')
                .global(true),
        )
        .arg(
            opt("color", "Coloring")
                .value_name("WHEN")
                .global(true)
                .value_parser(["auto", "always", "never"])
                .ignore_case(true),
        )
        .arg(
            flag("locked", "Assert that `Cargo.lock` will remain unchanged")
                .help_heading(heading::MANIFEST_OPTIONS)
                .global(true),
        )
        .arg(
            flag("offline", "Run without accessing the network")
                .help_heading(heading::MANIFEST_OPTIONS)
                .global(true),
        )
        .arg(
            flag(
                "frozen",
                "Equivalent to specifying both --locked and --offline",
            )
            .help_heading(heading::MANIFEST_OPTIONS)
            .global(true),
        )
        .arg(multi_opt("config", "KEY=VALUE|PATH", "Override a configuration value").global(true))
        .arg(
            Arg::new("unstable-features")
                .help("Unstable (nightly-only) flags to Cargo, see 'cargo -Z help' for details")
                .short('Z')
                .value_name("FLAG")
                .action(ArgAction::Append)
                .global(true),
        )
}

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

    base_cli().subcommand(subcommand)
}

struct RislExecutor;

impl Executor for RislExecutor {
    fn exec(
        &self,
        cmd: &ProcessBuilder,
        _id: PackageId,
        _target: &Target,
        _mode: CompileMode,
        on_stdout_line: &mut dyn FnMut(&str) -> CargoResult<()>,
        on_stderr_line: &mut dyn FnMut(&str) -> CargoResult<()>,
    ) -> CargoResult<()> {
        dbg!(cmd);

        cmd.exec_with_streaming(on_stdout_line, on_stderr_line, false)
            .map(drop)
    }
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

    let exec = Arc::new(RislExecutor) as Arc<dyn Executor>;

    ops::compile_with_exec(&ws, &compile_opts, &exec)?;

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
