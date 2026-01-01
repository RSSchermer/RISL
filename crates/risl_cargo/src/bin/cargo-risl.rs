use std::env;
use std::process::{Command, ExitCode};

const HELP_TEXT: &str = "\
Usage: cargo risl <check|build|run|test> [args...]

Example:
  cargo risl build --release

Runs the corresponding 'cargo <cmd>' using the RISL compiler.

For each of <check|build|run|test> respectively, accepts the same [args...] as the corresponding
'cargo <cmd>'. For more details, run 'cargo risl <cmd> --help' or refer to the Cargo documentation
(https://doc.rust-lang.org/cargo/).\
";

const SUBCOMMANDS: &[&str] = &["check", "build", "run", "test"];

fn main() -> ExitCode {
    let mut args = env::args().skip(1);

    let Some("risl") = args.next().as_deref() else {
        eprintln!("Must be used in a Cargo context, e.g.: cargo risl <cmd> [args...]");

        return ExitCode::FAILURE;
    };

    let Some(subcommand) = args.next() else {
        eprintln!("{}", HELP_TEXT);

        return ExitCode::FAILURE;
    };

    if matches!(subcommand.as_str(), "-h" | "--help" | "help") {
        println!("{}", HELP_TEXT);
    }

    if !SUBCOMMANDS.contains(&subcommand.as_str()) {
        eprintln!(
            "Unsupported subcommand '{subcommand}'. Supported: {}\n\n{HELP_TEXT}",
            SUBCOMMANDS.join(", ")
        );

        return ExitCode::FAILURE;
    }

    // Prepare and run: cargo <cmd> [args...]
    match Command::new("cargo")
        .arg(&subcommand)
        .args(args)
        .env("RUSTC_WRAPPER", "rislc")
        .status()
    {
        Ok(status) => {
            if status.success() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(err) => {
            eprintln!("Failed to execute 'cargo {subcommand} ...': {err}");

            ExitCode::FAILURE
        }
    }
}
