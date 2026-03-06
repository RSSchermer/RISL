use anyhow::Result;
use clap::Parser;
use xshell::{Shell, cmd};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Runs behavioral tests
    RunBehavioralTests,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let sh = Shell::new()?;

    match &cli.command {
        Commands::RunBehavioralTests => run_behavioral_tests(&sh),
    }
}

fn run_behavioral_tests(sh: &Shell) -> Result<()> {
    println!("Recompiling rislc to ensure it is up to date with the current source...");
    cmd!(sh, "cargo build --bin=rislc")
        .env("RUSTFLAGS", "-Awarnings")
        .run()?;

    println!("Running behavioral tests...");
    cmd!(sh, "cargo test --package=behavioral-tests")
        .env("RUSTFLAGS", "-Awarnings")
        .env("RUST_BACKTRACE", "full")
        .env("RUSTC_WRAPPER", "target/debug/rislc")
        .run()?;

    Ok(())
}
