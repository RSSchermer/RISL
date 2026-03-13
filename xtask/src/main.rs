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
    RunBehavioralTests {
        /// The list of features to run
        #[arg(short, long, value_delimiter = ',')]
        features: Vec<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let sh = Shell::new()?;

    match &cli.command {
        Commands::RunBehavioralTests { features } => run_behavioral_tests(&sh, features),
    }
}

fn run_behavioral_tests(sh: &Shell, features: &[String]) -> Result<()> {
    println!("Recompiling rislc to ensure it is up to date with the current source...");
    cmd!(sh, "cargo build --bin=rislc")
        .env("RUSTFLAGS", "-Awarnings")
        .run()?;

    println!("Running behavioral tests...");

    let mut cargo_test = cmd!(sh, "cargo test --package=behavioral-tests");

    if features.is_empty() {
        cargo_test = cargo_test.arg("--all-features");
    } else {
        cargo_test = cargo_test.arg("--features").arg(features.join(","));
    }

    cargo_test
        .env("RUSTFLAGS", "-Awarnings")
        .env("RUST_BACKTRACE", "full")
        .env("RUSTC_WRAPPER", "target/debug/rislc")
        .run()?;

    Ok(())
}
