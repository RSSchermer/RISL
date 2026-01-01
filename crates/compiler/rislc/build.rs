use std::env;
use std::process::{Command, ExitCode};

use anyhow::Context;

const EXPECTED_COMMIT_HASH: &str = "f5209000832c9d3bc29c91f4daef4ca9f28dc797";
const EXPECTED_COMMIT_DATE: &str = "2025-12-12";

fn setup_instructions() -> String {
    format!("\
The RISL compiler integrates with Rust compiler internals. As such, a specific version of the Rust \
toolchain is required to build the compiler. For this version of `risl-cargo` that required \
toolchain version is:

  nightly-{EXPECTED_COMMIT_DATE}

Additionally, the following Rustup components must be installed for that toolchain:

  rust-src
  rustc-dev
  llvm-tools-preview

If you are working on a Cargo project that uses RISL, the recommended way of ensuring these \
prerequisites are met is to add a `rust-toolchain.toml` file in the root directory of your \
project, and any subdirectory from which you wish to run Cargo commands (Rustup will only look for \
a `rust-toolchain.toml` in the command's current working directory). A minimal \
`rust-toolchain.toml` file would look like this:

  [toolchain]
  channel = \"nightly-{EXPECTED_COMMIT_DATE}\"
  components = [\"rust-src\", \"rustc-dev\", \"llvm-tools-preview\"]

You may of course add additional components and specify additional toolchain requirements as needed.

Alternatively, you can install the required toolchain and components manually using the following \
command:

  rustup toolchain install nightly-{EXPECTED_COMMIT_DATE} --component rust-src,rustc-dev,llvm-tools-preview

Then, to activate the toolchain for a given directory, run the following command while in that \
directory:

  rustup override set nightly-{EXPECTED_COMMIT_DATE}

For additional information on the `rust-toolchain.toml` file and toolchain selection, refer to the \
\"Overrides\" section of the Rustup documentation: https://rust-lang.github.io/rustup/overrides.html\
")
}

fn verify_rustc_version() -> anyhow::Result<()> {
    let rustc_path = env::var("RUSTC")?;
    let output = Command::new(rustc_path).arg("-vV").output()?;
    let raw_info = String::from_utf8(output.stdout)?;

    let mut release = None;
    let mut commit_date = None;
    let mut commit_hash = None;

    for line in raw_info.lines() {
        if let Some(value) = line.strip_prefix("release:") {
            release = Some(value.trim().to_string());
        }

        if let Some(value) = line.strip_prefix("commit-date:") {
            commit_date = Some(value.trim().to_string());
        }

        if let Some(value) = line.strip_prefix("commit-hash:") {
            commit_hash = Some(value.trim().to_string());
        }
    }

    let Some(release) = release else {
        anyhow::bail!("Could not find release version in rustc -vV output");
    };

    let Some(commit_date) = commit_date else {
        anyhow::bail!("Could not find commit-date in rustc -vV output");
    };

    let Some(commit_hash) = commit_hash else {
        anyhow::bail!("Could not find commit-hash in rustc -vV output");
    };

    if commit_hash != EXPECTED_COMMIT_HASH {
        anyhow::bail!(
            "Unsupported rustc version: {release} ({commit_date} {commit_hash})\n\n{}",
            setup_instructions()
        );
    }

    Ok(())
}

struct RequiredComponents {
    rust_src: bool,
    rust_dev: bool,
    llvm_tools: bool,
}

impl RequiredComponents {
    fn missing_components(&self) -> Vec<&str> {
        let mut missing = Vec::new();

        if !self.rust_src {
            missing.push("rust-src");
        }

        if !self.rust_dev {
            missing.push("rustc-dev");
        }

        if !self.llvm_tools {
            missing.push("llvm-tools-preview");
        }

        missing
    }
}

fn verify_installed_components() -> anyhow::Result<()> {
    let output = Command::new("rustup")
        .args(["component", "list", "--installed"])
        .output()
        .context(
            "\
Failed to run `rustup` command.

Building the RISL compiler requires rustup, make sure it is installed \
(https://rust-lang.org/tools/install/) and available on the PATH.",
        )?;

    let raw_info = String::from_utf8(output.stdout)?;

    let mut components = RequiredComponents {
        rust_src: false,
        rust_dev: false,
        llvm_tools: false,
    };

    for component in raw_info.lines() {
        if component.starts_with("rust-src") {
            components.rust_src = true;
        } else if component.starts_with("rustc-dev") {
            components.rust_dev = true;
        } else if component.starts_with("llvm-tools") {
            components.llvm_tools = true;
        }
    }

    let missing = components.missing_components();

    if !missing.is_empty() {
        anyhow::bail!(
            "Missing required Rustup components: {}\n\n{}",
            missing.join(", "),
            setup_instructions()
        );
    }

    Ok(())
}

fn main_impl() -> anyhow::Result<()> {
    verify_rustc_version()?;
    verify_installed_components()?;

    let origin = if cfg!(target_os = "macos") {
        "@loader_path"
    } else {
        "$ORIGIN"
    };
    println!("cargo:rustc-link-arg-bin=rislc=-Wl,-rpath,{origin}/../toolchain/lib");

    Ok(())
}

fn main() -> ExitCode {
    match main_impl() {
        Ok(_) => ExitCode::SUCCESS,
        Err(err) => {
            println!("cargo::error={}", err);

            ExitCode::FAILURE
        }
    }
}
