//! `xtask` — an optional, host-only generator for fARM64's *mechanical* lookup
//! tables.
//!
//! This tool is **not** part of the published library and never runs as part of
//! a downstream build (the library is hermetic: no `build.rs`, no network). It
//! exists only as a developer convenience to (re)emit the purely mechanical
//! data from a curated, ARM-spec-derived dataset:
//!
//! * the large `Code` / `Mnemonic` enums (`../src/mnemonic.rs`), and
//! * the `&'static str` register / condition / system-register name tables
//!   (`../src/tables/names.rs`).
//!
//! It does **not** generate any decode *logic* — the decoder is the hand-written
//! recursive tree under `../src/decode/`, authored from the ARM Architecture
//! Reference Manual. The generated output is committed and diffable; the emit
//! bodies below are stubs to be implemented.
//!
//! Subcommand:
//!
//! * `gen` — emit `../src/mnemonic.rs` (enum variants) and
//!   `../src/tables/names.rs` (name tables) from the curated dataset.

use std::process::ExitCode;

/// The curated, ARM-spec-derived input dataset consumed by the emitter.
///
/// Populated from a committed description of the A64 mnemonic / encoding space
/// (instruction names, register and condition spellings, and the
/// system-register `op0/op1/CRn/CRm/op2` directory). Fields are filled in as the
/// generator stages are implemented.
#[derive(Debug, Default)]
struct Dataset {
    /// One entry per `Code` / `Mnemonic` enum variant to emit.
    mnemonics: Vec<MnemonicDef>,
    /// `(packed_key, name)` rows for the system-register name table.
    sysregs: Vec<SysRegDef>,
}

/// One curated mnemonic / encoding identity (drives a `Code` and/or `Mnemonic`
/// enum variant and its name-table row).
#[derive(Debug, Default)]
#[allow(dead_code)] // fields consumed once the generator stages are implemented
struct MnemonicDef {
    /// The enum variant identifier (e.g. `AddImm64`).
    variant: String,
    /// The rendered mnemonic spelling (e.g. `add`).
    name: String,
}

/// One curated system-register directory entry.
#[derive(Debug, Default)]
#[allow(dead_code)] // consumed once the generator stages are implemented
struct SysRegDef {
    /// Packed `op0/op1/CRn/CRm/op2` key.
    packed: u16,
    /// The register's canonical name (e.g. `nzcv`).
    name: String,
}

/// Emits the mechanical Rust source (enums + name tables) from the [`Dataset`].
#[derive(Debug, Default)]
struct Emitter;

impl Emitter {
    fn emit_all(&self, _data: &Dataset) -> std::io::Result<()> {
        // Write ../src/mnemonic.rs (Code/Mnemonic variants) and
        // ../src/tables/names.rs (register/condition/sysreg name tables) from the
        // curated dataset. No decode logic is emitted here.
        unimplemented!("xtask gen: emit mechanical enum + name tables")
    }
}

fn cmd_gen() -> std::io::Result<()> {
    let data = Dataset::default();
    let _ = (&data.mnemonics, &data.sysregs, MnemonicDef::default(), SysRegDef::default());
    Emitter.emit_all(&data)
}

fn usage() {
    eprintln!("usage: cargo xtask gen");
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("gen") => match cmd_gen() {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("xtask gen failed: {e}");
                ExitCode::FAILURE
            }
        },
        _ => {
            usage();
            ExitCode::FAILURE
        }
    }
}
