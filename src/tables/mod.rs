//! Mechanical name / enum lookup tables.
//!
//! Everything under this module is **purely mechanical data**: the `&'static
//! str` register / condition / system-register name tables that back the public
//! `name()` accessors. It contains no decode *logic* — the decoder is the
//! hand-written tree under [`crate::decode`].
//!
//! The tables are generated offline by `cargo xtask gen` from a curated,
//! ARM-spec-derived dataset (instruction mnemonics, register and condition
//! spellings, the system-register `op0/op1/CRn/CRm/op2` directory) and
//! **committed**, so downstream builds are hermetic (no `build.rs`, no XML, no
//! network). `xtask` does not parse any third-party decoder source and emits no
//! decode logic. These files may be hand-edited until the generator is wired up.

pub mod names;
pub mod sysins;
