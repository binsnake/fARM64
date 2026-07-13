//! Mechanical name / enum lookup tables.
//!
//! Everything under this module is **purely mechanical data**: the `&'static
//! str` register / condition / system-register name tables that back the public
//! `name()` accessors. It contains no decode *logic* — the decoder is the
//! hand-written tree under [`crate::decode`].
//!
//! The tables are maintained directly as committed Rust source, so downstream
//! builds are hermetic (no `build.rs`, no XML, no network). The workspace's
//! `xtask` is only an unfinished scaffold and does not currently generate or
//! rewrite these files.

pub mod names;
pub mod sysins;
