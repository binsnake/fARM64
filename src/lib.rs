//! # fARM64 — a pure-Rust AArch64 (A64) disassembler and semantic encoder
//!
//! `fARM64` decodes 64-bit ARM (AArch64 / A64) machine code into a rich,
//! `Copy` value-type [`Instruction`], renders it with a pluggable [`Formatter`],
//! and re-encodes instruction semantics with [`encode()`]. The Arm architectural
//! decode tree is hand-written from the Arm Architecture Reference Manual (the
//! "Arm ARM") and cross-checked during development against independent tools
//! and corpora. Apple AMX and GXF are implementation-defined exceptions based
//! on public reverse-engineering references and are separately runtime-gated.
//!
//! ## Portability is a hard, cross-cutting guarantee
//!
//! This crate is `#![no_std]` **unconditionally**. The core decode path and the
//! default formatter perform **zero heap allocation** and have **no dependency
//! on `alloc` or `std` at all**:
//!
//! * The decoder ([`Decoder::decode_into`]) writes into a caller-owned, `Copy`
//!   [`Instruction`]; there is no `Vec`, no `Box`, no internal pointers.
//! * The default formatter ([`format::FmtFormatter`]) writes into a
//!   caller-supplied [`core::fmt::Write`] sink **or** a fixed `&mut [u8]`
//!   buffer via [`format::BufSink`] (with overflow tracking, never overrun).
//! * All names (registers, mnemonics, system registers) are `&'static str`
//!   sourced from `const`/`static` tables.
//! * No thread-locals, no environment/IO/time access, no panics-as-control-flow,
//!   deterministic, and no floating-point *arithmetic* in the decoder.
//!
//! Builds cleanly for hosted targets, `wasm32-unknown-unknown`, and
//! `aarch64-unknown-none` (freestanding / no-CRT, no allocator).
//!
//! ## Feature matrix
//!
//! | Feature | Tier | Effect |
//! |-|-|-|
//! | *(none / default)* | A | `no_std`, **no `alloc`**, freestanding. Decoder + [`format::FmtFormatter`] + all enums. Always builds. |
//! | `alloc` | B | Adds `String`/`Vec` conveniences ([`format_to_string`], a cached [`info::InstructionInfoFactory`], and a token-collecting `String` sink). |
//! | `std` | C | Implies `alloc`; adds [`std::error::Error`] for [`DecodeError`] and std-only test/bench helpers. |
//! | `fmt-gnu` | A | Adds a UAL-equivalent GNU compatibility adapter. Pure `no_std`. |
//! | `sve` | A | Compiles the SVE/SVE2 decoder and encoder modules. |
//! | `sme` | A | Compiles the SME/SME2 decoder and encoder modules. |
//! | `crypto` | A | Compiles the Advanced SIMD crypto decoder; public enums and encoder support remain present. |
//! | `full` | A | Enables `sve`, `sme`, and `crypto`. |
//!
//! Cargo features decide which optional implementation modules are **compiled**;
//! the runtime [`FeatureSet`] decides what is **accepted** at decode time. Not
//! every runtime extension has a corresponding Cargo feature.
//!
//! ## Supported targets
//!
//! | Target | Notes |
//! |-|-|
//! | `x86_64-*`, `aarch64-*` (hosted) | development / `std` testing |
//! | `wasm32-unknown-unknown` | default features (`no_std`, no `alloc`) |
//! | `aarch64-unknown-none` | bare-metal, no-CRT; checked with `--no-default-features` |
//! | any target providing `core` | the default tier is `core`-only |
//!
//! ## Quick start (zero-alloc, `no_std`-friendly)
//!
//! ```no_run
//! use fARM64::{Decoder, DecoderOptions, format::{Formatter, FmtFormatter, BufSink}};
//!
//! // `ADD W0, W1, #1` little-endian; decode at a known address.
//! let code = [0x20, 0x04, 0x00, 0x11];
//! let mut dec = Decoder::new(&code, 0x1000, DecoderOptions::default());
//! let insn = dec.decode();
//!
//! // Format into a fixed stack buffer — no heap involved.
//! let mut buf = [0u8; 64];
//! let mut sink = BufSink::new(&mut buf);
//! FmtFormatter::new().format(&insn, &mut sink);
//! let _text: &str = sink.as_str();
//! ```
//!
//! ## Licensing & provenance
//!
//! Licensed under the MIT License. Arm architectural instruction handling is
//! based on the publicly documented Arm ARM. The implementation also contains
//! separately gated Apple implementation-defined AMX/GXF support based on
//! public reverse-engineering references: <https://github.com/corsix/amx>,
//! <https://asahilinux.org/docs/hw/cpu/apple-instructions/>, and
//! <https://blog.svenpeter.dev/posts/m1_sprr_gxf/>.

#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]
// Unsafe code is intentionally forbidden. Any future need for unsafe would
// require an explicit review of this crate-wide policy.
#![forbid(unsafe_code)]
#![deny(missing_debug_implementations)]
#![allow(clippy::result_unit_err)]
// `fARM64` is an intentional, stylized crate name (a play on "farm" + "ARM64").
// Allow the non-snake-case crate identifier; all in-crate items are snake_case.
#![allow(non_snake_case)]

// `std` is strictly additive (Tier C) and implies `alloc`.
#[cfg(feature = "std")]
extern crate std;

// `alloc` is strictly additive (Tier B). The default build links neither.
#[cfg(feature = "alloc")]
extern crate alloc;

// ---------------------------------------------------------------------------
// Module tree.
// ---------------------------------------------------------------------------

pub mod decoder;
pub mod enums;
pub mod error;
pub mod features;
pub mod format;
pub mod info;
pub mod instruction;
pub mod mnemonic;
pub mod operand;
pub mod register;
pub mod sysop;
pub mod sysreg;

/// The hand-written recursive A64 decode tree (derived from the ARM ARM).
///
/// Top-level dispatch on `op0 = word<28:25>` into per-group decoders that match
/// sub-fields and build the [`Instruction`] directly; ARM shared pseudocode
/// (`DecodeBitMasks`, `AdvSIMDExpandImm`, ...) lives in
/// [`decode::bits`]. Zero-alloc and panic-free.
pub mod decode;

/// The hand-written A64 **encoder** — the inverse of [`decode`].
///
/// Reconstructs the raw 32-bit instruction word from the *semantics* of an
/// [`Instruction`] (its [`Code`]/[`Mnemonic`]/operands/`ip`), never reading
/// [`Instruction::word`]. Dispatches on [`Instruction::code`] to per-group
/// encoders. `no_std`, zero-alloc, and total (returns [`encode::EncodeError`]
/// rather than panicking); unsupported codes and semantic operand shapes return
/// an error.
pub mod encode;

/// Mechanical name / enum lookup tables (the `&'static str` register, condition,
/// and system-register name tables). These are maintained as committed Rust
/// source and contain no decode logic. The workspace's `xtask` is currently only
/// a scaffold for a possible future generator.
pub mod tables;

// ---------------------------------------------------------------------------
// Public prelude re-exports — the idiomatic projection users program against.
// The fat internal representation never reaches this surface.
// ---------------------------------------------------------------------------

pub use crate::decoder::{Decoder, DecoderIntoIter, DecoderIter, DecoderOptions};
pub use crate::encode::{encode, EncodeError};
pub use crate::enums::{
    Condition, ExtendType, FlagEffect, FlowControl, ShiftType, VectorArrangement,
};
pub use crate::error::DecodeError;
pub use crate::features::{Feature, FeatureSet};
pub use crate::format::{
    Formatter, FormatterOptions, FormatterOutput, SymbolResolver, SymbolResult, TokenKind,
};
pub use crate::info::{instruction_info, InstructionInfo, OpAccess, UsedMemory, UsedRegister};
pub use crate::instruction::Instruction;
pub use crate::mnemonic::Code;
pub use crate::mnemonic::Mnemonic;
pub use crate::operand::{MemIndexMode, OpKind, Operand, PredQual, SliceIndicator, SveMemMode};
pub use crate::register::{gp_register, RegClass, RegWidth, Register};
pub use crate::sysreg::SystemReg;

#[cfg(feature = "alloc")]
pub use crate::format::format_to_string;

/// Compile-time invariants. These are zero-cost assertions evaluated by the
/// compiler; a violation is a build error, never a runtime check.
mod static_asserts {
    use crate::instruction::Instruction;
    use crate::operand::Operand;

    /// Const-evaluated assertion helper (works on the pinned MSRV without any
    /// external `static_assertions` dependency).
    macro_rules! const_assert {
        ($($tt:tt)*) => {
            const _: [(); 0 - !{ const ASSERT: bool = $($tt)*; ASSERT } as usize] = [];
        };
    }

    // Each rich `Operand` is `Copy` and compact (16 bytes on a 64-bit target);
    // the whole `[Operand; MAX_OPERANDS]` lives inline in `Instruction` with no
    // heap indirection. Tightening this ceiling is a deliberate review gate.
    const_assert!(core::mem::size_of::<Operand>() <= 16);

    // The public `Instruction` is a `Copy` value type whose size is dominated by
    // its inline `[Operand; MAX_OPERANDS]` (5 * 16 = 80 bytes) plus the small
    // header (word/ip/code/mnemonic/op_count/flags). Keep the rich,
    // allocation-free value type and assert its realized ceiling.
    const_assert!(core::mem::size_of::<Instruction>() <= 112);

    /// `Copy` witness — fails to compile if either type loses `Copy`.
    #[allow(dead_code)]
    fn _assert_copy<T: Copy>() {}
    #[allow(dead_code)]
    fn _witness() {
        _assert_copy::<Instruction>();
        _assert_copy::<Operand>();
    }
}

/// The maximum number of explicit operands any A64 encoding produces.
///
/// A few SIMD list / SME forms reach five operand slots; the inline
/// `[Operand; MAX_OPERANDS]` in [`Instruction`] is sized to this.
pub const MAX_OPERANDS: usize = 5;

/// Fixed A64 instruction length, in bytes. A64 is a fixed-width ISA: every
/// instruction is exactly 4 bytes and 4-byte aligned, and `PC` always advances
/// by 4. Exposed for callers that stride a buffer.
pub const INSN_LEN: usize = 4;
