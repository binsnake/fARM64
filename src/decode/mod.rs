//! Hand-written recursive A64 decode tree (derived from the ARM ARM).
//!
//! This module replaces the previous generic table-interpreter with a
//! hand-written decoder transcribed from the *ARM Architecture Reference Manual*
//! top-level encoding classification (ARM ARM C4.1, "A64 instruction set
//! encoding"). There is no bytecode and no table walk on the hot path: the entry
//! point [`decode_into`] dispatches on `op0 = word<28:25>` to one of the eight
//! A64 encoding groups, and each group decoder (in the sibling modules) matches
//! its own sub-fields and builds the [`Instruction`] directly.
//!
//! Field extraction and the ARM shared pseudocode (`DecodeBitMasks`,
//! `AdvSIMDExpandImm`, `VFPExpandImm`, `Replicate`, `HighestSetBit`,
//! sign-extension, ...) are plain hand-written functions in [`bits`].
//!
//! Top-level `op0` map (ARM ARM C4.1, table "Main encoding table"):
//!
//! | `op0` (`word<28:25>`) | Encoding group |
//! |-|-|
//! | `0000` | Reserved (and `UDF`) |
//! | `0001` | Unallocated |
//! | `0010` | SVE encodings (feature-gated) |
//! | `0011` | Unallocated |
//! | `100x` | Data Processing -- Immediate |
//! | `101x` | Branches, Exception generating and System |
//! | `x1x0` | Loads and Stores |
//! | `x101` | Data Processing -- Register |
//! | `x111` | Data Processing -- Scalar FP and Advanced SIMD |

pub mod bits;

pub mod branch_sys;
pub mod dp_imm;
pub mod dp_reg;
pub mod ldst;
pub mod ldst_simd;
pub mod mops;
pub mod simd_fp;
#[cfg(feature = "sme")]
pub mod sme;
#[cfg(feature = "sve")]
pub mod sve;

use crate::features::FeatureSet;
use crate::instruction::Instruction;
use crate::INSN_LEN;

/// Decode a single A64 instruction `word` at address `ip` into `out`.
///
/// This is the hand-written decode-tree entry point that
/// [`crate::Decoder::decode_into`] delegates to. It resets `out` to a clean
/// state, dispatches on the top-level `op0 = word<28:25>` field to the matching
/// group decoder, and leaves `out` as the [`Code::Invalid`](crate::Code::Invalid)
/// default for reserved or unallocated encodings.
///
/// Pure, total, allocation-free, and panic-free for all 2^32 inputs.
#[inline]
pub fn decode_into(word: u32, ip: u64, features: FeatureSet, out: &mut Instruction) {
    // Start from a clean, invalid instruction carrying this word/ip so that an
    // unhandled or reserved encoding surfaces as `Code::Invalid` with the
    // correct address and length, never stale data.
    *out = Instruction::new_invalid(word, ip);

    // op0 = word<28:25> selects the eight top-level A64 encoding groups.
    let op0 = (word >> 25) & 0xF;
    match op0 {
        // 0000 Reserved (UDF lives here); 0001/0011 unallocated.
        0b0000 => decode_reserved(word, ip, features, out),
        0b0001 | 0b0011 => { /* unallocated: leave invalid */ }

        // 0010 SVE encodings (only when compiled in; runtime-gated inside).
        0b0010 => decode_sve(word, ip, features, out),

        // 100x Data Processing -- Immediate.
        0b1000 | 0b1001 => dp_imm::decode(word, ip, features, out),

        // 101x Branches, Exception generating and System.
        0b1010 | 0b1011 => branch_sys::decode(word, ip, features, out),

        // x1x0 Loads and Stores (0100, 0110, 1100, 1110).
        0b0100 | 0b0110 | 0b1100 | 0b1110 => ldst::decode(word, ip, features, out),

        // x101 Data Processing -- Register (0101, 1101).
        0b0101 | 0b1101 => dp_reg::decode(word, ip, features, out),

        // x111 Data Processing -- Scalar FP & Advanced SIMD (0111, 1111).
        0b0111 | 0b1111 => simd_fp::decode(word, ip, features, out),

        // The 4-bit `op0` is fully covered above; this arm is unreachable but
        // keeps the match total without a panic.
        _ => {}
    }

    debug_assert_eq!(out.len(), INSN_LEN);
}

/// Decode a single A64 instruction and return it by value.
///
/// Convenience wrapper over [`decode_into`] for callers that do not reuse an
/// [`Instruction`] buffer.
#[inline]
pub fn decode(word: u32, ip: u64, features: FeatureSet) -> Instruction {
    let mut out = Instruction::default();
    decode_into(word, ip, features, &mut out);
    out
}

/// Reserved encoding space (`op0 == 0b0000`), including the permanently
/// undefined `UDF` encoding `word<31:16> == 0` (ARM ARM C4.1.1).
///
/// Only the `UDF` pattern (`word<31:16> == 0`) is allocated here; it renders
/// `udf #<imm16>` with `imm16 = word<15:0>`. Every other reserved word is left
/// [`crate::mnemonic::Code::Invalid`]. Total and panic-free.
#[inline]
fn decode_reserved(word: u32, ip: u64, features: FeatureSet, out: &mut Instruction) {
    use crate::mnemonic::Code;
    use crate::operand::Operand;
    let _ = (ip, features);

    // UDF: the permanently-undefined encoding has all of bits<31:16> clear.
    if (word >> 16) == 0 {
        out.set(Code::Udf);
        out.push_operand(Operand::ImmUnsigned((word & 0xffff) as u64));
        return;
    }

    // SME (Scalable Matrix Extension) shares this reserved `op0 == 0b0000`
    // region: its outer-product, MOVA, ADDHA/ADDVA and ZA load/store encodings
    // all have `word<31> == 1` (the UDF pattern above has `word<31> == 0`). Only
    // meaningful when the `sme` cargo feature is compiled in and the runtime
    // `FeatureSet` accepts it; otherwise the word is left invalid.
    if (word >> 31) == 1 {
        decode_sme(word, ip, features, out);
    }
    // Otherwise leave `out` invalid (other reserved encodings are unallocated).
}

/// SME encoding sub-region of the reserved group (`op0 == 0b0000`,
/// `word<31> == 1`). Compiled only with the `sme` cargo feature; runtime-gated
/// on [`crate::Feature::Sme`] inside.
#[inline]
fn decode_sme(word: u32, ip: u64, features: FeatureSet, out: &mut Instruction) {
    #[cfg(feature = "sme")]
    sme::decode(word, ip, features, out);
    #[cfg(not(feature = "sme"))]
    let _ = (word, ip, features, out);
}

/// SVE encoding group (`op0 == 0b0010`). Only meaningful when the `sve` cargo
/// feature is compiled in and the runtime [`FeatureSet`] accepts it; otherwise
/// the word is left invalid.
#[inline]
fn decode_sve(word: u32, ip: u64, features: FeatureSet, out: &mut Instruction) {
    #[cfg(feature = "sve")]
    sve::decode(word, ip, features, out);
    #[cfg(not(feature = "sve"))]
    let _ = (word, ip, features, out);
}
