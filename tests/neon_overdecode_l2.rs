//! Advanced-SIMD over-decode hardening, batch L2 — integer three-same.
//!
//! Two reserved-encoding families in the Advanced-SIMD *integer* three-same
//! space, each derived from an LLVM field sweep (`oracle.py dec/enc`, i.e.
//! `clang .inst` + `llvm-objdump --mattr=+all`) and proven 0-regression with a
//! pre/post differential over the affected top bytes (`0x0E`/`0x2E`/`0x4E`/
//! `0x6E` vector, `0x5E`/`0x7E` scalar): a structured sweep over all
//! `(Q,U,size,opcode)` × 4 register samples eliminated 64 over-decoded words
//! with **0** LLVM-valid words newly rejected, **0** Invalid→valid flips, and
//! **0** valid-word text changes. Every newly-rejected word is `<unknown>` in
//! LLVM.
//!
//! Families hardened (both in `src/decode/simd_fp/simd_arith.rs`):
//!
//!  1. **Scalar three-same non-saturating shifts SSHL/USHL/SRSHL/URSHL**
//!     (opcode `01000`/`01010`) — these scalar shifts are *doubleword only*; the
//!     `b`/`h`/`s` forms (`size != 11`) are reserved → UNDEFINED. Their
//!     saturating siblings SQSHL/UQSHL/SQRSHL/UQRSHL (`01001`/`01011`) keep
//!     accepting every element width. E.g. `5E2F577F` (fARM64 `srshl b31,…`),
//!     `5E2B4666` (`sshl b6,…`), `7E2F577F` (`urshl`), `7E2B4666` (`ushl`) →
//!     UNDEFINED; only the `d` form (`5EEF477F` etc.) decodes.
//!
//!  2. **NEON SQDMULH/SQRDMULH byte forms `.8b`/`.16b`** (opcode `10110`,
//!     `size==00`) — SQDMULH/SQRDMULH are defined only for `.4h`/`.8h`
//!     (`size==01`) and `.2s`/`.4s` (`size==10`); both the byte forms
//!     (`size==00`) and the doubleword forms (`size==11`) are reserved →
//!     UNDEFINED. E.g. `0E24B4A8` (fARM64 `sqdmulh v8.8b,…`), `2E24B4A8`
//!     (`sqrdmulh v8.8b,…`) → UNDEFINED.

#![cfg(feature = "std")]

use fARM64::decode::decode;
use fARM64::format::{format_to_string, FmtFormatter};
use fARM64::{encode, FeatureSet};

fn text(word: u32) -> String {
    let insn = decode(word, 0, FeatureSet::ALL);
    format_to_string(&FmtFormatter::new(), &insn)
}

fn norm(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Decode, assert disasm == `expected`, and prove a bit-exact encode round-trip.
#[track_caller]
fn check(word: u32, expected: &str) {
    let insn = decode(word, 0, FeatureSet::ALL);
    assert!(!insn.is_invalid(), "{:08X} decoded Invalid (want `{}`)", word, expected);
    assert_eq!(norm(&text(word)), norm(expected), "{:08X} disasm mismatch", word);
    let enc = encode(&insn)
        .unwrap_or_else(|e| panic!("{:08X} ({}) encode error {:?}", word, insn.mnemonic().name(), e));
    assert_eq!(enc, word, "{:08X} round-trip produced {:08X}", word, enc);
}

/// Assert a word is rejected (reserved / UNDEFINED).
#[track_caller]
fn reserved(word: u32) {
    assert!(decode(word, 0, FeatureSet::ALL).is_invalid(), "{word:08X} should be reserved (Invalid)");
}

/// Whether a word decodes to *some* non-Invalid instruction.
fn valid(word: u32) -> bool {
    !decode(word, 0, FeatureSet::ALL).is_invalid()
}

// Scalar three-same builder: `0 1 U 11110 size 1 Rm opcode 1 Rn Rd`.
fn scalar(u: u32, size: u32, opcode: u32) -> u32 {
    (0b01011110u32 << 24)
        | (u << 29)
        | (size << 22)
        | (1 << 21)
        | (15 << 16)
        | (opcode << 11)
        | (1 << 10)
        | (27 << 5)
        | 31
}

// Vector three-same builder: `0 Q U 01110 size 1 Rm opcode 1 Rn Rd`.
fn vector(q: u32, u: u32, size: u32, opcode: u32) -> u32 {
    (0b00001110u32 << 24)
        | (q << 30)
        | (u << 29)
        | (size << 22)
        | (1 << 21)
        | (4 << 16)
        | (opcode << 11)
        | (1 << 10)
        | (5 << 5)
        | 8
}

// ---------------------------------------------------------------------------
// Valid forms — must still decode + round-trip exactly (regression guard).
// ---------------------------------------------------------------------------

#[test]
fn valid_forms_still_decode_and_roundtrip() {
    // (1) Scalar non-saturating shifts: the doubleword (`d`) form is the only
    // allocated scalar size and must keep decoding.
    check(0x5EEF477F, "sshl d31, d27, d15");
    check(0x5EEF577F, "srshl d31, d27, d15");
    check(0x7EEF477F, "ushl d31, d27, d15");
    check(0x7EEF577F, "urshl d31, d27, d15");

    // The *saturating* shifts SQSHL/UQSHL/SQRSHL/UQRSHL accept every width and
    // must be unaffected by the non-saturating guard.
    check(scalar(0, 0b00, 0b01001), "sqshl b31, b27, b15");
    check(scalar(0, 0b01, 0b01001), "sqshl h31, h27, h15");
    check(scalar(0, 0b10, 0b01011), "sqrshl s31, s27, s15");
    check(scalar(1, 0b11, 0b01001), "uqshl d31, d27, d15");
    check(scalar(1, 0b00, 0b01011), "uqrshl b31, b27, b15");
    // Saturating add/sub likewise accept every width.
    check(scalar(0, 0b00, 0b00001), "sqadd b31, b27, b15");
    check(scalar(1, 0b10, 0b00101), "uqsub s31, s27, s15");

    // (2) NEON SQDMULH/SQRDMULH at the allocated `.4h/.8h/.2s/.4s` sizes.
    check(0x0E64B4A8, "sqdmulh v8.4h, v5.4h, v4.4h");
    check(0x0EA4B4A8, "sqdmulh v8.2s, v5.2s, v4.2s");
    check(0x4E64B4A8, "sqdmulh v8.8h, v5.8h, v4.8h");
    check(0x4EA4B4A8, "sqdmulh v8.4s, v5.4s, v4.4s");
    check(0x2E64B4A8, "sqrdmulh v8.4h, v5.4h, v4.4h");
    check(0x6EA4B4A8, "sqrdmulh v8.4s, v5.4s, v4.4s");

    // Scalar SQDMULH/SQRDMULH (`h`/`s` only) are an adjacent allocation and must
    // keep decoding.
    check(scalar(0, 0b01, 0b10110), "sqdmulh h31, h27, h15");
    check(scalar(1, 0b10, 0b10110), "sqrdmulh s31, s27, s15");
}

// ---------------------------------------------------------------------------
// (1) Scalar SSHL/USHL/SRSHL/URSHL are doubleword-only.
// ---------------------------------------------------------------------------

#[test]
fn scalar_nonsat_shifts_d_only_reserved() {
    // The exact over-decode examples from the task (all non-`d` sizes).
    reserved(0x5E2F577F); // fARM64 `srshl b31,…`
    reserved(0x5E2B4666); // fARM64 `sshl b6,…`
    reserved(0x7E2F577F); // fARM64 `urshl b31,…`
    reserved(0x7E2B4666); // fARM64 `ushl b6,…`

    // Exhaustive: SSHL(0,01000)/SRSHL(0,01010)/USHL(1,01000)/URSHL(1,01010) are
    // valid only at `size==11` (doubleword); `size` in {00,01,10} is reserved.
    for u in [0u32, 1] {
        for opcode in [0b01000u32, 0b01010] {
            for size in 0u32..4 {
                let w = scalar(u, size, opcode);
                if size == 0b11 {
                    assert!(valid(w), "{w:08X} scalar `d` shift should decode");
                } else {
                    reserved(w);
                }
            }
        }
    }
}

#[test]
fn scalar_sat_shifts_all_sizes_still_valid() {
    // SQSHL(01001)/SQRSHL(01011) and the unsigned siblings accept every width —
    // the non-saturating guard must NOT touch them.
    for u in [0u32, 1] {
        for opcode in [0b01001u32, 0b01011] {
            for size in 0u32..4 {
                let w = scalar(u, size, opcode);
                assert!(valid(w), "{w:08X} saturating scalar shift must stay valid");
            }
        }
    }
    // SQADD/UQADD/SQSUB/UQSUB also accept every width.
    for u in [0u32, 1] {
        for opcode in [0b00001u32, 0b00101] {
            for size in 0u32..4 {
                let w = scalar(u, size, opcode);
                assert!(valid(w), "{w:08X} saturating add/sub must stay valid");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// (2) NEON SQDMULH/SQRDMULH byte forms `.8b`/`.16b` are reserved.
// ---------------------------------------------------------------------------

#[test]
fn neon_sqdmulh_byte_forms_reserved() {
    // The exact over-decode examples from the task.
    reserved(0x0E24B4A8); // fARM64 `sqdmulh v8.8b,…`
    reserved(0x2E24B4A8); // fARM64 `sqrdmulh v8.8b,…`
    reserved(0x4E24B4A8); // `.16b`
    reserved(0x6E24B4A8); // `.16b`

    // Exhaustive: SQDMULH(U=0)/SQRDMULH(U=1), opcode 10110, is valid only at
    // `size` in {01,10}; the byte (`size==00`) and doubleword (`size==11`) forms
    // are reserved, for every Q.
    for q in [0u32, 1] {
        for u in [0u32, 1] {
            for size in 0u32..4 {
                let w = vector(q, u, size, 0b10110);
                if size == 0b01 || size == 0b10 {
                    assert!(valid(w), "{w:08X} SQ[R]DMULH .4h/.8h/.2s/.4s should decode");
                } else {
                    reserved(w);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// These are base FP/SIMD encodings (no extension feature involved); the guards
// reject the reserved words under both the base feature set and the full one.
// ---------------------------------------------------------------------------

#[test]
fn reserved_regardless_of_features() {
    for &w in &[0x5E2F577F, 0x5E2B4666, 0x7E2F577F, 0x0E24B4A8, 0x2E24B4A8] {
        assert!(
            decode(w, 0, FeatureSet::NONE).is_invalid(),
            "{w:08X} reserved under base features"
        );
        assert!(
            decode(w, 0, FeatureSet::ALL).is_invalid(),
            "{w:08X} reserved under ALL features"
        );
    }
}
