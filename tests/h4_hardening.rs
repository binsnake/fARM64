//! H4 over-decode hardening + FEAT_RPRFM: reserved-Invalid checks, surviving
//! valid forms, and the new RPRFM round-trips / feature gating.
//!
//! Four areas are hardened against over-decoding (each reserved condition was
//! confirmed `<unknown>` in LLVM `clang`/`llvm-objdump --mattr=+all`, with no
//! LLVM-valid word newly rejected):
//!
//! * **Add/subtract (extended register)** — `opt` (`word<23:22>`) MUST be `00`.
//!   `ADD/ADDS/SUB/SUBS` (32/64-bit) with `opt != 00` are UNDEFINED.
//! * **NEON load/store multiple structures** — `word<21>` is a fixed-zero bit,
//!   and the `.1d` arrangement (`size==11`, `Q==0`) is reserved for the genuine
//!   multi-structure forms (`LD2/LD3/LD4` and `ST2/ST3/ST4`).
//! * **NEON load/store single structure** — the no-offset form requires the
//!   `Rm` field (`word<20:16>`) to be zero; the `S`/`D` element forms require
//!   `size<1>==0` (32/64-bit single-element, reserved otherwise); the replicate
//!   forms (`LD1R`..`LD4R`) require `S` (`word<12>`) == 0.
//! * **NEON FP16-widening MLAL by element / vector** — `FMLAL/FMLSL/FMLAL2/
//!   FMLSL2` require `size<0>==0` (vector three-same) and `size==10`
//!   (by-element); other `size` values are different instructions or reserved.
//!
//! Plus the new **FEAT_RPRFM** `RPRFM <rprfop>, <Xm>, [<Xn|SP>]` range-prefetch,
//! which carves the `Rt<4:3>==11` sub-slot out of the `PRFM (register offset)`
//! encoding (fixing the previous `prfm` over-decode).

#![cfg(feature = "std")]

use fARM64::decode::decode;
use fARM64::format::{format_to_string, FmtFormatter};
use fARM64::{encode, Feature, FeatureSet};

/// Render `word` (decoded at ip 0 with all features) to its disassembly text.
fn text(word: u32) -> String {
    let insn = decode(word, 0, FeatureSet::ALL);
    format_to_string(&FmtFormatter::new(), &insn)
}

/// Collapse whitespace runs so mnemonic/operand padding is irrelevant.
fn norm(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Decode `word`, assert the disassembly matches `expected`, then prove a
/// bit-exact encoder round-trip.
fn check(word: u32, expected: &str) {
    let insn = decode(word, 0, FeatureSet::ALL);
    assert!(!insn.is_invalid(), "{:08X} decoded Invalid (want `{}`)", word, expected);
    assert_eq!(norm(&text(word)), norm(expected), "{:08X} disasm mismatch", word);
    let enc = encode(&insn)
        .unwrap_or_else(|e| panic!("{:08X} ({}) encode error {:?}", word, insn.mnemonic().name(), e));
    assert_eq!(enc, word, "{:08X} round-trip produced {:08X}", word, enc);
}

/// Assert `word` decodes Invalid (a reserved/UNDEFINED encoding).
fn invalid(word: u32) {
    let insn = decode(word, 0, FeatureSet::ALL);
    assert!(insn.is_invalid(), "{:08X} should be Invalid but decoded `{}`", word, text(word));
}

/// `FeatureSet::ALL` minus one feature bit (both words).
fn without(f: Feature) -> FeatureSet {
    let bit = f as u32;
    let mask = !(1u64 << bit);
    FeatureSet {
        features0: mask,
        features1: mask,
    }
}

// ---------------------------------------------------------------------------
// PRIORITY 1: add/subtract (extended register) `opt` (word<23:22>) must be 00.
// ---------------------------------------------------------------------------

#[test]
fn addsub_extended_opt_zero_valid_forms() {
    // opt==00 is valid: the canonical add/sub extended-register family.
    // `0B2EF362` (opt=00) is `add w2, w27, w14, sxtx #4` per LLVM.
    check(0x0B2EF362, "add w2, w27, w14, sxtx #0x4");
    // 64-bit, the uxtb..sxtx options at imm3 0..4 and the SP/cmp aliases.
    check(0x8B3B6186, "add x6, x12, x27, uxtx");
    check(0x8B304F9F, "add sp, x28, w16, uxtw #0x3");
    check(0xEB2A61FF, "cmp x15, x10, uxtx");
    check(0x8B206FFF, "add sp, sp, x0, lsl #0x3");
    check(0x4B2DEB78, "sub w24, w27, w13, sxtx #0x2");
}

#[test]
fn addsub_extended_opt_nonzero_invalid() {
    // The task example: opt=11 (`0B6EF362`) is UNDEFINED (was over-decoded as
    // `add w2, w27, w14, sxtx #4`).
    invalid(0x0B6EF362);
    // Sweep opt in {01,10,11} over a representative add/sub extended word; all
    // are reserved. Base word = `0B2EF362` (opt=00, valid). opt sits at bits
    // 23:22.
    let base = 0x0B2EF362u32 & !(0b11 << 22);
    for opt in 1u32..=3 {
        invalid(base | (opt << 22));
    }
    // Across sf/op/S too: take a 64-bit SUBS (`cmp`) and flip opt.
    let cmp = 0xEB2A61FFu32 & !(0b11 << 22);
    for opt in 1u32..=3 {
        invalid(cmp | (opt << 22));
    }
}

// ---------------------------------------------------------------------------
// PRIORITY 2a: load/store MULTIPLE structures.
// ---------------------------------------------------------------------------

#[test]
fn ldst_multiple_valid_forms_survive() {
    // `.1d` LD1/ST1 (size==11, Q==0) IS allowed for the LD1/ST1 forms.
    check(0x0C407CA7, "ld1 {v7.1d}, [x5]");
    check(0x4C407CA7, "ld1 {v7.2d}, [x5]");
    check(0x0CDF7CA1, "ld1 {v1.1d}, [x5], #0x8");
    // LD4 .2d (Q==1) is fine — the multi-structure `.2d` (only Q==1) form.
    check(0x4C400CA7, "ld4 {v7.2d, v8.2d, v9.2d, v10.2d}, [x5]");
}

#[test]
fn ldst_multiple_bit21_reserved() {
    // word<21> is fixed-zero. The task example `0CE27D59` (bit21==1) is
    // UNDEFINED (was `ld1 {v25.1d}, [x10], x2`).
    invalid(0x0CE27D59);
    // Flip bit21 on a known-valid LD1 .1d post-reg word: must go Invalid.
    let valid = 0x0CC27CA1; // ld1 {v1.1d}, [x5], x2
    check(valid, "ld1 {v1.1d}, [x5], x2");
    invalid(valid | (1 << 21));
}

#[test]
fn ldst_multiple_1d_multistruct_reserved() {
    // `.1d` (size==11, Q==0) is reserved for LD2/LD3/LD4 and ST2/ST3/ST4.
    // ST4 .1d, ST3 .1d, ST2 .1d (no-offset, opcode 0000/0100/1000):
    invalid(0x0C000CA7); // st4 {...1d}
    invalid(0x0C004CA7); // st3 {...1d}
    invalid(0x0C008CA7); // st2 {...1d}
    // LD2/LD3/LD4 .1d likewise.
    invalid(0x0C400CA7); // ld4 {...1d}
    invalid(0x0C404CA7); // ld3 {...1d}
    invalid(0x0C408CA7); // ld2 {...1d}
}

// ---------------------------------------------------------------------------
// PRIORITY 2b: load/store SINGLE structure (+ replicate).
// ---------------------------------------------------------------------------

#[test]
fn ldst_single_valid_forms_survive() {
    check(0x0D0080A7, "st1 {v7.s}[0], [x5]");
    check(0x0D0090A7, "st1 {v7.s}[1], [x5]");
    check(0x0D0084A7, "st1 {v7.d}[0], [x5]");
    check(0x4F8E00D4, "fmlal v20.4s, v6.4h, v14.h[0]"); // (unrelated nearby valid)
}

#[test]
fn ldst_single_no_offset_rm_must_be_zero() {
    // No-offset single form requires Rm (word<20:16>) == 0. Rm=2 and Rm=31
    // variants of a valid `.b`[0] single ST1 are reserved.
    check(0x0D0000A7, "st1 {v7.b}[0], [x5]");
    invalid(0x0D0200A7); // Rm=2
    invalid(0x0D1F00A7); // Rm=31
}

#[test]
fn ldst_single_s_and_d_size_high_bit_reserved() {
    // `.s` element: size==00 valid; size==10 reserved (size<1> must be 0).
    invalid(0x0D0088A7); // st1 {v7.s}, size=10
    // `.d` element: size==01 valid; size==11 reserved.
    invalid(0x0D008CA7); // st1 {v7.d}, size=11
}

#[test]
fn ldst_replicate_s_bit_reserved() {
    // LD*R require S (word<12>) == 0. A valid LD3R and its S==1 sibling.
    check(0x0D40E4A7, "ld3r {v7.4h, v8.4h, v9.4h}, [x5]");
    invalid(0x0D40F4A7); // same but S==1
    // The task example `0DF7DDCE` (LD2R, S==1) is UNDEFINED.
    invalid(0x0DF7DDCE);
}

// ---------------------------------------------------------------------------
// PRIORITY 3: FMLAL/FMLSL/FMLAL2/FMLSL2 size constraints.
// ---------------------------------------------------------------------------

#[test]
fn fmlal_vector_valid_forms_survive() {
    check(0x0E3BEEF2, "fmlal v18.2s, v23.2h, v27.2h");
    check(0x0EBBEEE7, "fmlsl v7.2s, v23.2h, v27.2h");
    check(0x2E3BCEE7, "fmlal2 v7.2s, v23.2h, v27.2h");
    check(0x2EBBCEE7, "fmlsl2 v7.2s, v23.2h, v27.2h");
}

#[test]
fn fmlal_vector_size_low_bit_reserved() {
    // Vector three-same FMLAL/FMLSL family: size<0> (word<22>) must be 0.
    // Task examples: `0EFBEEE7` (fmlsl, size<0>==1) and `2EF3CCB5` (fmlsl2).
    invalid(0x0EFBEEE7);
    invalid(0x2EF3CCB5);
    // Flipping size<0> on each valid form yields a reserved word.
    for valid in [0x0E3BEEF2u32, 0x0EBBEEE7, 0x2E3BCEE7, 0x2EBBCEE7] {
        invalid(valid | (1 << 22));
    }
}

#[test]
fn fmlal_byelement_valid_forms_survive() {
    check(0x0FB348C4, "fmlsl v4.2s, v6.2h, v3.h[7]");
    check(0x2F8380C4, "fmlal2 v4.2s, v6.2h, v3.h[0]");
    check(0x2F83C0C4, "fmlsl2 v4.2s, v6.2h, v3.h[0]");
}

#[test]
fn fmlal_byelement_size_reserved() {
    // By-element FMLAL/FMLSL/FMLAL2/FMLSL2 require size==10. The FMLSL
    // by-element word with size != 10 is a different instruction or reserved:
    // `0F0340C4` (size=00) was over-decoded as `fmlsl ...`; LLVM: UNDEFINED.
    invalid(0x0F0340C4); // size=00
    invalid(0x0F4340C4); // size=01
    invalid(0x0FC340C4); // size=11
}

// ---------------------------------------------------------------------------
// PRIORITY 4: FEAT_RPRFM range prefetch (+ the fixed prfm over-decode).
// ---------------------------------------------------------------------------

#[test]
fn rprfm_examples_decode_and_roundtrip() {
    // The task example: `F8A25BFA` was mis-decoded as a `prfm`; it is RPRFM.
    check(0xF8A25BFA, "rprfm #0xa, x2, [sp]");
    check(0xF8A24BFA, "rprfm #0x2, x2, [sp]");
    // Named rprfop subset (imm6 in {0,1,4,5}).
    check(0xF8A24BF8, "rprfm pldkeep, x2, [sp]");
    check(0xF8A24BF9, "rprfm pstkeep, x2, [sp]");
    check(0xF8A24BFC, "rprfm pldstrm, x2, [sp]");
    check(0xF8A24BFD, "rprfm pststrm, x2, [sp]");
    // Vary the index register and base.
    check(0xF8A04BFA, "rprfm #0x2, x0, [sp]");
}

#[test]
fn rprfm_roundtrip_sweep() {
    // Round-trip every rprfop (imm6 0..63) over the four option<1>==1 values and
    // both S settings — exercising the full imm6 = option<2>:option<0>:S:Rt<2:0>
    // distribution. Rt<4:3> fixed at 11 (the RPRFM sub-slot).
    for option in [0b010u32, 0b011, 0b110, 0b111] {
        for s in 0u32..=1 {
            for rt_lo in 0u32..=7 {
                let rt = 0b11000 | rt_lo;
                let rm = 9;
                let rn = 13;
                let word = (0b11111000u32 << 24)
                    | (1 << 23)
                    | (1 << 21)
                    | (rm << 16)
                    | (option << 13)
                    | (s << 12)
                    | (1 << 11)
                    | (rn << 5)
                    | rt;
                let insn = decode(word, 0, FeatureSet::ALL);
                assert!(!insn.is_invalid(), "{:08X} RPRFM decoded Invalid", word);
                let enc = encode(&insn)
                    .unwrap_or_else(|e| panic!("{:08X} encode error {:?}", word, e));
                assert_eq!(enc, word, "{:08X} round-trip produced {:08X}", word, enc);
            }
        }
    }
}

#[test]
fn rprfm_feature_gated() {
    // Without FEAT_RPRFM, the word does not decode as RPRFM (the slot reverts to
    // Invalid: ordinary PRFM does not allocate the Rt<4:3>==11 prefetch ops).
    let insn = decode(0xF8A25BFA, 0, without(Feature::Rprfm));
    assert!(insn.is_invalid(), "RPRFM should be gated off without FEAT_RPRFM");
    // With the feature it decodes.
    let insn = decode(0xF8A25BFA, 0, FeatureSet::ALL);
    assert!(!insn.is_invalid());
}

#[test]
fn prfm_register_with_low_rt_still_prfm() {
    // The `Rt<4:3> != 11` half of the slot stays ordinary PRFM (register offset).
    check(0xF8A24BE0, "prfm pldl1keep, [sp, w2, uxtw]");
}
