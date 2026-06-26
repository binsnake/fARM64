//! SME ZA-array load/store over-decode hardening (`wt/i1`).
//!
//! Covers the contiguous ZA tile-slice loads/stores `LD1{B,H,W,D,Q}` /
//! `ST1{B,H,W,D,Q}` ZA and the whole-array `LDR`/`STR` ZA, all in the SME `111`
//! quadrant (`word<31:25> == 1110000`, i.e. the `0xE0`/`0xE1` high bytes).
//!
//! fARM64 previously over-decoded large reserved sub-regions that LLVM
//! (`clang`/`llvm-objdump --mattr=+all`) leaves UNDEFINED. The exact reserved
//! conditions, found by oracle field sweeps, are:
//!
//! * **`LD1*`/`ST1*` ZA (all of B/H/W/D and Q):** `word<4>` is a fixed-zero bit
//!   sitting above the tile+slice select group `word<3:0>`; `word<4> == 1` is
//!   UNDEFINED for every element size. Examples: `E03779B1` (st1b), `E0623A9B`
//!   (st1h), `E0A66D13` (st1w), `E0F34739` (st1d).
//! * **`LD1Q`/`ST1Q` ZA (`word<24> == 1`):** allocated only when the size field
//!   `word<23:22> == 11`; sizes `01`/`10` are reserved (`00` is the `LDR`/`STR`
//!   ZA region). Examples: `E14CDA26` (ld1q), `E16EF362` (st1q).
//! * **`LDR`/`STR` ZA (`word<24> == 1`, `word<23:22> == 00`):** the fields
//!   `word<20:16>`, `word<15>`, `word<12:10>` and `word<4>` are fixed zero; any
//!   nonzero value is UNDEFINED. Example: `E13779B1` (str za).
//!
//! Each test confirms (a) the surviving LLVM-valid forms still decode and
//! round-trip, and (b) the newly-rejected reserved encodings are `Invalid`.

#![cfg(all(feature = "std", feature = "sme"))]

use fARM64::decode::decode;
use fARM64::{encode, Feature, FeatureSet};

/// `true` when `word` is a `LD1*`/`ST1*` ZA contiguous tile-slice form whose
/// tile number (the high bits of the slice-select group `word<3:0>`) is nonzero.
///
/// NOTE: the current decoder renders these forms with a hard-wired tile `0` and
/// drops the encoded tile number (a *pre-existing* operand-rendering limitation,
/// present identically on `main`); for such words decode→encode is lossy. This
/// over-decode-hardening change does not touch that, so the bit-exact round-trip
/// below is restricted to the tile-0 forms (where it is lossless). Non-tile-0
/// forms are still asserted to decode (`assert_decodes`).
fn ld1_st1_tile_nonzero(word: u32) -> bool {
    let in_region = (word >> 25) == 0b1110000;
    if !in_region {
        return false;
    }
    let b24 = (word >> 24) & 1;
    let sz = (word >> 22) & 0b11;
    // LDR/STR ZA (b24==1, sz==00) has no tile field of this shape.
    if b24 == 1 && sz == 0 {
        return false;
    }
    // tile-number width = 4 - slice-index width; slice-index width per size:
    // B=4, H=3, W=2, D=1, Q=0.
    let imm_bits = if b24 == 1 { 0 } else { [4u32, 3, 2, 1][sz as usize] };
    let tile = (word & 0xf) >> imm_bits;
    tile != 0
}

/// Decode `word` and assert it is not `Invalid` (survives the hardening).
fn assert_decodes(word: u32) {
    assert!(!decode(word, 0, FeatureSet::ALL).is_invalid(), "{:08X} should decode", word);
}

/// Decode `word`, re-encode, re-decode; require identical mnemonic + operands.
fn assert_roundtrip(word: u32) {
    let insn = decode(word, 0, FeatureSet::ALL);
    assert!(!insn.is_invalid(), "{:08X} decoded Invalid", word);
    let enc = encode(&insn)
        .unwrap_or_else(|e| panic!("{:08X} ({}) encode error {:?}", word, insn.mnemonic().name(), e));
    assert_eq!(enc, word, "{:08X} ({}) re-encoded to {:08X}", word, insn.mnemonic().name(), enc);
    let insn2 = decode(enc, 0, FeatureSet::ALL);
    assert_eq!(insn.mnemonic(), insn2.mnemonic(), "{:08X} mnemonic drift", word);
    assert_eq!(insn.op_count(), insn2.op_count(), "{:08X} operand-count drift", word);
    for i in 0..insn.op_count() {
        assert_eq!(
            format!("{:?}", insn.op(i)),
            format!("{:?}", insn2.op(i)),
            "{:08X} ({}) operand {} differs",
            word,
            insn.mnemonic().name(),
            i
        );
    }
}

/// Round-trip when lossless (tile 0 / `LDR`/`STR` ZA); otherwise just survival.
fn assert_survives(word: u32) {
    if ld1_st1_tile_nonzero(word) {
        assert_decodes(word);
    } else {
        assert_roundtrip(word);
    }
}

fn assert_invalid(word: u32, why: &str) {
    assert!(decode(word, 0, FeatureSet::ALL).is_invalid(), "{:08X} should be Invalid ({})", word, why);
}

/// LLVM-valid canonical examples (oracle `--mattr=+all`) must keep decoding.
#[test]
fn surviving_valid_examples() {
    // (word, mnemonic, operand_count).
    let cases: &[(u32, &str, usize)] = &[
        // LD1*/ST1* ZA tile-slice (tile 0, predicated, index scaled by elem size).
        (0xE011E5A3, "ld1b", 3),
        (0xE059A9C2, "ld1h", 3),
        (0xE0D76421, "ld1d", 3),
        (0xE1DE4A4D, "ld1q", 3),
        (0xE024806B, "st1b", 3),
        (0xE1F50B6D, "st1q", 3),
        (0xE0B24FE7, "st1w", 3),
        // LDR/STR ZA whole-array vector.
        (0xE100004D, "ldr", 2),
        (0xE1200106, "str", 2),
        (0xE10062A0, "ldr", 2),
        (0xE10043EB, "ldr", 2),
    ];
    for &(w, m, n) in cases {
        let insn = decode(w, 0, FeatureSet::ALL);
        assert_eq!(insn.mnemonic().name(), m, "{:08X} mnemonic", w);
        assert_eq!(insn.op_count(), n, "{:08X} operand count", w);
        assert_survives(w);
    }
}

/// The seven named over-decode words from the task brief — all `<unknown>` in
/// LLVM, now `Invalid` in fARM64.
#[test]
fn named_overdecodes_are_invalid() {
    assert_invalid(0xE14CDA26, "ld1q size!=11"); // sz=01, word<24>=1
    assert_invalid(0xE16EF362, "st1q size!=11"); // sz=01, word<24>=1
    assert_invalid(0xE03779B1, "st1b word<4>=1");
    assert_invalid(0xE0F34739, "st1d word<4>=1");
    assert_invalid(0xE0623A9B, "st1h word<4>=1");
    assert_invalid(0xE0A66D13, "st1w word<4>=1");
    assert_invalid(0xE13779B1, "str za reserved fields set");
}

/// `word<4> == 1` is reserved for every `LD1*`/`ST1*` ZA element size. Take a
/// known-valid base per size, set bit 4, and confirm it becomes `Invalid`.
#[test]
fn ld1_st1_bit4_reserved() {
    // Known-valid base words (bit 4 == 0), one per size and per load/store.
    let valids = [
        0xE011E5A3u32, // ld1b
        0xE024806B,    // st1b
        0xE059A9C2,    // ld1h
        0xE0623A8Bu32, // st1h (E0623A9B with bit4 cleared)
        0xE0B24FE7,    // st1w
        0xE0A66D03u32, // st1w (E0A66D13 with bit4 cleared)
        0xE0D76421,    // ld1d
        0xE0F34729u32, // st1d (E0F34739 with bit4 cleared)
        0xE1DE4A4D,    // ld1q
        0xE1F50B6D,    // st1q
    ];
    for &v in &valids {
        // sanity: the base (bit4==0) decodes.
        assert!(!decode(v, 0, FeatureSet::ALL).is_invalid(), "{:08X} base should decode", v);
        // setting word<4> makes it UNDEFINED.
        assert_invalid(v | (1 << 4), "ld1/st1 word<4>=1");
    }
}

/// `LD1Q`/`ST1Q` (`word<24> == 1`) requires `word<23:22> == 11`; sizes `01`/`10`
/// are reserved. (`00` is routed to `LDR`/`STR` ZA, which has its own rules.)
#[test]
fn ld1q_st1q_size_reserved() {
    // Valid ld1q (sz==11) base.
    let base = 0xE1DE4A4Du32;
    assert!(!decode(base, 0, FeatureSet::ALL).is_invalid(), "ld1q sz=11 should decode");
    // sz=01 and sz=10 with word<24>==1 are UNDEFINED.
    let clear_sz = base & !(0b11 << 22);
    assert_invalid(clear_sz | (0b01 << 22), "ld1q-region sz=01");
    assert_invalid(clear_sz | (0b10 << 22), "ld1q-region sz=10");
    // sz=11 still decodes (the valid Q form).
    assert!(!decode(clear_sz | (0b11 << 22), 0, FeatureSet::ALL).is_invalid(), "sz=11 should decode");
}

/// `LDR`/`STR` ZA: `word<20:16>`, `word<15>`, `word<12:10>`, `word<4>` are fixed
/// zero. Each, set individually on a valid base, must produce `Invalid`.
#[test]
fn ldr_str_za_fixed_fields() {
    let base = 0xE100004Du32; // ldr za[w12, #0xd], [x2, #0xd, mul vl]
    assert!(!decode(base, 0, FeatureSet::ALL).is_invalid(), "ldr-za base should decode");
    // Each fixed-zero bit, set on its own, is UNDEFINED.
    for bitpos in [16u32, 17, 18, 19, 20, 15, 10, 11, 12, 4] {
        assert_invalid(base | (1 << bitpos), "ldr/str za fixed-zero bit set");
    }
    // The STR variant (op = word<21>) likewise.
    let str_base = base | (1 << 21);
    assert!(!decode(str_base, 0, FeatureSet::ALL).is_invalid(), "str-za base should decode");
    for bitpos in [16u32, 15, 12, 4] {
        assert_invalid(str_base | (1 << bitpos), "str za fixed-zero bit set");
    }
}

/// Exhaustive structural sweep: every surviving form still round-trips, and no
/// `word<4> == 1` encoding in the `LD1*`/`ST1*` region decodes.
#[test]
fn ld1_st1_sweep() {
    let regfills = [(0u32, 0u32, 0u32), (31, 7, 31), (23, 6, 13), (4, 0, 3)];
    let mut decoded = 0usize;
    for b24 in 0..2u32 {
        for sz in 0..4u32 {
            for op in 0..2u32 {
                for v in 0..2u32 {
                    for rs in 0..4u32 {
                        for low5 in 0..32u32 {
                            for &(rm, pg, rn) in &regfills {
                                let w = (0b1110000 << 25)
                                    | (b24 << 24)
                                    | (sz << 22)
                                    | (op << 21)
                                    | (rm << 16)
                                    | (v << 15)
                                    | (rs << 13)
                                    | (pg << 10)
                                    | (rn << 5)
                                    | low5;
                                let insn = decode(w, 0, FeatureSet::ALL);
                                if insn.is_invalid() {
                                    continue;
                                }
                                decoded += 1;
                                // No surviving form may have word<4> set...
                                // (LDR/STR ZA also clears bit 4; LD1/ST1 too.)
                                assert_eq!(w & (1 << 4), 0, "{:08X} decoded with word<4>=1", w);
                                assert_survives(w);
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(decoded >= 500, "expected many ZA ld/st forms to survive, got {}", decoded);
}

/// The whole region must be gated on FEAT_SME.
#[test]
fn gated_by_feature_sme() {
    let no_sme = without_feature(FeatureSet::ALL, Feature::Sme);
    let words = [0xE011E5A3u32, 0xE1DE4A4D, 0xE100004D, 0xE1200106];
    for w in words {
        assert!(decode(w, 0, no_sme).is_invalid(), "{:08X} should require FEAT_SME", w);
        assert!(!decode(w, 0, FeatureSet::ALL).is_invalid(), "{:08X} should decode with FEAT_SME", w);
    }
}

/// `FeatureSet::ALL` minus one feature bit (in both words).
fn without_feature(fs: FeatureSet, f: Feature) -> FeatureSet {
    let bit = f as u32;
    FeatureSet {
        features0: fs.features0 & !(1u64 << bit),
        features1: fs.features1 & !(1u64 << bit),
    }
}
