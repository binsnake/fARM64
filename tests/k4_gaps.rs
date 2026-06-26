//! K4 batch — additive instruction-gap coverage: decode + round-trip + feature
//! gating. All canonical example words are LLVM (`clang`/`llvm-objdump
//! --mattr=+all`) oracle encodings.
//!
//! Covers:
//!   * SVE2.2 zeroing (`/z`) narrow/long FP converts FCVTNT/FCVTLT/FCVTXNT/
//!     BFCVTNT and the zeroing URECPE/URSQRTE reciprocal estimates.
//!   * FEAT_SVE_AES2 multi-vector quadword AES (AESE/AESD/AESEMC/AESDIMC) and
//!     the polynomial multiply-long PMULL/PMLAL, plus the SVE2.1 multi-vector
//!     narrowing converts SQCVTN/UQCVTN/SQCVTUN.
//!   * FEAT_FPRCVT scalar FP<->int converts with differing register widths.
//!   * FEAT_MOPS memory-set-with-tag option variants (SETGO*) and the TCHANGE
//!     translation-table change instructions (register/immediate/`nb`).

#![cfg(feature = "std")]

use fARM64::decode::decode;
use fARM64::format::{BufSink, FmtFormatter, Formatter};
use fARM64::{encode, Feature, FeatureSet};

/// Render `insn` to its textual disassembly.
fn text(word: u32) -> String {
    let insn = decode(word, 0x1000, FeatureSet::ALL);
    let mut buf = [0u8; 160];
    let mut sink = BufSink::new(&mut buf);
    FmtFormatter::new().format(&insn, &mut sink);
    sink.as_str().to_string()
}

/// Decode `word`, re-encode, require an identical word; then re-decode and
/// require mnemonic + operand stability.
fn assert_roundtrip(word: u32) {
    let insn = decode(word, 0x1000, FeatureSet::ALL);
    assert!(!insn.is_invalid(), "{word:08X} decoded Invalid");
    let enc = encode(&insn)
        .unwrap_or_else(|e| panic!("{word:08X} ({}) encode error {e:?}", insn.mnemonic().name()));
    assert_eq!(enc, word, "{word:08X} ({}) re-encoded to {enc:08X}", insn.mnemonic().name());
    let insn2 = decode(enc, 0x1000, FeatureSet::ALL);
    assert_eq!(insn.mnemonic(), insn2.mnemonic(), "{word:08X} mnemonic drift");
    assert_eq!(insn.op_count(), insn2.op_count(), "{word:08X} operand-count drift");
}

/// Each `(word, expected disassembly)` pair is the LLVM oracle rendering, modulo
/// the intentional fARM64 hex-radix differences documented in AGENT_GUIDE.md.
const CASES: &[(u32, &str)] = &[
    // --- K4-1: SVE2.2 /z narrow/long FP converts + reciprocal estimates ---
    (0x64C2AF9B, "fcvtnt  z27.s, p3/z, z28.d"),
    (0x64C3BA93, "fcvtlt  z19.d, p6/z, z20.s"),
    (0x6402B633, "fcvtxnt z19.s, p5/z, z17.d"),
    (0x6482A903, "bfcvtnt z3.h, p2/z, z8.s"),
    (0x4482A903, "urecpe  z3.s, p2/z, z8.s"),
    (0x4483B3FB, "ursqrte z27.s, p4/z, z31.s"),
    // --- K4-2: FEAT_SVE_AES2 multi-vector quadword + narrowing converts ---
    (0x4533EE14, "aesdimc { z20.b, z21.b }, { z20.b, z21.b }, z16.q[2]"),
    (0x453AEAEE, "aese    { z14.b, z15.b }, { z14.b, z15.b }, z23.q[3]"),
    (0x4534F90C, "pmull   { z12.q, z13.q }, z8.d, z20.d"),
    (0x4521FEDC, "pmlal   { z28.q, z29.q }, z22.d, z1.d"),
    (0x4531530F, "sqcvtun z15.h, { z24.s, z25.s }"),
    // quad-vector AES + the e/d/emc/dimc + sqcvtn/uqcvtn neighbours.
    (0x4526EA14, "aese    { z20.b - z23.b }, { z20.b - z23.b }, z16.q[0]"),
    (0x4527EE14, "aesdimc { z20.b - z23.b }, { z20.b - z23.b }, z16.q[0]"),
    (0x4531430F, "sqcvtn  z15.h, { z24.s, z25.s }"),
    (0x45314B0F, "uqcvtn  z15.h, { z24.s, z25.s }"),
    // --- K4-3: FEAT_FPRCVT scalar FP<->int convert, differing widths ---
    (0x1EF40243, "fcvtms  s3, h18"),
    (0x1E7A0155, "fcvtas  s21, d10"),
    (0x1E75000B, "fcvtmu  s11, d0"),
    (0x1E7C0143, "scvtf   d3, s10"),  // int->fp, differing widths
    (0x9E2A0143, "fcvtns  d3, s10"),  // sf=1 dst-width form
    // --- K4-4: FEAT_MOPS SETGO* option + TCHANGE ---
    (0x1DDF0211, "setgop  [x17]!, x16!"),
    (0x1DDF4211, "setgom  [x17]!, x16!"),
    (0x1DDFB211, "setgoetn [x17]!, x16!"),
    (0xD58403E9, "tchangeb x9, xzr"),
    (0xD5840209, "tchangeb x9, x16"),
    (0xD58603E9, "tchangeb x9, xzr, nb"),
    (0xD5960209, "tchangeb x9, #0x10, nb"),
    (0xD58203E9, "tchangef x9, xzr, nb"),
];

#[test]
fn examples_decode_and_render() {
    for &(w, expected) in CASES {
        assert_eq!(text(w), expected, "{w:08X} rendering");
    }
}

#[test]
fn examples_round_trip() {
    for &(w, _) in CASES {
        assert_roundtrip(w);
    }
}

// --- Feature gating -------------------------------------------------------

#[test]
fn sve2p2_zeroing_converts_gated() {
    // The /z narrow/long converts and the URECPE/URSQRTE /z need FEAT_SVE2p2.
    let no = FeatureSet::BASE.with(Feature::Sve).with(Feature::Bf16).with(Feature::Sve2p1);
    let yes = no.with(Feature::Sve2p2);
    for w in [0x64C2AF9Bu32, 0x64C3BA93, 0x6402B633, 0x6482A903, 0x4482A903, 0x4483B3FB] {
        assert!(decode(w, 0, no).is_invalid(), "{w:08X} must need Sve2p2");
        assert!(!decode(w, 0, yes).is_invalid(), "{w:08X} should decode with Sve2p2");
    }
}

#[test]
fn sve_aes2_gated() {
    let base = FeatureSet::BASE.with(Feature::Sve).with(Feature::Sve2p1);
    let yes = base.with(Feature::SveAes2);
    for w in [0x4533EE14u32, 0x453AEAEE, 0x4534F90C, 0x4521FEDC, 0x4526EA14] {
        assert!(decode(w, 0, base).is_invalid(), "{w:08X} must need SveAes2");
        assert!(!decode(w, 0, yes).is_invalid(), "{w:08X} should decode with SveAes2");
    }
    // The multi-vector narrowing converts are gated on Sve2p1, not AES2.
    for w in [0x4531430Fu32, 0x45314B0F, 0x4531530F] {
        assert!(decode(w, 0, FeatureSet::BASE.with(Feature::Sve)).is_invalid(), "{w:08X} needs Sve2p1");
        assert!(!decode(w, 0, base).is_invalid(), "{w:08X} should decode with Sve2p1");
    }
}

#[test]
fn fprcvt_gated() {
    let no = FeatureSet::BASE.with(Feature::Fp16);
    let yes = no.with(Feature::Fprcvt);
    for w in [0x1EF40243u32, 0x1E7A0155, 0x1E75000B, 0x1E7C0143] {
        assert!(decode(w, 0, no).is_invalid(), "{w:08X} must need Fprcvt");
        assert!(!decode(w, 0, yes).is_invalid(), "{w:08X} should decode with Fprcvt");
    }
    // FP16-source form additionally needs FEAT_FP16.
    assert!(decode(0x1EF40243, 0, FeatureSet::BASE.with(Feature::Fprcvt)).is_invalid());
}

#[test]
fn mops_and_tchange_gated() {
    // SETGO* needs FEAT_MOPS.
    for w in [0x1DDF0211u32, 0x1DDF4211, 0x1DDFB211] {
        assert!(decode(w, 0, FeatureSet::BASE).is_invalid(), "{w:08X} needs Mops");
        assert!(!decode(w, 0, FeatureSet::BASE.with(Feature::Mops)).is_invalid());
    }
    // TCHANGE needs FEAT_Tchange.
    for w in [0xD58403E9u32, 0xD5840209, 0xD58603E9, 0xD5960209] {
        assert!(decode(w, 0, FeatureSet::BASE).is_invalid(), "{w:08X} needs Tchange");
        assert!(!decode(w, 0, FeatureSet::BASE.with(Feature::Tchange)).is_invalid());
    }
}

// --- Reserved/over-decode guards ------------------------------------------

#[test]
fn reserved_slots_stay_invalid() {
    // Words that LLVM rejects (UNDEFINED) in or adjacent to the touched regions.
    let reserved: &[u32] = &[
        // SVE2.2 /z converts: `<20:19>` must be 00 (this has 10).
        0x6412A903,
        // AES2 multi-vector: `<17>` must be 1 for the AES group (this has 0).
        0x4520EA14,
        // AES2 pair base must be even (odd dst -> UNDEFINED).
        0x4522EA15,
        // FPRCVT: same-width pair (S->S) is UNALLOCATED in this slot.
        0x1E2A0143,
        // SETGO*: the value field `<20:16>` must be xzr (this has x0).
        0x1DC04211,
        // SETGO*: op2 12..15 are unallocated.
        0x1DDFC211,
        // TCHANGE: op0 (`<20:19>`) 01/11 are unallocated.
        0xD5A403E9,
        0xD5E403E9,
        // TCHANGE register form: high CRm bits must be 0.
        0xD5840409,
    ];
    for &w in reserved {
        assert!(decode(w, 0, FeatureSet::ALL).is_invalid(), "{w:08X} should be Invalid");
    }
}
