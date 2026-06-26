//! L1 batch: SVE/SVE2 over-decode hardening + a few SVE2.1/2.2 gaps.
//!
//! The "over-decode" half proves fARM64 no longer accepts words that LLVM
//! (`clang`/`llvm-objdump --mattr=+all`) reports as `<unknown>`: each guarded
//! family below pairs the previously-mis-decoded word (now `Invalid`) with the
//! canonical LLVM-valid encoding (still decodes). The reserved condition per
//! family is documented at each `mk`/comment.
//!
//! The "gap" half adds the SVE2.1 multi-vector narrowing shift, SVE2.1
//! CNTP-predicate-as-counter, and SVE2.2 BF2CVT / SCVTFLT / BFSCALE — checking
//! the canonical words decode to the expected mnemonic and round-trip through
//! `encode` back to the identical word.

#![cfg(feature = "std")]

use fARM64::decode::decode;
use fARM64::{encode, Feature, FeatureSet};

/// Decode, re-encode, re-decode; require the word reproduces exactly and the
/// mnemonic/operands are stable.
fn assert_roundtrip(word: u32) {
    let insn = decode(word, 0, FeatureSet::ALL);
    assert!(!insn.is_invalid(), "{:08X} decoded Invalid", word);
    let enc = encode(&insn)
        .unwrap_or_else(|e| panic!("{:08X} ({}) encode error {:?}", word, insn.mnemonic().name(), e));
    assert_eq!(enc, word, "{:08X} ({}) re-encoded to {:08X}", word, insn.mnemonic().name(), enc);
    let insn2 = decode(enc, 0, FeatureSet::ALL);
    assert_eq!(insn.mnemonic(), insn2.mnemonic(), "{:08X} mnemonic drift", word);
    assert_eq!(insn.op_count(), insn2.op_count(), "{:08X} operand-count drift", word);
}

fn is_invalid(word: u32) -> bool {
    decode(word, 0, FeatureSet::ALL).is_invalid()
}

fn mnem(word: u32) -> &'static str {
    decode(word, 0, FeatureSet::ALL).mnemonic().name()
}

// ===========================================================================
// Over-decode guards: (reserved word -> Invalid) paired with (canonical valid).
// ===========================================================================

#[test]
fn narrowing_shift_bit23_reserved() {
    // Single-vector SHRN*/SQ*SHR*/UQ*SHR* and SQXTN*/UQXTN*/SQXTUN* have `<23>=0`.
    // `<23>=1` selects the *multi-vector* family; the single-vector reading of
    // those words is reserved → Invalid.
    for &(bad, good) in &[
        (0x45AF14C1u32, 0x452F14C1u32), // shrnt   z1.b, z6.h, #1
        (0x45E713EE, 0x456713EE),       // shrnb   z14.s, z31.d, #25
        (0x45B430D2, 0x453430D2),       // uqshrnb z18.h, z6.s, #12
        (0x45A84020, 0x45284020),       // sqxtnb  z0.b, z1.h (extract-narrow)
    ] {
        assert!(is_invalid(bad), "{:08X} (b23 set) should be Invalid", bad);
        assert!(!is_invalid(good), "{:08X} canonical should decode", good);
        assert_roundtrip(good);
    }
}

#[test]
fn dup_indexed_bit21_reserved() {
    // DUP indexed / scalar-broadcast `MOV Zd.T, Zn.T[i]` requires `<21>=1`.
    assert!(is_invalid(0x05CB21BB), "05CB21BB (b21=0) should be Invalid");
    assert!(!is_invalid(0x05EB21BB), "05EB21BB canonical should decode"); // mov z27.b, z13.b[53]
    assert!(!is_invalid(0x05212020), "05212020 scalar broadcast should decode"); // mov z0.b, b1
    assert_roundtrip(0x05EB21BB);
    assert_roundtrip(0x05212020);
}

#[test]
fn abdl_bit12_and_histseg_size_reserved() {
    // {S,U}ABDL{B,T} (`<15:13>=001`) fix `<12>=1`; `<12>=0` is reserved.
    for &(bad, good) in &[
        (0x45CE27DC, 0x45CE37DC), // sabdlt z28.d, z30.s, z14.s
        (0x455E2982, 0x455E3982), // uabdlb z2.h, z12.b, z30.b
        (0x45423020 & !(1 << 12), 0x45423020), // sabdlb z0.h, z1.b, z2.b
    ] {
        assert!(is_invalid(bad), "{:08X} ABDL <12>=0 should be Invalid", bad);
        assert!(!is_invalid(good), "{:08X} canonical should decode", good);
        assert_roundtrip(good);
    }
    // HISTSEG is `.b` only — non-zero size is reserved.
    for bad in [0x456BA3EBu32, 0x45ABA3EB, 0x45EBA3EB] {
        assert!(is_invalid(bad), "{:08X} histseg size!=0 should be Invalid", bad);
    }
    assert!(!is_invalid(0x452BA3EB), "452BA3EB histseg should decode");
    assert_roundtrip(0x452BA3EB);
}

#[test]
fn incp_decp_reserved_fields() {
    // INCP/DECP/{SQ,UQ}{INC,DEC}P: `<20>=0`, `<19>=1`, `<9>=0`; vector & plain
    // forms also fix `<10>=0`.
    for &(bad, good) in &[
        (0x25FC8F2A, 0x25EC892A), // incp   x10, p9.d
        (0x25398FA1, 0x25298DA1), // uqincp x1, p13.b
    ] {
        assert!(is_invalid(bad), "{:08X} INCP reserved should be Invalid", bad);
        assert!(!is_invalid(good), "{:08X} canonical should decode", good);
        assert_roundtrip(good);
    }
    // INC/DEC element-count vector: `<12>=0`; non-saturating also `<11>=0`.
    assert!(is_invalid(0x04F7DDCE), "04F7DDCE decd <11>=1 should be Invalid");
    assert!(!is_invalid(0x04F7C5CE), "04F7C5CE decd should decode");
    assert_roundtrip(0x04F7C5CE);
}

#[test]
fn minmax_mul_imm_bit13_reserved() {
    // SMIN/UMIN/SMAX/UMAX/MUL by immediate fix `<13>=0` (no `lsl #8`).
    for &(bad, good) in &[
        (0x256AE249, 0x256AC249), // smin z9.h, z9.h, #0x12
        (0x2570E249, 0x2570C249), // mul  z9.h, z9.h, #0x12
    ] {
        assert!(is_invalid(bad), "{:08X} min/max/mul <13>=1 should be Invalid", bad);
        assert!(!is_invalid(good), "{:08X} canonical should decode", good);
        assert_roundtrip(good);
    }
}

#[test]
fn rdvl_rn_and_udot_h_idx_reserved() {
    // RDVL fixes the `Rn` field to `11111`.
    assert!(is_invalid(0x04BC5304), "04BC5304 RDVL Rn!=11111 should be Invalid");
    assert!(!is_invalid(0x04BF5304), "04BF5304 RDVL should decode");
    assert_roundtrip(0x04BF5304);
    // UDOT/SDOT 2-way `.h` indexed requires `<23:22>=10`.
    assert!(is_invalid(0x44D6CF8A), "44D6CF8A udot .h idx <22>=1 should be Invalid");
    assert!(!is_invalid(0x4496CF8A), "4496CF8A udot .h idx should decode");
    assert_roundtrip(0x4496CF8A);
}

// ===========================================================================
// Added gaps — decode to the expected mnemonic and round-trip.
// ===========================================================================

#[test]
fn multivector_narrowing_shift() {
    // SVE2.1 `op Zd.<Tb>, { Zn.<T>, Zn+1.<T> }, #shift` (the `<23>=1` family).
    let cases: &[(u32, &str)] = &[
        (0x45A80000, "sqshrn"),
        (0x45A82800, "sqrshrn"),
        (0x45A81000, "uqshrn"),
        (0x45A83800, "uqrshrn"),
        (0x45A82000, "sqshrun"),
        (0x45A80800, "sqrshrun"),
        (0x45AF0000, "sqshrn"),  // .b<-.h, #1
        (0x45BF0000, "sqshrn"),  // .h<-.s, #1
        (0x45B00045, "sqshrn"),  // .h<-.s, #16
    ];
    for &(w, m) in cases {
        assert_eq!(mnem(w), m, "{:08X} mnemonic", w);
        assert_roundtrip(w);
    }
    // Odd source-pair base is reserved.
    assert!(is_invalid(0x45A80020), "odd Zn base should be Invalid");
}

#[test]
fn cntp_predicate_as_counter() {
    // SVE2.1 `CNTP <Xd>, <PNn>.<T>, VLx{2,4}` (full pn0..pn15 range).
    let cases = [
        0x25208300u32, // cntp x0, pn8.b, vlx2
        0x25208700,    // cntp x0, pn8.b, vlx4
        0x25608300,    // cntp x0, pn8.h, vlx2
        0x25A08300,    // cntp x0, pn8.s, vlx2
        0x25E08300,    // cntp x0, pn8.d, vlx2
        0x25208200,    // cntp x0, pn0.b, vlx2
        0x2520826D,    // cntp x13, pn3.b, vlx2
    ];
    for w in cases {
        assert_eq!(mnem(w), "cntp", "{:08X} mnemonic", w);
        assert_roundtrip(w);
    }
}

#[test]
fn sve2p2_fp8_bf16_gaps() {
    let cases: &[(u32, &str)] = &[
        (0x65083CF9, "bf2cvt"),  // bf2cvt z25.h, z7.b
        (0x654C3A4A, "scvtflt"), // scvtflt z10.h, z18.b
        (0x65099846, "bfscale"), // bfscale z6.h, p6/m, z6.h, z2.h
    ];
    for &(w, m) in cases {
        assert_eq!(mnem(w), m, "{:08X} mnemonic", w);
        assert_roundtrip(w);
    }
}

// ===========================================================================
// Feature gating for the added families.
// ===========================================================================

fn without(fs: FeatureSet, f: Feature) -> FeatureSet {
    let bit = f as u32;
    FeatureSet { features0: fs.features0 & !(1u64 << bit), features1: fs.features1 & !(1u64 << bit) }
}

#[test]
fn added_families_feature_gated() {
    // Multi-vector narrowing shift + CNTP-as-counter require SVE2.1.
    for w in [0x45A80000u32, 0x25208300] {
        assert!(!decode(w, 0, FeatureSet::ALL).is_invalid(), "{:08X} should decode with all", w);
        assert!(decode(w, 0, without(FeatureSet::ALL, Feature::Sve2p1)).is_invalid(),
            "{:08X} should require Sve2p1", w);
    }
    // BFSCALE requires SVE_B16B16; the FP8 converts require FP8.
    assert!(decode(0x65099846, 0, without(FeatureSet::ALL, Feature::SveB16b16)).is_invalid(),
        "bfscale should require SveB16b16");
    for w in [0x65083CF9u32, 0x654C3A4A] {
        assert!(decode(w, 0, without(FeatureSet::ALL, Feature::Fp8)).is_invalid(),
            "{:08X} should require Fp8", w);
    }
}
