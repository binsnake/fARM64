//! P batch — SVE2.2 FP8 / int convert cluster sharing the `0x65`, `<21>=0`,
//! `<15:13>=001` slot (all `<12>=1`).
//!
//! Four families, selected by the `<20:16>` opcode and `<11:10>` variant:
//!
//! * `<20:16>=01000` (size 00, FEAT_FP8): FP8 -> FP16/BF16 widen `Zd.h, Zn.b`,
//!   `<11:10>` = F1CVT/F2CVT/BF1CVT/BF2CVT.
//! * `<20:16>=01001` (size 00, FEAT_FP8): widen-long top `Zd.h, Zn.b`,
//!   `<11:10>` = F1CVTLT/F2CVTLT/BF1CVTLT/BF2CVTLT.
//! * `<20:16>=01010` (size 00, FEAT_FP8): FP16/FP32 -> FP8 convert-narrow from a
//!   consecutive 2-register source group `Zd.b, { Zn.<T>, Zn+1.<T> }` (even base),
//!   `<11:10>` = FCVTN(.h)/FCVTNB(.s)/BFCVTN(.h)/FCVTNT(.s).
//! * `<20:16>=01100` (size 01/10/11, FEAT_SVE2p3): int -> FP widen
//!   `Zd.<Tw>, Zn.<Tn>` (size 01 -> .h/.b, 10 -> .s/.h, 11 -> .d/.s),
//!   `<11:10>` = SCVTF/UCVTF/SCVTFLT/UCVTFLT.
//!
//! All canonical example words and renderings are the LLVM
//! (`clang`/`llvm-objdump --mattr=+all`) oracle; the reserved words are
//! `<unknown>` in LLVM.

#![cfg(feature = "std")]

use fARM64::decode::decode;
use fARM64::format::{BufSink, FmtFormatter, Formatter};
use fARM64::{encode, Feature, FeatureSet};

/// Render `word` to its textual disassembly, with the mnemonic/operand column
/// padding collapsed to a single space so the expected strings stay readable.
fn text(word: u32) -> String {
    let insn = decode(word, 0x1000, FeatureSet::ALL);
    let mut buf = [0u8; 160];
    let mut sink = BufSink::new(&mut buf);
    FmtFormatter::new().format(&insn, &mut sink);
    let raw = sink.as_str();
    match raw.split_once(char::is_whitespace) {
        Some((mnem, rest)) => format!("{mnem} {}", rest.trim_start()),
        None => raw.to_string(),
    }
}

/// Decode `word`, re-encode, require an identical word; then re-decode and
/// require mnemonic + operand-count stability.
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

/// `(word, expected disassembly)` pairs — the LLVM oracle renderings.
const CASES: &[(u32, &str)] = &[
    // FP8 -> FP16/BF16 widen (opc 01000, size 00).
    (0x65083113, "f1cvt z19.h, z8.b"),
    (0x65083513, "f2cvt z19.h, z8.b"),
    (0x65083913, "bf1cvt z19.h, z8.b"),
    (0x65083D13, "bf2cvt z19.h, z8.b"),
    // FP8 -> FP16/BF16 widen-long top (opc 01001, size 00).
    (0x65093113, "f1cvtlt z19.h, z8.b"),
    (0x65093513, "f2cvtlt z19.h, z8.b"),
    (0x65093913, "bf1cvtlt z19.h, z8.b"),
    (0x65093D13, "bf2cvtlt z19.h, z8.b"),
    // FP16/FP32 -> FP8 convert-narrow from group (opc 01010, size 00).
    (0x650A3113, "fcvtn z19.b, { z8.h, z9.h }"),
    (0x650A3513, "fcvtnb z19.b, { z8.s, z9.s }"),
    (0x650A3913, "bfcvtn z19.b, { z8.h, z9.h }"),
    (0x650A3D13, "fcvtnt z19.b, { z8.s, z9.s }"),
    // int -> FP widen (opc 01100), size 01 -> .h/.b.
    (0x654C3113, "scvtf z19.h, z8.b"),
    (0x654C3513, "ucvtf z19.h, z8.b"),
    (0x654C3913, "scvtflt z19.h, z8.b"),
    (0x654C3D13, "ucvtflt z19.h, z8.b"),
    // int -> FP widen, size 10 -> .s/.h.
    (0x658C3113, "scvtf z19.s, z8.h"),
    (0x658C3513, "ucvtf z19.s, z8.h"),
    (0x658C3913, "scvtflt z19.s, z8.h"),
    (0x658C3D13, "ucvtflt z19.s, z8.h"),
    // int -> FP widen, size 11 -> .d/.s.
    (0x65CC3113, "scvtf z19.d, z8.s"),
    (0x65CC3513, "ucvtf z19.d, z8.s"),
    (0x65CC3913, "scvtflt z19.d, z8.s"),
    (0x65CC3D13, "ucvtflt z19.d, z8.s"),
    // A few alternate register patterns (group base z0 / z16, alt dest).
    (0x650A3000, "fcvtn z0.b, { z0.h, z1.h }"),
    (0x650A3205, "fcvtn z5.b, { z16.h, z17.h }"),
    (0x654C3133, "scvtf z19.h, z9.b"),
];

#[test]
fn examples_decode_and_render() {
    for &(w, expected) in CASES {
        assert_eq!(text(w), expected, "{w:08X} rendering");
    }
}

#[test]
fn examples_roundtrip() {
    for &(w, _) in CASES {
        assert_roundtrip(w);
    }
}

/// Exhaustive round-trip over the FP8 single-source families (opc 01000/01001),
/// every variant and every source/destination register.
#[test]
fn fp8_widen_roundtrip_exhaustive() {
    for opc in [0b01000u32, 0b01001] {
        for var in 0u32..4 {
            for zn in 0u32..32 {
                for zd in 0u32..32 {
                    let w = (0b01100101u32 << 24)
                        | (opc << 16)
                        | (0b001 << 13)
                        | (1 << 12)
                        | (var << 10)
                        | (zn << 5)
                        | zd;
                    assert_roundtrip(w);
                }
            }
        }
    }
}

/// Exhaustive round-trip over the FP8 narrow group family (opc 01010): every
/// variant, every even source-group base, every destination register.
#[test]
fn fp8_narrow_group_roundtrip_exhaustive() {
    for var in 0u32..4 {
        for zn in (0u32..32).step_by(2) {
            for zd in 0u32..32 {
                let w = (0b01100101u32 << 24)
                    | (0b01010 << 16)
                    | (0b001 << 13)
                    | (1 << 12)
                    | (var << 10)
                    | (zn << 5)
                    | zd;
                assert_roundtrip(w);
            }
        }
    }
}

/// Exhaustive round-trip over the int->FP widen family (opc 01100): all three
/// sizes, every variant, every source/destination register.
#[test]
fn int_to_fp_roundtrip_exhaustive() {
    for size in [0b01u32, 0b10, 0b11] {
        for var in 0u32..4 {
            for zn in 0u32..32 {
                for zd in 0u32..32 {
                    let w = (0b01100101u32 << 24)
                        | (size << 22)
                        | (0b01100 << 16)
                        | (0b001 << 13)
                        | (1 << 12)
                        | (var << 10)
                        | (zn << 5)
                        | zd;
                    assert_roundtrip(w);
                }
            }
        }
    }
}

/// Reserved neighbours must stay `Invalid` (all `<unknown>` in LLVM).
#[test]
fn reserved_neighbours_invalid() {
    let reserved = [
        // FP8 widen (opc 01000) with size != 00.
        0x65483113u32,
        0x65883113,
        0x65C83113,
        // FP8 widen-long (opc 01001) with size != 00.
        0x65493113,
        // FP8 narrow group (opc 01010) with an odd source-group base (z9).
        0x650A3133,
        0x650A3533,
        0x650A3933,
        0x650A3D33,
        // FP8 narrow group with size != 00.
        0x654A3113,
        // int->FP widen (opc 01100) with size == 00 (reserved).
        0x650C3113,
        0x650C3513,
        0x650C3913,
        0x650C3D13,
        // `<12>` == 0 in these opcodes is not part of the cluster.
        0x65082113,
        0x650A2113,
        0x654C2113,
    ];
    for &w in &reserved {
        assert!(decode(w, 0, FeatureSet::ALL).is_invalid(), "{w:08X} must be Invalid");
    }
}

/// FEAT_FP8 gating: the FP8 widen / narrow families need `Fp8` and are otherwise
/// `Invalid` (with the rest of SVE enabled).
#[test]
fn fp8_families_gated_on_fp8() {
    let base = FeatureSet::BASE
        .with(Feature::Sve)
        .with(Feature::Sve2p1)
        .with(Feature::Sve2p2)
        .with(Feature::Sve2p3)
        .with(Feature::Fp16);
    let with_fp8 = base.with(Feature::Fp8);
    let fp8_words = [
        0x65083113u32, // f1cvt
        0x65093113,    // f1cvtlt
        0x650A3113,    // fcvtn
        0x650A3513,    // fcvtnb
    ];
    for &w in &fp8_words {
        assert!(decode(w, 0, base).is_invalid(), "{w:08X} must need Fp8");
        assert!(!decode(w, 0, with_fp8).is_invalid(), "{w:08X} should decode with Fp8");
    }
}

/// FEAT_SVE2p3 gating: the int->FP widen family needs `Sve2p3` and is otherwise
/// `Invalid` (even with Fp8 + SVE2.2 enabled).
#[test]
fn int_to_fp_gated_on_sve2p3() {
    let base = FeatureSet::BASE
        .with(Feature::Sve)
        .with(Feature::Sve2p1)
        .with(Feature::Sve2p2)
        .with(Feature::Fp8)
        .with(Feature::Fp16);
    let with_p3 = base.with(Feature::Sve2p3);
    let words = [0x654C3113u32, 0x654C3513, 0x654C3913, 0x654C3D13, 0x658C3113, 0x65CC3113];
    for &w in &words {
        assert!(decode(w, 0, base).is_invalid(), "{w:08X} must need Sve2p3");
        assert!(!decode(w, 0, with_p3).is_invalid(), "{w:08X} should decode with Sve2p3");
    }
}

// ===========================================================================
// SME2 multi-vector FP convert (FP16/BF16 <-> FP32) — `0xC1`, `<21>=1`,
// `<20:16>=00000`, `<15:10>=111000`.
// ===========================================================================

/// `(word, expected disassembly)` pairs — the LLVM oracle renderings.
const SME_CASES: &[(u32, &str)] = &[
    // Narrow: FP32 group -> FP16/BF16 single.
    (0xC120E000, "fcvt z0.h, { z0.s, z1.s }"),
    (0xC120E060, "fcvtn z0.h, { z2.s, z3.s }"),
    (0xC160E000, "bfcvt z0.h, { z0.s, z1.s }"),
    (0xC160E060, "bfcvtn z0.h, { z2.s, z3.s }"),
    // Widen: FP16 single -> FP32 group.
    (0xC1A0E000, "fcvt { z0.s, z1.s }, z0.h"),
    (0xC1A0E003, "fcvtl { z2.s, z3.s }, z0.h"),
    // Alternate register patterns.
    (0xC120E040, "fcvt z0.h, { z2.s, z3.s }"),
    (0xC120E102, "fcvt z2.h, { z8.s, z9.s }"),
    (0xC1A0E044, "fcvt { z4.s, z5.s }, z2.h"),
];

#[test]
fn sme_examples_decode_and_render() {
    for &(w, expected) in SME_CASES {
        assert_eq!(text(w), expected, "{w:08X} rendering");
    }
}

#[test]
fn sme_examples_roundtrip() {
    for &(w, _) in SME_CASES {
        assert_roundtrip(w);
    }
}

/// Exhaustive round-trip over the SME2 narrow forms (sz 00/01, both variants),
/// every even source-group base and every destination register.
#[test]
fn sme_narrow_roundtrip_exhaustive() {
    for size in [0u32, 1] {
        for interleave in [0u32, 1] {
            for zn in (0u32..32).step_by(2) {
                for zd in 0u32..32 {
                    let w = 0xc120_e000
                        | (size << 22)
                        | ((zn / 2) << 6)
                        | (interleave << 5)
                        | zd;
                    assert_roundtrip(w);
                }
            }
        }
    }
}

/// Exhaustive round-trip over the SME2 widen form (sz 10, both variants), every
/// even destination-group base and every source register.
#[test]
fn sme_widen_roundtrip_exhaustive() {
    for interleave in [0u32, 1] {
        for zd in (0u32..32).step_by(2) {
            for zn in 0u32..32 {
                let w = 0xc1a0_e000 | ((zd / 2) << 1) | (zn << 5) | interleave;
                assert_roundtrip(w);
            }
        }
    }
}

/// Reserved SME2 neighbours must stay `Invalid` (all `<unknown>` in LLVM). The
/// only reserved sub-case inside this `<20:16>=00000`, `<15:10>=111000` slot is
/// `<23:22>=11`; the odd-base / `<20:16>!=0` words belong to *other* families.
#[test]
fn sme_reserved_neighbours_invalid() {
    for &w in &[0xC1E0E000u32, 0xC1E0E060] {
        assert!(decode(w, 0, FeatureSet::ALL).is_invalid(), "{w:08X} must be Invalid");
    }
}

/// FEAT_SME_F16F16 gating: the interleaving narrow variants (`FCVTN`/`BFCVTN`)
/// and the whole widen direction need `SmeF16f16`; the plain narrow `FCVT`/
/// `BFCVT` need only `Sme2`.
#[test]
fn sme_f16f16_gating() {
    let sme2_only = FeatureSet::BASE.with(Feature::Sme).with(Feature::Sme2);
    let with_f16 = sme2_only.with(Feature::SmeF16f16);
    // Plain narrow FCVT / BFCVT: decode with SME2 alone.
    for &w in &[0xC120E000u32, 0xC160E000] {
        assert!(!decode(w, 0, sme2_only).is_invalid(), "{w:08X} should decode with Sme2");
    }
    // Interleaving narrow + widen: need SmeF16f16.
    for &w in &[0xC120E060u32, 0xC160E060, 0xC1A0E000, 0xC1A0E003] {
        assert!(decode(w, 0, sme2_only).is_invalid(), "{w:08X} must need SmeF16f16");
        assert!(!decode(w, 0, with_f16).is_invalid(), "{w:08X} should decode with SmeF16f16");
    }
}
