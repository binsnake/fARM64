//! SME outer-product family (H2): over-decode hardening, mnemonic repair, and
//! the FEAT_SME_MOP4 4-source forms.
//!
//! Covers the predicated outer products (`FMOPA`/`FMOPS`, `BFMOPA`/`BFMOPS`,
//! `BMOPA`/`BMOPS`, the FP8/FP16/BF16 non-widening variants, `SMOPA`/`UMOPA`/
//! `SUMOPA`/`USMOPA` integer with `.b`/`.h` sources) and the MOP4 quarter-tile
//! forms (`FMOP4A`/`SMOP4A`/...), all validated against LLVM (`--mattr=+all`).
//!
//! fARM64 renders the `ZAda` accumulator and MOP4 `Zn`/`Zm` sources with a plain
//! `z` prefix (binja style) where LLVM writes `za`; the *mnemonic* (first token)
//! and operand structure are what must match.

#![cfg(feature = "std")]

use fARM64::decode::decode;
use fARM64::format::{format_to_string, FmtFormatter};
use fARM64::{encode, Feature, FeatureSet};

fn render(word: u32) -> String {
    let insn = decode(word, 0x1000, FeatureSet::ALL);
    if insn.is_invalid() {
        return "<invalid>".to_string();
    }
    format_to_string(&FmtFormatter::new(), &insn)
}

#[track_caller]
fn check(word: u32, expected: &str) {
    assert_eq!(render(word), expected, "word={word:#010x}");
    // Every accepted word must round-trip exactly.
    let insn = decode(word, 0x1000, FeatureSet::ALL);
    let enc = encode(&insn)
        .unwrap_or_else(|e| panic!("{word:#010x} ({}) encode err {e:?}", insn.mnemonic().name()));
    assert_eq!(enc, word, "{word:#010x} re-encoded {enc:#010x}");
}

#[track_caller]
fn invalid(word: u32) {
    assert!(
        decode(word, 0x1000, FeatureSet::ALL).is_invalid(),
        "word={word:#010x} should be Invalid, got {}",
        render(word)
    );
}

// ---------------------------------------------------------------------------
// Predicated FP / BF16 outer products.
// ---------------------------------------------------------------------------

#[test]
fn predicated_fp_examples() {
    // FP32 / FP64 / FP16->FP32.
    check(0x809B4941, "fmopa   z1.s, p2/m, p2/m, z10.s, z27.s");
    check(0x80851312, "fmops   z2.s, p4/m, p0/m, z24.s, z5.s");
    check(0x80D69446, "fmopa   z6.d, p5/m, p4/m, z2.d, z22.d");
    check(0x81BB7F43, "fmopa   z3.s, p7/m, p3/m, z26.h, z27.h");
    // BF16->FP32.
    check(0x8184B942, "bfmopa  z2.s, p6/m, p5/m, z10.h, z4.h");
    check(0x819EC990, "bfmops  z0.s, p2/m, p6/m, z12.h, z30.h");
    // FP8->FP32 (FEAT_SME_F8F32): `.b` source, accumulate-only.
    check(0x80BB4940, "fmopa   z0.s, p2/m, p2/m, z10.b, z27.b");
    // FP8->FP16 (FEAT_SME_F8F16): `.h` destination, `.b` source.
    check(0x80BB4948, "fmopa   z0.h, p2/m, p2/m, z10.b, z27.b");
}

#[test]
fn mnemonic_repair_b16b16_and_fp16() {
    // Previously mis-identified as fmops/fmopa: these are b16b16 BMOPA/BMOPS.
    check(0x8091921A, "bmops   z2.s, p4/m, p4/m, z16.s, z17.s");
    check(0x809A39C8, "bmopa   z0.s, p6/m, p1/m, z14.s, z26.s");
    // `.h`-destination forms: BFMOPA (b16b16) and FMOPS (fp16->fp16).
    check(0x81BBCD88, "bfmopa  z0.h, p3/m, p6/m, z12.h, z27.h");
    check(0x819F55D9, "fmops   z1.h, p5/m, p2/m, z14.h, z31.h");
}

// ---------------------------------------------------------------------------
// Predicated integer outer products.
// ---------------------------------------------------------------------------

#[test]
fn predicated_int_examples() {
    // 32-bit accumulator, `.b` sources: the four signedness pairings.
    check(0xA0822DA1, "smopa   z1.s, p3/m, p1/m, z13.b, z2.b");
    check(0xA1A98383, "umopa   z3.s, p0/m, p4/m, z28.b, z9.b");
    check(0xA0A0AC03, "sumopa  z3.s, p3/m, p5/m, z0.b, z0.b");
    check(0xA19670C1, "usmopa  z1.s, p4/m, p3/m, z6.b, z22.b");
    // 64-bit accumulator, `.h` sources (I16I64).
    check(0xA0CFCB26, "smopa   z6.d, p2/m, p6/m, z25.h, z15.h");
    check(0xA1E301C4, "umopa   z4.d, p0/m, p0/m, z14.h, z3.h");
}

#[test]
fn mnemonic_repair_int_16bit() {
    // 32-bit accumulator with `.h` sources (FEAT_SME2): previously mis-spelled
    // usmops/usmopa — these are UMOPS/UMOPA.
    check(0xA191921A, "umops   z2.s, p4/m, p4/m, z16.h, z17.h");
    check(0xA19A39C8, "umopa   z0.s, p6/m, p1/m, z14.h, z26.h");
}

// ---------------------------------------------------------------------------
// Over-decode hardening: reserved-bit and unallocated-slot words are Invalid.
// ---------------------------------------------------------------------------

#[test]
fn over_decode_words_are_invalid() {
    // `8099B44C`: FMOPA `.s` with `word<2> == 1` (reserved ZAda high bit).
    invalid(0x8099B44C);
    // `80A66D13`: op24=0, sz=10, b21=1, S=1 — there is no FMOPS for the FP8 slot.
    invalid(0x80A66D13);
    // Integer `.s` with `word<2> == 1` reserved.
    invalid(0xA08453D7);
    // SUMOPA has no `.h`-source (16-bit) form: `word<3> == 1` here is UNDEFINED.
    invalid(0xA0BBCD88);
    // USMOPA likewise has no 16-bit form.
    invalid(0xA1BBCD88);
}

// ---------------------------------------------------------------------------
// MOP4 (FEAT_SME_MOP4) 4-source outer products.
// ---------------------------------------------------------------------------

#[test]
fn mop4_examples() {
    // FP64 / signed-int `.d` (the task's small-gap examples).
    check(0x80DE018B, "fmop4a  z3.d, z12.d, { z30.d, z31.d }");
    check(0xA0DE018B, "smop4a  z3.d, z12.h, { z30.h, z31.h }");
    check(0x80DE0189, "fmop4a  z1.d, z12.d, { z30.d, z31.d }");
    check(0xA0DE0189, "smop4a  z1.d, z12.h, { z30.h, z31.h }");
    // FP32 MOP4, single Zn / single Zm.
    check(0x80000180, "fmop4a  z0.s, z12.s, z16.s");
    // Subtract form.
    check(0x80000190, "fmop4s  z0.s, z12.s, z16.s");
    // FP8 -> FP32 / FP8 -> FP16 (accumulate only).
    check(0x80200180, "fmop4a  z0.s, z12.b, z16.b");
    check(0x80200188, "fmop4a  z0.h, z12.b, z16.b");
    // BF16 -> FP32 / FP16 -> FP16 / FP16 -> FP32 / BF16 -> BF16.
    check(0x81000180, "bfmop4a z0.s, z12.h, z16.h");
    check(0x81000188, "fmop4a  z0.h, z12.h, z16.h");
    check(0x81200180, "fmop4a  z0.s, z12.h, z16.h");
    check(0x81200188, "bfmop4a z0.h, z12.h, z16.h");
    // Integer `.s` MOP4: `.b` and `.h` sources (the two umop4s encodings differ
    // by source size, NOT the same instruction).
    check(0x813C8012, "umop4s  z2.s, z0.b, { z28.b, z29.b }");
    check(0x811C801A, "umop4s  z2.s, z0.h, { z28.h, z29.h }");
    // Integer `.d` MOP4 (signed-by-unsigned), single Zm source.
    check(0xA0E00188, "sumop4a z0.d, z12.h, z16.h");
}

#[test]
fn mop4_pair_sources() {
    // `{Zn, Zn+1}` first source (word<9> selects the pair); `{Zm, Zm+1}` second.
    check(0x80DE020B, "fmop4a  z3.d, { z0.d, z1.d }, { z30.d, z31.d }");
    // Single Zm (word<20> == 0): base = 16 + (word<19:17> << 1).
    check(0x80C0018F, "fmop4a  z7.d, z12.d, z16.d");
}

#[test]
fn mop4_reserved_bits_invalid() {
    // Odd `Zn` (word<5> == 1) is reserved.
    invalid(0x80DE018B | 0x0000_0020);
    // `Zm<0>` (word<16> == 1) is reserved.
    invalid(0x80DE018B | 0x0001_0000);
    // The low predicate-position bits `word<14:10>` must be zero.
    invalid(0x80DE018B | 0x0000_0400);
    // FP MOP4 `.d` has no signed-by-unsigned form (`word<21> == 1` is UNDEFINED).
    invalid(0x80E0018B);
}

// ---------------------------------------------------------------------------
// Feature gating.
// ---------------------------------------------------------------------------

#[test]
fn feature_gating() {
    // FEAT_SME alone keeps the base FMOPA but not the b16b16 / MOP4 forms.
    let base_sme = FeatureSet {
        features0: 0,
        features1: 0,
    }
    .with(Feature::Sme);
    assert!(!decode(0x809B4941, 0x1000, base_sme).is_invalid()); // fmopa .s
    assert!(decode(0x8091921A, 0x1000, base_sme).is_invalid()); // bmops needs B16B16
    assert!(decode(0x80DE018B, 0x1000, base_sme).is_invalid()); // fmop4a needs MOP4

    // With the specific features added, they decode.
    let with_b16 = base_sme.with(Feature::SmeB16b16);
    assert!(!decode(0x8091921A, 0x1000, with_b16).is_invalid());
    let with_mop4 = base_sme.with(Feature::SmeMop4);
    assert!(!decode(0x80DE018B, 0x1000, with_mop4).is_invalid());

    // The 16-bit integer `.s` forms need FEAT_SME2.
    let with_sme2 = base_sme.with(Feature::Sme2);
    assert!(decode(0xA191921A, 0x1000, base_sme).is_invalid());
    assert!(!decode(0xA191921A, 0x1000, with_sme2).is_invalid());
}

// ---------------------------------------------------------------------------
// Round-trip sweep: every word accepted in the outer-product region must
// re-encode to itself.
// ---------------------------------------------------------------------------

#[test]
fn roundtrip_sweep() {
    let his = [0x80u32, 0x81, 0x82, 0x83, 0xA0, 0xA1];
    for hi in his {
        let mut state: u32 = 0x9E37_79B9 ^ hi;
        for _ in 0..20000 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let word = (hi << 24) | (state & 0x01FF_FFFF);
            let insn = decode(word, 0x1000, FeatureSet::ALL);
            if insn.is_invalid() {
                continue;
            }
            let enc = encode(&insn).unwrap_or_else(|e| {
                panic!("{word:#010x} ({}) encode err {e:?}", insn.mnemonic().name())
            });
            assert_eq!(enc, word, "{word:#010x} ({}) re-encoded {enc:#010x}", insn.mnemonic().name());
        }
    }
}
