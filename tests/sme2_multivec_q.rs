//! Q batch — SME2 `0xC0`-`0xC3` multi-vector gap closure (additive).
//!
//! Six families, all LLVM-`+all` oracle renderings:
//!
//! * **Q1 — multi × multi-vector in-place ALU** (`{Zdn}, {Zdn}, {Zm}`): integer
//!   `smin/smax/umin/umax/srshl/urshl/sqdmulh`, FP `famax/famin/fscale`, and the
//!   BF16 (`size==00`) `bfmax/bfmin/bfmaxnm/bfminnm/bfscale`. Slot `0xC1`,
//!   `<21>=1`, `<15:12>=1011`. The non-`.b` FP `fmax/fmin/fmaxnm/fminnm` siblings
//!   are decoded by the pre-existing `SME2_ALU_FORMS` table.
//! * **Q2 — ZA-array-vector MOV/MOVAZ** (`.d`-only array slice `za.d[Ws, off,
//!   vgxN]`, both directions). Slot `0xC0`, `<21:19>=000`, `<18>=1`, `<16>=0`,
//!   `<11>=1`, `<12>=0`.
//! * **Q3 — single-vector LUTI6 (ZT0)** `luti6 Zd.b, zt0, Zn`. Slot `0xC0`,
//!   `<23:16>=0xC8`, `<15:10>=010000`.
//! * **Q4 — BF16 ZA-array accumulate** `bfadd`/`bfsub` (the `<22>=1` siblings of
//!   the `fadd`/`fsub` `.h` ZA-array forms).
//! * **Q5 — FP8 convert**: narrow `fcvt`/`fcvtn`/`bfcvt` (`Zd.b, {Zn..}`), widen
//!   `f1/f2/bf1/bf2 cvt(l)` (`{Zd.h, Zd+1.h}, Zn.b`). Slot `0xC1`,
//!   `<15:10>=111000`.
//! * **Q6 — FP round / int-convert** (`.s`-only, `{Zd.s..}, {Zn.s..}`):
//!   `frintn/p/m/a`, `scvtf`/`ucvtf`/`fcvtzs`/`fcvtzu`, vgx2/vgx4. Same slot as Q5.
//!
//! Every canonical word + rendering is the LLVM oracle; reserved words are
//! `<unknown>` in LLVM.

#![cfg(feature = "std")]

use fARM64::decode::decode;
use fARM64::format::{BufSink, FmtFormatter, Formatter};
use fARM64::{encode, Feature, FeatureSet};

/// Render `word` to its disassembly with the mnemonic/operand padding collapsed to
/// a single space.
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

/// Decode `word`, re-encode, require an identical word; then re-decode and require
/// mnemonic + operand-count stability.
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
    // Q1: multi × multi-vector in-place ALU.
    (0xC126B038, "smin { z24.b, z25.b }, { z24.b, z25.b }, { z6.b, z7.b }"),
    (0xC1ACB824, "smin { z4.s - z7.s }, { z4.s - z7.s }, { z12.s - z15.s }"),
    (0xC174B01D, "umax { z28.h, z29.h }, { z28.h, z29.h }, { z20.h, z21.h }"),
    (0xC1B8B03F, "umin { z30.s, z31.s }, { z30.s, z31.s }, { z24.s, z25.s }"),
    (0xC1ACB018, "smax { z24.s, z25.s }, { z24.s, z25.s }, { z12.s, z13.s }"),
    (0xC13CB225, "urshl { z4.b, z5.b }, { z4.b, z5.b }, { z28.b, z29.b }"),
    (0xC174B220, "srshl { z0.h, z1.h }, { z0.h, z1.h }, { z20.h, z21.h }"),
    (0xC120B40E, "sqdmulh { z14.b, z15.b }, { z14.b, z15.b }, { z0.b, z1.b }"),
    (0xC1E4B148, "famax { z8.d, z9.d }, { z8.d, z9.d }, { z4.d, z5.d }"),
    (0xC1ACB14D, "famin { z12.s, z13.s }, { z12.s, z13.s }, { z12.s, z13.s }"),
    (0xC1A2B18A, "fscale { z10.s, z11.s }, { z10.s, z11.s }, { z2.s, z3.s }"),
    // Q1: BF16 (size==00) re-types.
    (0xC126B106, "bfmax { z6.h, z7.h }, { z6.h, z7.h }, { z6.h, z7.h }"),
    (0xC128B103, "bfmin { z2.h, z3.h }, { z2.h, z3.h }, { z8.h, z9.h }"),
    (0xC130B93C, "bfmaxnm { z28.h - z31.h }, { z28.h - z31.h }, { z16.h - z19.h }"),
    (0xC132B12D, "bfminnm { z12.h, z13.h }, { z12.h, z13.h }, { z18.h, z19.h }"),
    (0xC132B194, "bfscale { z20.h, z21.h }, { z20.h, z21.h }, { z18.h, z19.h }"),
    // Q2: ZA-array-vector MOV/MOVAZ (.d only).
    (0xC006283E, "mov { z30.d, z31.d }, za.d[w9, 1, vgx2]"),
    (0xC0062C44, "mov { z4.d - z7.d }, za.d[w9, 2, vgx4]"),
    (0xC0044843, "mov za.d[w10, 3, vgx2], { z2.d, z3.d }"),
    (0xC0044F81, "mov za.d[w10, 1, vgx4], { z28.d - z31.d }"),
    (0xC0060EAC, "movaz { z12.d - z15.d }, za.d[w8, 5, vgx4]"),
    (0xC0062AA8, "movaz { z8.d, z9.d }, za.d[w9, 5, vgx2]"),
    // Q3: single-vector LUTI6 (ZT0).
    (0xC0C84009, "luti6 z9.b, zt0, z0"),
    (0xC0C843E0, "luti6 z0.b, zt0, z31"),
    // Q4: BF16 ZA-array accumulate.
    (0xC1E41C00, "bfadd za.h[w8, 0, vgx2], { z0.h, z1.h }"),
    (0xC1E51C00, "bfadd za.h[w8, 0, vgx4], { z0.h - z3.h }"),
    (0xC1E41C08, "bfsub za.h[w8, 0, vgx2], { z0.h, z1.h }"),
    (0xC1E51C08, "bfsub za.h[w8, 0, vgx4], { z0.h - z3.h }"),
    // Q5: FP8 convert (narrow).
    (0xC134E000, "fcvt z0.b, { z0.s - z3.s }"),
    (0xC134E020, "fcvtn z0.b, { z0.s - z3.s }"),
    (0xC124E000, "fcvt z0.b, { z0.h, z1.h }"),
    (0xC164E000, "bfcvt z0.b, { z0.h, z1.h }"),
    // Q5: FP8 convert (widen).
    (0xC126E000, "f1cvt { z0.h, z1.h }, z0.b"),
    (0xC126E001, "f1cvtl { z0.h, z1.h }, z0.b"),
    (0xC1A6E000, "f2cvt { z0.h, z1.h }, z0.b"),
    (0xC1A6E001, "f2cvtl { z0.h, z1.h }, z0.b"),
    (0xC166E000, "bf1cvt { z0.h, z1.h }, z0.b"),
    (0xC166E001, "bf1cvtl { z0.h, z1.h }, z0.b"),
    (0xC1E6E000, "bf2cvt { z0.h, z1.h }, z0.b"),
    (0xC1E6E001, "bf2cvtl { z0.h, z1.h }, z0.b"),
    // Q6: FP round / int-convert.
    (0xC1A9E000, "frintp { z0.s, z1.s }, { z0.s, z1.s }"),
    (0xC1A8E000, "frintn { z0.s, z1.s }, { z0.s, z1.s }"),
    (0xC1AAE000, "frintm { z0.s, z1.s }, { z0.s, z1.s }"),
    (0xC1ACE000, "frinta { z0.s, z1.s }, { z0.s, z1.s }"),
    (0xC1B9E000, "frintp { z0.s - z3.s }, { z0.s - z3.s }"),
    (0xC122E000, "scvtf { z0.s, z1.s }, { z0.s, z1.s }"),
    (0xC122E020, "ucvtf { z0.s, z1.s }, { z0.s, z1.s }"),
    (0xC121E000, "fcvtzs { z0.s, z1.s }, { z0.s, z1.s }"),
    (0xC121E020, "fcvtzu { z0.s, z1.s }, { z0.s, z1.s }"),
    (0xC132E000, "scvtf { z0.s - z3.s }, { z0.s - z3.s }"),
];

#[test]
fn canonical_renders_and_roundtrip() {
    for &(word, expected) in CASES {
        assert_eq!(text(word), expected, "{word:08X} render mismatch");
        assert_roundtrip(word);
    }
}

/// Reserved encodings adjacent to the new families that LLVM leaves `<unknown>` —
/// fARM64 must keep them Invalid (no over-decode).
const RESERVED: &[u32] = &[
    // Q1: reserved Zm group bits. vgx2 `<16>` RES0; vgx4 `<17:16>` / `<1>` RES0.
    0xC121B020, // smin vgx2 with <16>=1
    0xC1FDB821, // umin vgx4 with <16>=1
    0xC120B420, // smin with <10>=1 (sqdmulh table requires sub 0)
    0xC120B060, // sub-opcode 3 (unallocated)
    // Q2: reserved bits. vec->ZA `<5:3>` RES0; to-vec `<8>`/`<0>` RES0; vgx4 `<1>`.
    0xC0040808, // vec->ZA <3>=1
    0xC0066801, // to-vec <0>=1
    0xC0066900, // to-vec <8>=1
    0xC0062EAE, // movaz vgx4 <1>=1
    // Q3: LUTI6 single reserved (<10>=1 / <12>=1 / <14>=0).
    0xC0C84400,
    0xC0C85000,
    0xC0C80000,
    // Q5/Q6: reserved opcode / size in the convert slot.
    0xC1A9E001, // frintp with <0>=1
    0xC134E040, // fcvt narrow .s4 with <6>=1 reserved
];

#[test]
fn reserved_stays_invalid() {
    for &word in RESERVED {
        let insn = decode(word, 0x1000, FeatureSet::ALL);
        assert!(insn.is_invalid(), "{word:08X} should be Invalid, got {}", text(word));
    }
}

/// Feature gating: the new families require their extension to decode.
#[test]
fn feature_gating() {
    // Q1 integer multi×multi requires FEAT_SME2.
    let base = FeatureSet::BASE;
    assert!(decode(0xC126B038, 0x1000, base).is_invalid(), "smin needs SME2");
    // Q1 BF16 re-types require FEAT_SME_B16B16 (a SME2-only set leaves them Invalid).
    let sme2_only = FeatureSet::BASE.with(Feature::Sme).with(Feature::Sme2);
    assert!(
        decode(0xC126B106, 0x1000, sme2_only).is_invalid(),
        "bfmax needs SME_B16B16"
    );
    // Q4 bfadd requires FEAT_SME_B16B16.
    assert!(
        decode(0xC1E41C00, 0x1000, sme2_only).is_invalid(),
        "bfadd needs SME_B16B16"
    );
    // Q3 LUTI6 needs FEAT_LUT.
    assert!(decode(0xC0C84009, 0x1000, sme2_only).is_invalid(), "luti6 needs LUT");
    // Q5 FP8 convert needs FEAT_SME_F8F16.
    assert!(
        decode(0xC124E000, 0x1000, sme2_only).is_invalid(),
        "fcvt FP8 needs SME_F8F16"
    );
    // ...but Q6 round/int-convert needs only FEAT_SME2.
    assert!(
        !decode(0xC1A9E000, 0x1000, sme2_only).is_invalid(),
        "frintp needs only SME2"
    );
}
