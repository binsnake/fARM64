//! SME / SME2 batch L3: additive SME2 multi-vector gaps in the `0xC1` region,
//! each LLVM (`oracle.py dec/enc`, i.e. `clang .inst` + `llvm-objdump
//! --mattr=+all`) validated bidirectionally (disasm matches LLVM mnemonic +
//! operands; encode round-trips bit-exact), with reserved-slot guards proven not
//! to over-decode neighbouring encodings.
//!
//! Families added (decode + encode):
//!
//! 1. **Multi-vector × single-vector in-place ALU** (`{ Zdn }, { Zdn }, Zm`;
//!    `word<15:12> == 1010`, `word<21> == 1`, `word<20> == 0`): integer
//!    `ADD`/`SMAX`/`UMAX`/`SMIN`/`UMIN`/`SRSHL`/`URSHL`/`SQDMULH`, floating-point
//!    `FMAX`/`FMIN`/`FMAXNM`/`FMINNM`/`FSCALE`, and the BF16 (`size == 00`, `.h`)
//!    `BFMAX`/`BFMIN`/`BFMAXNM`/`BFMINNM`/`BFSCALE` (FEAT_SME_B16B16). vgx2/vgx4.
//! 2. **BF16 multi-vector `BFMUL`** — the `size == 00` (`.h`) case of the FMUL
//!    multi×multi and multi×single opcode slots (FEAT_SME_B16B16).
//! 3. **Multi-vector unpack `SUNPK`/`UUNPK`** — vgx2 (`{ Zd, Zd+1 }, Zn`) and
//!    vgx4 (`{ Zd - Zd+3 }, { Zn, Zn+1 }`), source element half the destination.
//! 4. **Multi-vector saturating extract-narrow** `SQCVT`/`UQCVT`/`SQCVTU`/
//!    `SQCVTN`/`UQCVTN`/`SQCVTUN`, 2-register and 4-register source groups.
//! 5. **2-vector saturating rounding shift-right-narrow** `SQRSHR`/`UQRSHR`/
//!    `SQRSHRU` (`Zd.h, { Zn, Zn+1 }.s, #shift`).
//! 6. **Consecutive-destination `LUTI6`** — the K3 `LUTI6` covers the stride-4
//!    destination; this adds the 4-register consecutive destination form.

#![cfg(feature = "sme")]

use fARM64::decode::decode;
use fARM64::format::{format_to_string, FmtFormatter};
use fARM64::{encode, Feature, FeatureSet};

fn text(word: u32) -> String {
    format_to_string(&FmtFormatter::new(), &decode(word, 0, FeatureSet::ALL))
}

fn norm(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// `FeatureSet::ALL` minus a single feature bit.
fn without(fs: FeatureSet, feat: Feature) -> FeatureSet {
    let bit = feat as u32;
    FeatureSet {
        features0: fs.features0 & !(1u64 << bit),
        features1: fs.features1 & !(1u64 << bit),
    }
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

// ===========================================================================
// 1. Multi-vector × single-vector in-place ALU.
// ===========================================================================

#[test]
fn mvs_alu_examples() {
    // The task's canonical examples.
    check(0xC1E3A318, "add { z24.d, z25.d }, { z24.d, z25.d }, z3.d");
    check(0xC169A22A, "srshl { z10.h, z11.h }, { z10.h, z11.h }, z9.h");
    check(0xC1E6A939, "fminnm { z24.d - z27.d }, { z24.d - z27.d }, z6.d");
    check(0xC1EFA13C, "fmaxnm { z28.d, z29.d }, { z28.d, z29.d }, z15.d");
}

#[test]
fn mvs_alu_integer_family() {
    // smax/umax/smin/umin (word<0> selects signed/unsigned), all sizes, both vgx.
    check(0xC1A0A000, "smax { z0.s, z1.s }, { z0.s, z1.s }, z0.s");
    check(0xC1A0A001, "umax { z0.s, z1.s }, { z0.s, z1.s }, z0.s");
    check(0xC120A020, "smin { z0.b, z1.b }, { z0.b, z1.b }, z0.b");
    check(0xC160A021, "umin { z0.h, z1.h }, { z0.h, z1.h }, z0.h");
    check(0xC1E0A800, "smax { z0.d - z3.d }, { z0.d - z3.d }, z0.d");
    // srshl/urshl, add, and sqdmulh (the word<10> == 1 sub-table).
    check(0xC1A0A220, "srshl { z0.s, z1.s }, { z0.s, z1.s }, z0.s");
    check(0xC1A0A221, "urshl { z0.s, z1.s }, { z0.s, z1.s }, z0.s");
    check(0xC120A300, "add { z0.b, z1.b }, { z0.b, z1.b }, z0.b");
    check(0xC1A0A400, "sqdmulh { z0.s, z1.s }, { z0.s, z1.s }, z0.s");
    check(0xC1A0AC00, "sqdmulh { z0.s - z3.s }, { z0.s - z3.s }, z0.s");
    // The single multiplier ranges over z0..z15 (word<19:16>).
    check(0xC1AFA000, "smax { z0.s, z1.s }, { z0.s, z1.s }, z15.s");
}

#[test]
fn mvs_alu_fp_family() {
    // fmax/fmin/fmaxnm/fminnm/fscale (.h/.s/.d), and the BF16 (.b → .h) siblings.
    check(0xC1A0A100, "fmax { z0.s, z1.s }, { z0.s, z1.s }, z0.s");
    check(0xC1A0A101, "fmin { z0.s, z1.s }, { z0.s, z1.s }, z0.s");
    check(0xC160A120, "fmaxnm { z0.h, z1.h }, { z0.h, z1.h }, z0.h");
    check(0xC1E0A121, "fminnm { z0.d, z1.d }, { z0.d, z1.d }, z0.d");
    check(0xC1A0A180, "fscale { z0.s, z1.s }, { z0.s, z1.s }, z0.s");
    // BF16 (size == 00 renders .h).
    check(0xC120A100, "bfmax { z0.h, z1.h }, { z0.h, z1.h }, z0.h");
    check(0xC120A101, "bfmin { z0.h, z1.h }, { z0.h, z1.h }, z0.h");
    check(0xC120A120, "bfmaxnm { z0.h, z1.h }, { z0.h, z1.h }, z0.h");
    check(0xC120A121, "bfminnm { z0.h, z1.h }, { z0.h, z1.h }, z0.h");
    check(0xC120A180, "bfscale { z0.h, z1.h }, { z0.h, z1.h }, z0.h");
    check(0xC120A980, "bfscale { z0.h - z3.h }, { z0.h - z3.h }, z0.h");
}

#[test]
fn mvs_alu_reserved() {
    // word<21> == 0 is UNDEFINED (the whole slot key requires it set).
    reserved(0xC1C3A318); // add with word<21> == 0
    // word<20> == 1 (Zm beyond z15) is RES0.
    reserved(0xC130A818);
    reserved(0xC1F0A000);
    // ADD / FSCALE / SQDMULH have no word<0> selector; a set bit is UNDEFINED.
    reserved(0xC120A301); // add, word<0> == 1
    reserved(0xC1A0A181); // fscale, word<0> == 1
    reserved(0xC1A0A401); // sqdmulh, word<0> == 1
    // Floating-point `.b` (the non-BF16 FP ops) does not exist — only BF16 there.
    // vgx4 reserved low bit word<1>.
    reserved(0xC1E0AB02);
}

// ===========================================================================
// 2. BF16 multi-vector BFMUL (size == 00 of the FMUL slot).
// ===========================================================================

#[test]
fn bfmul() {
    check(0xC129EB80, "bfmul { z0.h - z3.h }, { z28.h - z31.h }, z4.h");
    // multi × multi (vgx2) and multi × single (vgx2).
    check(0xC120E400, "bfmul { z0.h, z1.h }, { z0.h, z1.h }, { z0.h, z1.h }");
    check(0xC120E800, "bfmul { z0.h, z1.h }, { z0.h, z1.h }, z0.h");
    // multi × multi (vgx4).
    check(0xC121E400, "bfmul { z0.h - z3.h }, { z0.h - z3.h }, { z0.h - z3.h }");
}

// ===========================================================================
// 3. Multi-vector unpack SUNPK / UUNPK.
// ===========================================================================

#[test]
fn unpk() {
    // vgx2: `{ Zd, Zd+1 }.<T>, Zn.<T/2>` — source element is half the dest.
    check(0xC165E0FF, "uunpk { z30.h, z31.h }, z7.b");
    check(0xC165E000, "sunpk { z0.h, z1.h }, z0.b");
    check(0xC1A5E000, "sunpk { z0.s, z1.s }, z0.h");
    check(0xC1E5E001, "uunpk { z0.d, z1.d }, z0.s");
    // vgx4: `{ Zd - Zd+3 }.<T>, { Zn, Zn+1 }.<T/2>`.
    check(0xC175E000, "sunpk { z0.h - z3.h }, { z0.b, z1.b }");
    check(0xC1B5E355, "uunpk { z20.s - z23.s }, { z26.h, z27.h }");
    check(0xC1F5E208, "sunpk { z8.d - z11.d }, { z16.s, z17.s }");
}

#[test]
fn unpk_reserved() {
    reserved(0xC125E000); // size == 00 (sub-byte source)
    reserved(0xC1B5E002); // vgx4 word<1> RES0
    reserved(0xC1B5E020); // vgx4 word<5> RES0
}

// ===========================================================================
// 4. Multi-vector saturating extract-narrow.
// ===========================================================================

#[test]
fn cvt_narrow() {
    // 4-register source, interleaved-narrow and non-interleaved.
    check(0xC1B3E0E4, "uqcvtn z4.h, { z4.d - z7.d }");
    check(0xC1B3E000, "sqcvt z0.h, { z0.d - z3.d }");
    check(0xC1B3E020, "uqcvt z0.h, { z0.d - z3.d }");
    check(0xC1B3E040, "sqcvtn z0.h, { z0.d - z3.d }");
    check(0xC1F3E000, "sqcvtu z0.h, { z0.d - z3.d }");
    check(0xC1F3E040, "sqcvtun z0.h, { z0.d - z3.d }");
    check(0xC133E000, "sqcvt z0.b, { z0.s - z3.s }");
    // 2-register source (only sqcvt/uqcvt/sqcvtu, .h ← .s).
    check(0xC123E000, "sqcvt z0.h, { z0.s, z1.s }");
    check(0xC123E020, "uqcvt z0.h, { z0.s, z1.s }");
    check(0xC163E000, "sqcvtu z0.h, { z0.s, z1.s }");
}

#[test]
fn cvt_narrow_reserved() {
    // 2-register source signed→unsigned with word<5> == 1 is UNDEFINED.
    reserved(0xC163E020);
    // word<21> == 0 (outside the family key).
    reserved(0xC1A3E000); // size 10 has no 2-register form
}

// ===========================================================================
// 5. 2-vector saturating rounding shift-right-narrow.
// ===========================================================================

#[test]
fn narrow_shift2() {
    check(0xC1EFD663, "uqrshr z3.h, { z18.s, z19.s }, #1");
    check(0xC1EAD519, "sqrshr z25.h, { z8.s, z9.s }, #6");
    check(0xC1FBD487, "sqrshru z7.h, { z4.s, z5.s }, #5");
    // Shift boundaries (#16 and #1).
    check(0xC1E0D400, "sqrshr z0.h, { z0.s, z1.s }, #16");
    check(0xC1EFD400, "sqrshr z0.h, { z0.s, z1.s }, #1");
}

#[test]
fn narrow_shift2_reserved() {
    // Unsigned input + unsigned result simultaneously is UNDEFINED.
    reserved(0xC1F0D420); // word<5> == 1 (uinput) and word<20> == 1 (uresult)
}

// ===========================================================================
// 6. Consecutive-destination LUTI6.
// ===========================================================================

#[test]
fn luti6_consecutive() {
    check(0xC12AF678, "luti6 { z24.h - z27.h }, { z19.h, z20.h }, { z10, z11 }[0]");
    check(0xC12AF600, "luti6 { z0.h - z3.h }, { z16.h, z17.h }, { z10, z11 }[0]");
    // The index bit (word<22>).
    check(0xC16AF600, "luti6 { z0.h - z3.h }, { z16.h, z17.h }, { z10, z11 }[1]");
    // The K3 stride-4 form still decodes alongside it.
    check(0xC132FDE3, "luti6 { z3.h, z7.h, z11.h, z15.h }, { z15.h, z16.h }, { z18, z19 }[0]");
}

#[test]
fn luti6_consecutive_reserved() {
    // The consecutive destination base must be a multiple of 4 (word<1:0> RES0).
    reserved(0xC12AF601);
    reserved(0xC12AF602);
}

// ===========================================================================
// Feature gating.
// ===========================================================================

#[test]
fn feature_gating() {
    let no_sme2 = without(FeatureSet::ALL, Feature::Sme2);
    let no_b16 = without(FeatureSet::ALL, Feature::SmeB16b16);
    let no_lut = without(FeatureSet::ALL, Feature::Lut);

    // Sme2-gated forms must not decode without FEAT_SME2.
    assert!(decode(0xC1E3A318, 0, no_sme2).is_invalid(), "add needs SME2");
    assert!(decode(0xC165E0FF, 0, no_sme2).is_invalid(), "uunpk needs SME2");
    assert!(decode(0xC1B3E0E4, 0, no_sme2).is_invalid(), "uqcvtn needs SME2");
    assert!(decode(0xC1EFD663, 0, no_sme2).is_invalid(), "uqrshr needs SME2");

    // BF16 forms need FEAT_SME_B16B16.
    assert!(decode(0xC120A100, 0, no_b16).is_invalid(), "bfmax needs B16B16");
    assert!(decode(0xC129EB80, 0, no_b16).is_invalid(), "bfmul needs B16B16");

    // LUTI6 needs FEAT_LUT.
    assert!(decode(0xC12AF678, 0, no_lut).is_invalid(), "luti6 needs LUT");
}
