//! SVE / SVE2 memory (load / store / prefetch) over-decode hardening, batch I2.
//!
//! Each reserved condition below was derived from an LLVM (`clang .inst` +
//! `llvm-objdump --mattr=+all`) field sweep over the SVE memory quadrants
//! (`0x84`/`0x85`/`0xA4`/`0xA5`/`0xE4`/`0xE5`) and proven 0-regression with a
//! pre/post differential vs LLVM (~62k over-decoded words eliminated in-sweep,
//! 0 LLVM-valid words newly rejected, 0 valid words lost).
//!
//! Families hardened:
//!  1. **Scalar+scalar `Xm == 31` (xzr)** — for every non-fault-suppressing
//!     contiguous / replicating / non-temporal / structured load/store and the
//!     scalar+scalar prefetch, the no-offset case is the immediate form, so an
//!     `xzr` index is UNDEFINED. (`LDFF1*` is exempt: it renders `[Xn]`.)
//!  2. **32-bit-element gather loading a 64-bit value** — in the `.s`-destination
//!     quadrant, `LD1D`/`LDFF1D`/`LDNT1D` (`msz==3`) and the signed-word
//!     `LD1SW`/`LDFF1SW`/`LDNT1SW` (`msz==2`) are reserved (the value cannot fit
//!     a `.s` lane; those forms live in the `.d` quadrant `0xC4`/`0xC5`).
//!  3. **Gather/contiguous PREFETCH `prfop<4> != 0`** — `prfop` is a 4-bit field;
//!     `word<4>` is RES0.
//!  4. **Scatter store reserved size/scale forms** — byte stores cannot scale
//!     (`msz==0 && scale`), a `.s` (32-bit) offset cannot store a dword
//!     (`msz==3 && b22`), STNT1 has `b21` RES0, and the vec+imm form cannot store
//!     a dword into a `.s` element.
//!  5. **`LD1RQ*`/`LD1RO*` `word<22>` RES0 + imm-form `word<20>` RES0**, the
//!     structured-load imm form `word<20>` RES0, and the **`LDR`/`STR` predicate**
//!     transfer `Pt<4>` RES0.

#![cfg(feature = "std")]

use fARM64::decode::decode;
use fARM64::format::{format_to_string, FmtFormatter};
use fARM64::{encode, Feature, FeatureSet};

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

// ---------------------------------------------------------------------------
// Valid forms — must still decode + round-trip exactly (regression guard).
// ---------------------------------------------------------------------------

#[test]
fn valid_forms_still_decode_and_roundtrip() {
    // Scalar+scalar contiguous / replicating / non-temporal / structured.
    check(0xA40A4000, "ld1b {z0.b}, p0/z, [x0, x10]");
    check(0xA5E54000, "ld1d {z0.d}, p0/z, [x0, x5, lsl #0x3]");
    check(0xE4044861, "st1b {z1.b}, p2, [x3, x4]");
    check(0xA401C000, "ldnt1b {z0.b}, p0/z, [x0, x1]");
    check(0xA421C000, "ld2b {z0.b, z1.b}, p0/z, [x0, x1]");
    check(0xA4010000, "ld1rqb {z0.b}, p0/z, [x0, x1]");
    // Gather loads (32-bit and 64-bit element).
    check(0x84BB1101, "ld1sh {z1.s}, p4/z, [x8, z27.s, uxtw #0x1]");
    check(0x85014000, "ld1w {z0.s}, p0/z, [x0, z1.s, uxtw]");
    check(0xC5C1E000, "ldff1d {z0.d}, p0/z, [x0, z1.d]");
    // Scatter stores.
    check(0xE5418000, "st1w {z0.s}, p0, [x0, z1.s, uxtw]");
    check(0xE456A5CC, "st1b {z12.d}, p1, [z14.d, #0x16]");
    check(0xE44A2137, "stnt1b {z23.s}, p0, [z9.s, x10]");
    // Prefetch (scalar+scalar, gather, contiguous-imm).
    check(0x8404C060, "prfb pldl1keep, p0, [x3, x4]");
    check(0x84622060, "prfh pldl1keep, p0, [x3, z2.s, sxtw #0x1]");
    // LDR / STR predicate + vector.
    check(0x85810943, "ldr p3, [x10, #0xa, mul vl]");
    check(0xE5810943, "str p3, [x10, #0xa, mul vl]");
    check(0x858010A0, "ldr p0, [x5, #0x4, mul vl]");
    // A numeric (target==3) prefetch op with word<4>==0 is still valid.
    check(0x84602066, "prfh #0x6, p0, [x3, z0.s, sxtw #0x1]");
}

// ---------------------------------------------------------------------------
// 1. Scalar+scalar Xm==xzr reserved (the task example words + family).
// ---------------------------------------------------------------------------

#[test]
fn ss_xzr_reserved() {
    // Task examples.
    reserved(0xA47F528D); // ld1b  {z.d}, p/z, [x20, xzr]
    reserved(0xA4FF455D); // ld1h  {z.d}, ..., [x10, xzr, lsl #1]
    reserved(0xA57F528D); // ld1w  {z.d}, ..., [x20, xzr, lsl #2]
    // Across the family: ld1b/st1b/ldnt1/structured/ld1rq/ld1ro + prf ss.
    reserved(0xA41F4000); // ld1b  [x0, xzr]
    reserved(0xE41F4871); // st1b  {z17.b}, p2, [x3, xzr]
    reserved(0xA41FC000); // ldnt1b [x0, xzr]
    reserved(0xA43FC000); // ld2b   [x0, xzr]   (op6 structured ss)
    reserved(0xA43F0000); // ld1rob [x0, xzr]   (op0 ss)
    reserved(0x841FC000); // prfb   [x0, xzr]   (prf ss)
    // The `.q` single-register ss forms (SVE2.1) are reserved with xzr too.
    reserved(0xA51F8000); // ld1w {z.q}, [x0, xzr, lsl #2]
    reserved(0xE51F4000); // st1w {z.q}, [x0, xzr, lsl #2]
    reserved(0xE5DF4000); // st1d {z.q}, [x0, xzr, lsl #3]
}

#[test]
fn ldff1_xzr_still_valid() {
    // First-fault loads DO allow xzr (LLVM renders `[Xn]`); fARM64 renders the
    // explicit `xzr` — bit-exact round-trip either way. Must NOT be rejected.
    assert!(!decode(0xA41F6000, 0, FeatureSet::ALL).is_invalid()); // ldff1b [x0, xzr]
    assert_eq!(encode(&decode(0xA41F6000, 0, FeatureSet::ALL)).unwrap(), 0xA41F6000);
}

// ---------------------------------------------------------------------------
// 2. 32-bit gather: 64-bit-element load into .s reserved.
// ---------------------------------------------------------------------------

#[test]
fn gather32_64bit_element_reserved() {
    // Task examples (ld1d/ldff1d into .s).
    reserved(0x85BBCD88); // ld1d   {z8.s}, p3/z, [z12.s, #0xd8]
    reserved(0x85A66D13); // ldff1d {z.s}, ..., [x8, z6.s, uxtw #3]
    reserved(0x85802000); // ldff1d {z0.s}, ..., [x0, z0.s, uxtw]
    // Signed-word into .s (F4 task example + family).
    reserved(0x85000000); // ld1sw  {z0.s}, p0/z, [x0, z0.s, uxtw]
    // The same forms in the 64-bit (.d) quadrant remain valid.
    assert!(!decode(0xC5C1E000, 0, FeatureSet::ALL).is_invalid()); // ldff1d .d
    assert!(!decode(0x85014000, 0, FeatureSet::ALL).is_invalid()); // ld1w .s (msz2 unsigned, ok)
}

// ---------------------------------------------------------------------------
// 3. Prefetch prfop<4> reserved (task examples) + ss xzr.
// ---------------------------------------------------------------------------

#[test]
fn prefetch_reserved() {
    // Task examples (all have Zt<4>==1).
    reserved(0x84623A9B); // prfh ..., [x20, z2.s, sxtw #1]
    reserved(0x8415E5F1); // prfb ..., [z15.s, #0x15]
    reserved(0x842F577F); // prfw ...
    // prfop<4>==1 across forms.
    reserved(0x84603A90); // prfh gather, Zt=0x10
    reserved(0x85CA0750); // prfb contiguous-imm, Zt<4>=1
    // prfop<4>==0 with a numeric (target==3) op is valid (see valid_forms test).
    assert!(!decode(0x84622066, 0, FeatureSet::ALL).is_invalid());
}

// ---------------------------------------------------------------------------
// 4. Scatter store reserved size/scale forms.
// ---------------------------------------------------------------------------

#[test]
fn scatter_store_reserved() {
    // 32-bit (.s) offset cannot store a dword (st1d msz3 b22==1).
    reserved(0xE5C08000); // st1d {z.s}, [x0, z0.s, uxtw]
    reserved(0xE5C0C000); // st1d {z.s}, [x0, z0.s, sxtw]
    // Byte store cannot scale (st1b msz0, op4/6, b21==1).
    reserved(0xE4208000); // st1b [x0, z0.d, uxtw], scaled
    reserved(0xE420C000); // st1b [x0, z0.d, sxtw], scaled
    // STNT1 with b21==1 (RES0) reserved, and .s dword reserved.
    reserved(0xE5A02000); // stnt1d {z.d}, [z0.d, x0]  (msz3 b22 set / b21 form)
    // Valid scatter neighbors still decode.
    assert!(!decode(0xE5418000, 0, FeatureSet::ALL).is_invalid()); // st1w .s uxtw
    assert!(!decode(0xE4044861, 0, FeatureSet::ALL).is_invalid()); // st1b ss
}

// ---------------------------------------------------------------------------
// 5. LD1RQ/LD1RO RES0 bits, structured-imm RES0, LDR/STR predicate Pt<4>.
// ---------------------------------------------------------------------------

#[test]
fn replicating_and_struct_imm_reserved() {
    // LD1RQ/LD1RO: word<22> RES0.
    reserved(0xA4C00000); // ld1rqh, b22 set
    reserved(0xA4E00000); // ld1roh, b22 set
    // LD1RQ/LD1RO scalar+imm: word<20> RES0.
    reserved(0xA4302000 | (1 << 20)); // ld1rob imm, b20 set
    // Structured load imm form: word<20> RES0.
    reserved(0xA5B0E000 | (1 << 20)); // ld2d imm, b20 set
}

#[test]
fn ldr_str_predicate_reserved() {
    // Task example: ldr p6, [x28, #0x95, mul vl] has Pt<4>==1.
    reserved(0x85921796);
    // STR predicate with Pt<4>==1.
    reserved(0xE5810943 | 0x10);
    reserved(0x85810943 | 0x10);
    // The vector LDR/STR (word<14>==1) uses a full 5-bit Zt — not affected.
    assert!(!decode(0x858050A0, 0, FeatureSet::ALL).is_invalid()); // ldr z0, [x5, #4, mul vl]
}

// ---------------------------------------------------------------------------
// Feature gating: without FEAT_SVE the whole space decodes Invalid.
// ---------------------------------------------------------------------------

#[test]
fn sve_feature_gated() {
    let no_sve = {
        let bit = Feature::Sve as u32;
        FeatureSet {
            features0: FeatureSet::ALL.features0 & !(1u64 << bit),
            features1: FeatureSet::ALL.features1 & !(1u64 << bit),
        }
    };
    assert!(decode(0xA40A4000, 0, no_sve).is_invalid());
    assert!(decode(0x84BB1101, 0, no_sve).is_invalid());
}
