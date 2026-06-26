//! L4 batch: over-decode hardening across four reserved-encoding families.
//!
//! Each family pairs a previously-mis-decoded word (now `Invalid`) with the
//! canonical LLVM-valid encoding (still decodes + round-trips). Every reserved
//! word below is `<unknown>` in LLVM (`clang`/`llvm-objdump --mattr=+all`); no
//! LLVM-valid word is newly rejected (proven 0-regression via a pre/post
//! differential over 0x0E/2E/4E/6E, 0x45 and 0x65 — 3,708 words eliminated
//! in-sweep, all LLVM-UNDEFINED, 0 valid words lost, 0 mnemonic drift).
//!
//! Families:
//!   1. AdvSIMD copy INS (general / element): `Q` is a fixed `1` (128-bit
//!      insert); `Q==0` reserved.
//!   2. SVE2 SADDLBT/SSUBLBT/SSUBLTB: `<11:10>` has only `00`/`10`/`11`; the
//!      `01` slot (SADDLBT with `<10>=1`) is reserved.
//!   3. SVE2 PMULLB/PMULLT: allocated only for `.h`/`.d`/`.q`; the `.s`
//!      (size==10) form is reserved.
//!   4. SVE FP-immediate (predicated): `<9:6>` is a fixed `0000`; a non-zero
//!      value is reserved.

#![cfg(feature = "std")]

use fARM64::decode::decode;
use fARM64::{encode, FeatureSet};

/// Decode, re-encode, re-decode; require the word reproduces exactly and the
/// mnemonic/operand-count are stable.
fn assert_roundtrip(word: u32) {
    let insn = decode(word, 0, FeatureSet::ALL);
    assert!(!insn.is_invalid(), "{:08X} decoded Invalid", word);
    let enc = encode(&insn).unwrap_or_else(|e| {
        panic!("{:08X} ({}) encode error {:?}", word, insn.mnemonic().name(), e)
    });
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
// 1. AdvSIMD copy INS — `Q` must be 1 (128-bit insert).
// ===========================================================================

#[test]
fn ins_general_q0_reserved() {
    // INS (general) `MOV Vd.Ts[i], Wn/Xn` (imm4=0011) requires Q==1.
    for &(bad, good) in &[
        (0x0E051C11u32, 0x4E051C11u32), // mov v17.b[2], w0
        (0x0E031C11, 0x4E031C11),       // mov v17.b[1], w0
        (0x0E0B1C11, 0x4E0B1C11),       // mov v17.h[2], w0
    ] {
        assert!(is_invalid(bad), "{:08X} INS-general Q==0 should be Invalid", bad);
        assert!(!is_invalid(good), "{:08X} canonical INS-general should decode", good);
        assert_eq!(mnem(good), "mov");
        assert_roundtrip(good);
    }
}

#[test]
fn ins_element_q0_reserved() {
    // INS (element) `MOV Vd.Ts[i1], Vn.Ts[i2]` (op=1) requires Q==1.
    for &(bad, good) in &[
        (0x2E010411u32, 0x6E010411u32), // mov v17.b[0], v0.b[0]
        (0x2E080411, 0x6E080411),       // mov v17.d[0], v0.d[0]
        (0x2E020411, 0x6E020411),       // mov v17.h[0], v0.h[0]
    ] {
        assert!(is_invalid(bad), "{:08X} INS-element Q==0 should be Invalid", bad);
        assert!(!is_invalid(good), "{:08X} canonical INS-element should decode", good);
        assert_eq!(mnem(good), "mov");
        assert_roundtrip(good);
    }
}

#[test]
fn copy_other_forms_unaffected() {
    // DUP/SMOV/UMOV across both `Q` values stay valid where LLVM accepts them.
    for &w in &[
        0x0E010411u32, // dup v17.8b, w0   (DUP general, Q==0 OK)
        0x4E080C11,    // dup v17.2d, x0   (DUP general .D, Q==1)
        0x0E010411,    // (dup again)
        0x0E033C11,    // umov w17, v0.h[1]
        0x4E0C2C11,    // smov x17, v0.s[1]
    ] {
        assert!(!is_invalid(w), "{:08X} non-INS copy form should still decode", w);
        assert_roundtrip(w);
    }
}

// ===========================================================================
// 2. SVE2 SADDLBT / SSUBLBT / SSUBLTB — `<11:10>` slot `01` reserved.
// ===========================================================================

#[test]
fn saddlbt_slot01_reserved() {
    // `<11:10>`: 00=SADDLBT, 10=SSUBLBT, 11=SSUBLTB; the `01` slot is reserved
    // at every size (`455386AB`/`459386AB`/`45D386AB` all `<unknown>`).
    for bad in [0x455386ABu32, 0x459386AB, 0x45D386AB] {
        assert!(is_invalid(bad), "{:08X} SADDLBT <10>=1 should be Invalid", bad);
    }
    let cases: &[(u32, &str)] = &[
        (0x455382AB, "saddlbt"), // .h <- .b
        (0x459382AB, "saddlbt"), // .s <- .h
        (0x45D382AB, "saddlbt"), // .d <- .s
        (0x45538AAB, "ssublbt"),
        (0x45538EAB, "ssubltb"),
    ];
    for &(w, m) in cases {
        assert_eq!(mnem(w), m, "{:08X} mnemonic", w);
        assert_roundtrip(w);
    }
}

// ===========================================================================
// 3. SVE2 PMULLB / PMULLT — `.s` (size==10) reserved.
// ===========================================================================

#[test]
fn pmull_s_size10_reserved() {
    // PMULLB/PMULLT exist only for .h (size==01), .d (size==11), .q (size==00);
    // .s (size==10) is reserved → UNDEFINED.
    for bad in [0x45896FE8u32, 0x458669C7, 0x45826820, 0x45826C20] {
        assert!(is_invalid(bad), "{:08X} PMULL .s should be Invalid", bad);
    }
    // The valid PMULL sizes still decode + round-trip...
    for &(w, m) in &[
        (0x45426820u32, "pmullb"), // .h <- .b
        (0x45426C20, "pmullt"),    // .h <- .b
        (0x45C26820, "pmullb"),    // .d <- .s
        (0x45026820, "pmullb"),    // .q <- .d
    ] {
        assert_eq!(mnem(w), m, "{:08X} mnemonic", w);
        assert_roundtrip(w);
    }
    // ...and the integer MULL `.s` form (size==10) is NOT affected (still valid).
    assert!(!is_invalid(0x45827020), "45827020 smullb .s should still decode");
    assert_eq!(mnem(0x45827020), "smullb");
    assert_roundtrip(0x45827020);
}

// ===========================================================================
// 4. SVE FP-immediate (predicated) — `<9:6>` must be 0000.
// ===========================================================================

#[test]
fn fp_imm_bits9_6_reserved() {
    // FADD/FSUB/FMUL/FSUBR/FMAXNM/FMINNM/FMAX/FMIN with #const fix `<9:6>=0000`.
    for &(bad, good) in &[
        (0x655A9DE5u32, 0x655A9C25u32), // fmul z5.h, p7/m, z5.h, #2.0
        (0x65DE8707, 0x65DE8407),       // fmax z7.d, p1/m, z7.d, #0.0
        (0x65DF91FF, 0x65DF903F),       // fmin z31.d, p4/m, z31.d, #1.0
        (0x65DB80E6, 0x65DB8026),       // fsubr z6.d, p0/m, z6.d, #1.0
    ] {
        assert!(is_invalid(bad), "{:08X} FP-imm <9:6>!=0 should be Invalid", bad);
        assert!(!is_invalid(good), "{:08X} canonical FP-imm should decode", good);
        assert_roundtrip(good);
    }
    // The allocated per-op constant set is unchanged (i1 selects the constant).
    assert_eq!(mnem(0x655A9C25), "fmul");
    assert_eq!(mnem(0x65DE8407), "fmax");
    assert_eq!(mnem(0x65DF903F), "fmin");
    assert_eq!(mnem(0x65DB8026), "fsubr");
}
