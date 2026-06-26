//! N batch: final micro over-decode close-out across the scalar Advanced-SIMD
//! two-reg-misc family and a handful of reserved SVE/SVE2 fields.
//!
//! Each reserved word below is `<unknown>` in LLVM (`clang` + `llvm-objdump
//! --mattr=+all`); the paired canonical word still decodes + round-trips. Proven
//! 0-regression via pre/post differentials:
//!   * scalar two-reg-misc (0x5E/0x7E): 90 reserved words eliminated in-sweep,
//!     all LLVM-UNDEFINED; 112 LLVM-valid scalar forms retained; the vector
//!     (0x0E/0x4E/0x2E/0x6E) path is byte-for-byte unchanged.
//!   * SVE (0x04/0x05/0x25/0x45): 101 reserved words eliminated in-sweep, all
//!     LLVM-UNDEFINED; 11,715 LLVM-valid words retained.
//!
//! Families:
//!   1. NEON scalar two-reg-misc — only the scalar-allocated opcodes survive;
//!      the vector-only ones (CLS/CLZ/CNT/REV*/SADDLP/XTN/SHLL/MVN/RBIT and the
//!      FRINT*/FABS/FNEG/FSQRT FP rounding ops) and out-of-range sizes (FADDP
//!      `size<1>=1`, FP16 pairwise `size<0>=1`, FCVTXN size!=01) are reserved.
//!   2. SVE2 saturating extract-narrow (SQXTNT/UQXTNT/SQXTUNT, 0x45): the
//!      destination `tsz` must be an exact power of two (`001`/`010`/`100`).
//!   3. SVE small reserved fields: SADDV `.d`, predicate REV/PUNPK `<12:10>`,
//!      PMOV-from-vector `<4>`, BRKAS/BRKBS merge bit, vector INC/DEC-P `.b`,
//!      DUP/MOV-immediate `.b` with `lsl #8`.

#![cfg(feature = "std")]

use fARM64::decode::decode;
use fARM64::{encode, FeatureSet};

/// Decode, re-encode, re-decode; require the word reproduces exactly and the
/// mnemonic/operand-count are stable.
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
// 1. NEON scalar two-reg-misc — reserved opcodes / sizes.
// ===========================================================================

#[test]
fn scalar_misc_vector_only_int_ops_reserved() {
    // Integer ops that exist only in the vector form have no scalar slot:
    //   XTN (op 10010, U=0), CNT (00101), REV64 (00000,U=0)/REV32 (00000,U=1),
    //   SADDLP (00010,U=0)/UADDLP (00010,U=1).
    for bad in [
        0x5E21290Bu32, // xtn b11, h8
        0x5E605A39,    // cnt h25, h17
        0x5E600A4B,    // rev64 (vector-only)
        0x7E600A4B,    // rev32 (vector-only)
        0x5EA02BAA,    // saddlp (vector-only)
        0x7EA02BAA,    // uaddlp (vector-only)
    ] {
        assert!(is_invalid(bad), "{:08X} vector-only int misc should be Invalid", bad);
    }
}

#[test]
fn scalar_misc_fp_rounding_reserved() {
    // The FP rounding / int-rounding ops (FRINTM/FRINTP/FRINTX, FRINT32Z/X) have
    // no scalar form — only the convert-to-int / compare-#0 / reciprocal-estimate
    // ops do.
    for bad in [
        0x5E619ABF_u32, // frintm d31, d21
        0x7E619ABF,     // frintx d31, d21
        0x5EF98909,     // frintp h9, h8  (FP16 scalar misc)
        0x5E61EAAD,     // frint32z d13, d21
        0x7E61EAAD,     // frint32x d13, d21
    ] {
        assert!(is_invalid(bad), "{:08X} scalar FP-rounding misc should be Invalid", bad);
    }
}

#[test]
fn scalar_misc_fcvtxn_size_reserved() {
    // FCVTXN scalar is fixed at size==01 (`s <- d`); any other size is reserved.
    assert!(is_invalid(0x7EA16BC9), "7EA16BC9 fcvtxn size==10 should be Invalid");
    assert!(is_invalid(0x7E216822), "7E216822 fcvtxn size==00 should be Invalid");
    assert!(is_invalid(0x7EE16822), "7EE16822 fcvtxn size==11 should be Invalid");
    // The one allocated size (01) still decodes + round-trips.
    assert_eq!(mnem(0x7E616822), "fcvtxn");
    assert_roundtrip(0x7E616822);
}

#[test]
fn scalar_pairwise_faddp_and_fp16_size_reserved() {
    // FADDP has no min variant: `size<1>` (the max/min selector) must be 0.
    assert!(is_invalid(0x5EB0DAC3), "5EB0DAC3 FP16 FADDP size==10 should be Invalid");
    assert!(is_invalid(0x7EB0DAC3), "7EB0DAC3 S/D FADDP size==10 should be Invalid");
    // FP16 scalar pairwise has no double-precision form (`size<0>` must be 0).
    assert!(is_invalid(0x5E70D822), "5E70D822 FP16 FADDP size==01 should be Invalid");
    // The allocated pairwise forms survive.
    for &(w, m) in &[
        (0x5E30D822u32, "faddp"),   // h, .2h
        (0x7E30D822, "faddp"),      // s, .2s
        (0x7E70D822, "faddp"),      // d, .2d
        (0x5EB0F822, "fminp"),      // FP16 min variant (size==10) still OK
        (0x5EF1B822, "addp"),       // integer ADDP scalar
    ] {
        assert_eq!(mnem(w), m, "{:08X} mnemonic", w);
        assert_roundtrip(w);
    }
}

#[test]
fn scalar_misc_allocated_forms_survive() {
    // A representative spread of the 112 LLVM-valid scalar two-reg-misc forms.
    for &(w, m) in &[
        (0x5E203822u32, "suqadd"), // b
        (0x7E203822, "usqadd"),    // b
        (0x5E207822, "sqabs"),     // b
        (0x7E207822, "sqneg"),     // b
        (0x5E214822, "sqxtn"),     // b <- h
        (0x7E212822, "sqxtun"),    // b <- h
        (0x7E214822, "uqxtn"),     // b <- h
        (0x5EE08822, "cmgt"),      // d, #0
        (0x5EE0B822, "abs"),       // d
        (0x7EE0B822, "neg"),       // d
        (0x5E21A822, "fcvtns"),    // s
        (0x5EA1D822, "frecpe"),    // s
        (0x5EA1F822, "frecpx"),    // s
        (0x5EA0C822, "fcmgt"),     // s, #0.0
        (0x5E79A822, "fcvtns"),    // h (FP16)
        (0x5EF9F822, "frecpx"),    // h (FP16)
        (0x7EF9D822, "frsqrte"),   // h (FP16)
    ] {
        assert!(!is_invalid(w), "{:08X} allocated scalar misc should decode", w);
        assert_eq!(mnem(w), m, "{:08X} mnemonic", w);
        assert_roundtrip(w);
    }
}

#[test]
fn vector_misc_unaffected() {
    // The vector two-reg-misc path keeps every form (XTN/CNT/REV/SADDLP/FRINT*).
    for &(w, m) in &[
        (0x0E212800u32, "xtn"),    // xtn v0.8b, v0.8h
        (0x0E205800, "cnt"),       // cnt v0.8b, v0.8b
        (0x0E200800, "rev64"),     // rev64 v0.8b, v0.8b
        (0x0E202800, "saddlp"),    // saddlp v0.4h, v0.8b
        (0x4E619800, "frintm"),    // frintm v0.2d, v0.2d
        (0x4E61A800, "fcvtns"),    // fcvtns v0.2d, v0.2d
    ] {
        assert!(!is_invalid(w), "{:08X} vector misc should decode", w);
        assert_eq!(mnem(w), m, "{:08X} mnemonic", w);
        assert_roundtrip(w);
    }
}

// ===========================================================================
// 2. SVE2 saturating extract-narrow TOP — `tsz` must be a power of two.
// ===========================================================================

#[test]
fn sve_extract_narrow_tsz_reserved() {
    // The destination element `tsz` (`tszh:tszl`) must be `001`/`010`/`100`; the
    // non-power-of-two patterns are reserved.
    for bad in [
        0x45384F82u32, // uqxtnt with tsz==011
        0x4578561A,    // sqxtunt with tsz==111
        0x45684744,    // sqxtnt with tsz==101
    ] {
        assert!(is_invalid(bad), "{:08X} extract-narrow bad tsz should be Invalid", bad);
    }
    // All three sizes / six ops with a valid tsz still decode + round-trip.
    for &(w, m) in &[
        (0x45284382u32, "sqxtnb"),  // .b <- .h
        (0x45284782, "sqxtnt"),     // .b <- .h
        (0x45284F82, "uqxtnt"),     // .b <- .h
        (0x45304782, "sqxtnt"),     // .h <- .s
        (0x45605782, "sqxtunt"),    // .s <- .d
        (0x45605382, "sqxtunb"),    // .s <- .d
    ] {
        assert_eq!(mnem(w), m, "{:08X} mnemonic", w);
        assert_roundtrip(w);
    }
}

// ===========================================================================
// 3. SVE small reserved fields.
// ===========================================================================

#[test]
fn sve_saddv_d_reserved() {
    // SADDV `.d` (size==11) is reserved; UADDV keeps its `.d` form.
    assert!(is_invalid(0x04C02EF6), "04C02EF6 saddv .d should be Invalid");
    assert!(is_invalid(0x04C03962), "04C03962 saddv .d should be Invalid");
    for &(w, m) in &[
        (0x04003962u32, "saddv"), // .b
        (0x04803962, "saddv"),    // .s
        (0x04C13962, "uaddv"),    // uaddv .d still valid
    ] {
        assert!(!is_invalid(w), "{:08X} should decode", w);
        assert_eq!(mnem(w), m, "{:08X} mnemonic", w);
        assert_roundtrip(w);
    }
}

#[test]
fn sve_pred_rev_punpk_bits_reserved() {
    // Predicate REV / PUNPK are unary and fix `<12:10>=000`.
    assert!(is_invalid(0x05744501), "05744501 rev p.h <12:10>!=0 should be Invalid");
    for &(w, m) in &[
        (0x05744101u32, "rev"),     // rev p1.h, p8.h
        (0x05344101, "rev"),        // rev p1.b, p8.b
        (0x05314101, "punpkhi"),    // punpkhi p1.h, p8.b
        (0x05304101, "punpklo"),    // punpklo p1.h, p8.b
    ] {
        assert_eq!(mnem(w), m, "{:08X} mnemonic", w);
        assert_roundtrip(w);
    }
}

#[test]
fn sve_pmov_from_vector_bit4_reserved() {
    // PMOV-from-vector writes a 4-bit `Pd`; `<4>` must be 0.
    assert!(is_invalid(0x056E38BF), "056E38BF pmov-from-vector <4>=1 should be Invalid");
    // The canonical word (and the to-vector twin) still decode.
    assert_eq!(mnem(0x056E38AF), "pmov");
    assert_roundtrip(0x056E38AF);
    assert_eq!(mnem(0x052A3880), "pmov"); // pmov p0.b, z4
    assert_roundtrip(0x052A3880);
}

#[test]
fn sve_brkas_brkbs_merge_bit_reserved() {
    // The flag-setting BRKAS/BRKBS (`S=1`) are zeroing-only: the merge bit `<4>`
    // must be 0.
    assert!(is_invalid(0x25D05893), "25D05893 brkbs <4>=1 should be Invalid");
    assert!(is_invalid(0x25505893), "25505893 brkas <4>=1 should be Invalid");
    for &(w, m) in &[
        (0x25D05883u32, "brkbs"),  // brkbs p3.b, p6/z, p4.b
        (0x25505883, "brkas"),     // brkas p3.b, p6/z, p4.b
        (0x25905893, "brkb"),      // brkb (non-S) keeps /m (M=1)
        (0x25105893, "brka"),      // brka (non-S) keeps /m (M=1)
    ] {
        assert_eq!(mnem(w), m, "{:08X} mnemonic", w);
        assert_roundtrip(w);
    }
}

#[test]
fn sve_incdecp_vector_byte_reserved() {
    // The vector INC/DEC-P forms operate on `.h`/`.s`/`.d`; `.b` is reserved.
    for bad in [
        0x252980CBu32, // uqincp z11.b, p6
        0x252C805E,    // incp z30.b
        0x2528805E,    // sqincp z30.b
    ] {
        assert!(is_invalid(bad), "{:08X} vector INC/DEC-P .b should be Invalid", bad);
    }
    // The valid vector sizes and all scalar forms (incl. `.b`) survive.
    for &(w, m) in &[
        (0x25A9805Eu32, "uqincp"), // vector .s
        (0x256C805E, "incp"),      // vector .h
        (0x252D8986, "decp"),      // scalar, p.b
        (0x25298986, "uqincp"),    // scalar, p.b
    ] {
        assert!(!is_invalid(w), "{:08X} should decode", w);
        assert_eq!(mnem(w), m, "{:08X} mnemonic", w);
        assert_roundtrip(w);
    }
}

#[test]
fn sve_dup_imm_byte_shift_reserved() {
    // DUP/MOV-immediate with `lsl #8` (`sh==1`) is reserved for `.b` elements.
    assert!(is_invalid(0x2538EFC5), "2538EFC5 mov z.b with lsl #8 should be Invalid");
    for &(w, m) in &[
        (0x2538CFC5u32, "mov"),  // mov z5.b, #0x7e   (no shift)
        (0x2578EFC5, "mov"),     // mov z5.h, #0x7e00 (.h keeps the shift)
    ] {
        assert!(!is_invalid(w), "{:08X} should decode", w);
        assert_eq!(mnem(w), m, "{:08X} mnemonic", w);
        assert_roundtrip(w);
    }
}
