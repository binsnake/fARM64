//! M batch: final over-decode tail close-out across six reserved-encoding
//! families. Each family pairs previously-mis-decoded words (now `Invalid`) with
//! the canonical LLVM-valid encoding (still decodes + round-trips). Every
//! reserved word below is `<unknown>` in LLVM (`clang`/`llvm-objdump
//! --mattr=+all`); no LLVM-valid word is newly rejected — proven 0-regression
//! via a pre/post differential over the touched regions (474 words eliminated
//! in-sweep, all LLVM-UNDEFINED; 521 LLVM-valid words preserved; 0 corruption,
//! 0 mnemonic drift).
//!
//! Families:
//!   1. FEAT_THE / FEAT_LSE128 RCW-pair / LSE128-pair ops: the size field
//!      `word<31:30>` allocates only `00` (RCW* / LSE128) and `01` (RCWS*);
//!      sizes `10`/`11` are reserved.
//!   2. SVE DUP-immediate (`MOV <Zd>.<T>, #imm`): `word<18:17>` is a fixed `00`.
//!   3. SVE arithmetic-immediate (ADD/SUB/...): the `lsl #8` shift (`<13>=1`) is
//!      reserved for the `.b` element size (`size==00`).
//!   4. SVE unpredicated MOVPRFX: `word<20:16>` is a fixed `00000`.
//!   5. NEON FP16 two-register-misc: `word<22>` is a fixed `1`, and the
//!      integer-rounding (`FRINT32/64Z/X`) and reciprocal-estimate
//!      (`URECPE`/`URSQRTE`) ops have no half-precision form.
//!   6. SVE PMOV (to vector, `D=<16>=1`): `word<9>` is a fixed `0` (the source
//!      predicate uses only `<8:5>`).
//!   7. FEAT_LS64 64-byte ops (LD64B/ST64B/ST64BV/ST64BV0): the data register
//!      `Xt` must be even and in `[x0, x22]`.

#![cfg(feature = "std")]

use fARM64::decode::decode;
use fARM64::{encode, Feature, FeatureSet};

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
// 1. RCW-pair / LSE128-pair — size field word<31:30> must be 00 or 01.
// ===========================================================================

#[test]
fn rcw_pair_size_reserved() {
    // For every op2==00 pair op, size 10 (`0x99..` prefix) is reserved; the
    // canonical word uses size 00 (RCW* / LSE128) or 01 (RCWS*).
    for &(bad, good) in &[
        (0x99B1B02Du32, 0x19B1B02Du32), // rcwsetpa  x13, x17, [x1]
        (0x9931A02D, 0x1931A02D),       // rcwswpp   x13, x17, [x1]
        (0x9931902D, 0x1931902D),       // rcwclrp   x13, x17, [x1]
        (0x9931802D, 0x1931802D),       // swpp      x13, x17, [x1]
        (0x9931102D, 0x1931102D),       // ldclrp    x13, x17, [x1]
        (0x9931302D, 0x1931302D),       // ldsetp    x13, x17, [x1]
    ] {
        assert!(is_invalid(bad), "{:08X} RCW-pair size==10 should be Invalid", bad);
        assert!(!is_invalid(good), "{:08X} canonical RCW-pair should decode", good);
        assert_roundtrip(good);
    }
}

#[test]
fn rcw_pair_rcws_size01_preserved() {
    // The RCWS* (size 01) and the canonical size-00 forms still decode.
    for &w in &[
        0x59B1B02Du32, // rcwssetpa x13, x17, [x1]
        0x5931A02D,    // rcwsswpp  x13, x17, [x1]
        0x19B1B02D,    // rcwsetpa
    ] {
        assert!(!is_invalid(w), "{:08X} RCWS*/RCW* should decode", w);
        assert_roundtrip(w);
    }
}

// ===========================================================================
// 2. SVE DUP-immediate — word<18:17> must be 00.
// ===========================================================================

#[test]
fn sve_dup_imm_reserved() {
    // `MOV <Zd>.<T>, #imm` fixes <18:17>=00; flipping either bit is reserved.
    for &(bad, good) in &[
        (0x25BCD880u32, 0x25B8D880u32), // mov z0.s, #-60   (<18>=1 bad)
        (0x25BAD880, 0x25B8D880),       // (<17>=1 bad)
        (0x25BED880, 0x25B8D880),       // (<18:17>=11 bad)
    ] {
        assert!(is_invalid(bad), "{:08X} DUP-imm <18:17>!=00 should be Invalid", bad);
        assert!(!is_invalid(good), "{:08X} canonical DUP-imm should decode", good);
        assert_eq!(mnem(good), "mov");
        assert_roundtrip(good);
    }
}

// ===========================================================================
// 3. SVE arithmetic-immediate — `.b` + `lsl #8` shift is reserved.
// ===========================================================================

#[test]
fn sve_arith_imm_b_shift_reserved() {
    // size==00 (`.b`) with sh==1 (`<13>=1`) cannot hold a shifted-by-8 value.
    for &(bad, good) in &[
        (0x2521E415u32, 0x2521C415u32), // sub z21.b, z21.b, #imm  (sh bad -> no-shift good)
        (0x2520E415, 0x2520C415),       // add z21.b, z21.b, #imm
        (0x2527E415, 0x2527C415),       // uqsub z21.b
    ] {
        assert!(is_invalid(bad), "{:08X} arith-imm .b+shift should be Invalid", bad);
        assert!(!is_invalid(good), "{:08X} canonical .b no-shift arith-imm should decode", good);
        assert_roundtrip(good);
    }
}

#[test]
fn sve_arith_imm_shift_other_sizes_preserved() {
    // The `.h`/`.s`/`.d` element sizes accept the `lsl #8` shift.
    for &w in &[
        0x2561E415u32, // sub z21.h, z21.h, #0x2000
        0x25A1E415,    // sub z21.s, z21.s, #0x2000
        0x25E1E415,    // sub z21.d, z21.d, #0x2000
    ] {
        assert!(!is_invalid(w), "{:08X} shifted .h/.s/.d arith-imm should decode", w);
        assert_roundtrip(w);
    }
}

// ===========================================================================
// 4. SVE unpredicated MOVPRFX — word<20:16> must be 00000.
// ===========================================================================

#[test]
fn sve_movprfx_reserved() {
    for &(bad, good) in &[
        (0x0425BFA0u32, 0x0420BFA0u32), // movprfx z0, z29  (<20:16>=00101 bad)
        (0x0421BFA0, 0x0420BFA0),       // (<16>=1 bad)
        (0x043FBFA0, 0x0420BFA0),       // (<20:16>=11111 bad)
    ] {
        assert!(is_invalid(bad), "{:08X} MOVPRFX <20:16>!=0 should be Invalid", bad);
        assert!(!is_invalid(good), "{:08X} canonical MOVPRFX should decode", good);
        assert_eq!(mnem(good), "movprfx");
        assert_roundtrip(good);
    }
}

// ===========================================================================
// 5. NEON FP16 two-reg-misc — word<22> must be 1; FRINT32/64 + URECPE/URSQRTE
//    have no FP16 form.
// ===========================================================================

#[test]
fn fp16_two_reg_misc_sz_reserved() {
    // word<22>==0 is the SP/DP slot, reserved in the FP16 class.
    for &(bad, good) in &[
        (0x0E39AA68u32, 0x0E79AA68u32), // fcvtns v8.4h, v19.4h
        (0x2E39AA68, 0x2E79AA68),       // fcvtnu v8.4h, v19.4h
        (0x0EB8FA68, 0x0EF8FA68),       // fabs   v8.4h, v19.4h
    ] {
        assert!(is_invalid(bad), "{:08X} FP16-misc <22>==0 should be Invalid", bad);
        assert!(!is_invalid(good), "{:08X} canonical FP16-misc should decode", good);
        assert_roundtrip(good);
    }
}

#[test]
fn fp16_two_reg_misc_no_fp16_opcodes_reserved() {
    // FRINT32Z/FRINT64Z/FRINT32X/FRINT64X (opcode 1111x, a==0) and URECPE/URSQRTE
    // (opcode 11100, a==1) have no FP16 form even with <22>==1.
    for &bad in &[
        0x0E79EA68u32, // frint32z (a=0, op 11110)
        0x0E79FA68,    // frint64z (a=0, op 11111)
        0x2E79EA68,    // frint32x (a=0, op 11110)
        0x2E79FA68,    // frint64x (a=0, op 11111)
        0x0EF9CA68,    // urecpe   (a=1, op 11100)
        0x2EF9CA68,    // ursqrte  (a=1, op 11100)
    ] {
        assert!(is_invalid(bad), "{:08X} FP16-misc with no FP16 form should be Invalid", bad);
    }
}

#[test]
fn fp16_two_reg_misc_valid_set_preserved() {
    // A spread of genuinely-FP16 ops across U/a still decode + round-trip.
    for &w in &[
        0x0E798A68u32, // frintn v8.4h
        0x0E79BA68,    // fcvtms v8.4h
        0x0E79DA68,    // scvtf  v8.4h
        0x0EF8CA68,    // fcmgt  v8.4h, #0.0
        0x0EF9DA68,    // frecpe v8.4h
        0x2EF8FA68,    // fneg   v8.4h
        0x2EF9FA68,    // fsqrt  v8.4h
    ] {
        assert!(!is_invalid(w), "{:08X} valid FP16-misc should decode", w);
        assert_roundtrip(w);
    }
}

// ===========================================================================
// 6. SVE PMOV — to-vector direction (D==1) requires word<9>==0.
// ===========================================================================

#[test]
fn sve_pmov_to_vector_bit9_reserved() {
    for &(bad, good) in &[
        (0x05ED3AF7u32, 0x05ED38F7u32), // pmov z23[6], p7.d  (<9>=1 bad)
        (0x056D3A65, 0x056D3865),       // pmov z5[2], p3.s   (<9>=1 bad)
    ] {
        assert!(is_invalid(bad), "{:08X} PMOV-to-vector <9>=1 should be Invalid", bad);
        assert!(!is_invalid(good), "{:08X} canonical PMOV should decode", good);
        assert_eq!(mnem(good), "pmov");
        assert_roundtrip(good);
    }
}

#[test]
fn sve_pmov_from_vector_bit9_preserved() {
    // The from-vector direction (D==0) uses the full 5-bit Zn field, so word<9>
    // is part of the register and both values are valid.
    for &w in &[
        0x05EC3AE7u32, // pmov p7.d, z23[6]  (<9>=1, but D==0 -> Zn bit)
        0x05EC38E7,    // pmov p7.d, z7[6]
    ] {
        assert!(!is_invalid(w), "{:08X} PMOV-from-vector should decode", w);
        assert_roundtrip(w);
    }
}

// ===========================================================================
// 7. FEAT_LS64 — Xt data register must be even and in [x0, x22].
// ===========================================================================

#[test]
fn ls64_rt_reserved() {
    for &bad in &[
        0xF83DA0BAu32, // st64bv0 ..., x26 (Rt=26 > 22)
        0xF83AA0A1,    // st64bv0 Rt=1 (odd)
        0xF83AA0B8,    // st64bv0 Rt=24 (even but > 22)
        0xF83FD0B8,    // ld64b   Rt=24
        0xF83F90B8,    // st64b   Rt=24
        0xF83AB0B8,    // st64bv  Rt=24
    ] {
        assert!(is_invalid(bad), "{:08X} LS64 Rt invalid should be Invalid", bad);
    }
}

#[test]
fn ls64_valid_rt_preserved() {
    for &w in &[
        0xF83AA0A0u32, // st64bv0 x26, x0, [x5]
        0xF83AA0B6,    // st64bv0 x26, x22, [x5]   (Rt=22, max valid)
        0xF83FD0A0,    // ld64b   x0, [x5]
        0xF83F90A0,    // st64b   x0, [x5]
        0xF83AB0A0,    // st64bv  x26, x0, [x5]
    ] {
        assert!(!is_invalid(w), "{:08X} valid LS64 should decode", w);
        assert_roundtrip(w);
    }
}

#[test]
fn ls64_gated_by_feature() {
    // Without FEAT_LS64 the 64-byte ops are not admitted (and are not LSE).
    let no_ls64 = FeatureSet::BASE.with(Feature::Lse);
    for &w in &[0xF83AA0A0u32, 0xF83FD0A0, 0xF83F90A0, 0xF83AB0A0] {
        assert!(decode(w, 0, no_ls64).is_invalid(), "{:08X} must be gated by FEAT_LS64", w);
    }
}
