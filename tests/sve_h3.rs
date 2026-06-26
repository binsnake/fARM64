//! SVE area extensions, group H3: decode + render + bidirectional round-trip.
//!
//! Families covered (every canonical word is an LLVM `clang .inst` +
//! `llvm-objdump --mattr=+all` oracle encoding):
//!
//! 1. **SVE2.1 quadword (`.q`) SINGLE-register contiguous load/store** — `LD1W`/
//!    `LD1D`/`ST1W`/`ST1D` rendering `{ <Zt>.Q }`, both scalar+scalar and
//!    scalar+imm (`MUL VL`). These complete the prior `.q` batch and fix the
//!    `op=1`/`b20=1` imm slot that legacy contiguous decode mis-read as `LD1RQW`.
//! 2. **FEAT_SVE_B16B16 non-widening BFloat16 arithmetic** — predicated
//!    `BFADD`/`BFSUB`/`BFMLA`/`BFMLS`/`BFMAX`/.. and unpredicated
//!    `BFADD`/`BFSUB`/`BFMUL`, all sharing the `size==00` slot of the FP encodings,
//!    plus `BFCLAMP` (and the SVE2.1 `FCLAMP`).
//! 3. **PSEL** (predicate select) — replaces the predicate-`DUP` over-decode that
//!    shadowed the whole PSEL slot.
//! 4. **FP8 widening MLAL vector forms** — `FMLALB/T` (`.h<-.b`) and
//!    `FMLALL{BB,BT,TB,TT}` (`.s<-.b`); the FP8 / FP16 / BF16 matrix
//!    `FMMLA`/`BFMMLA` (`.h`); and the SVE `FDOT` family that the FP32 `FMLALB/T`
//!    / `BFDOT` over-decoders previously shadowed (the bit-23 disambiguation fix).
//! 5. **Smaller additions** — `BFMLSLB/T`, `SABAL`/`UABAL`, `FRINT32/64 Z/X` (`/z`
//!    zeroing) and `LASTP`/`FIRSTP`.

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
fn check(word: u32, expected: &str) {
    let insn = decode(word, 0, FeatureSet::ALL);
    assert!(!insn.is_invalid(), "{:08X} decoded Invalid (want `{}`)", word, expected);
    assert_eq!(norm(&text(word)), norm(expected), "{:08X} disasm mismatch", word);
    let enc = encode(&insn)
        .unwrap_or_else(|e| panic!("{:08X} ({}) encode error {:?}", word, insn.mnemonic().name(), e));
    assert_eq!(enc, word, "{:08X} round-trip produced {:08X}", word, enc);
}

fn without(fs: FeatureSet, f: Feature) -> FeatureSet {
    let bit = f as u32;
    FeatureSet { features0: fs.features0 & !(1u64 << bit), features1: fs.features1 & !(1u64 << bit) }
}

// ---------------------------------------------------------------------------
// 1. Quadword single-register load/store.
// ---------------------------------------------------------------------------

#[test]
fn quadword_single_examples() {
    check(0xA5049695, "ld1w {z21.q}, p5/z, [x20, x4, lsl #0x2]");
    check(0xA591921A, "ld1d {z26.q}, p4/z, [x16, x17, lsl #0x3]");
    check(0xE5114F5C, "st1w {z28.q}, p3, [x26, x17, lsl #0x2]");
    check(0xE5C4FAB2, "st1d {z18.q}, p6, [x21, #0x4, mul vl]");
    // Previously mis-decoded as `ld1rqw`; the imm `.q` form (op=1, b20=1).
    check(0xA51E22EA, "ld1w {z10.q}, p0/z, [x23, #-2, mul vl]");
}

#[test]
fn quadword_single_no_longer_ld1rq() {
    // The `.q` imm slot must decode as `ld1w {.q}`, NOT `ld1rqw {.s}`.
    let insn = decode(0xA51E22EA, 0, FeatureSet::ALL);
    assert_eq!(insn.mnemonic().name(), "ld1w");
    assert!(text(0xA51E22EA).contains(".q"));
    // A genuine LD1RQW (different bit-20) still decodes as ld1rqw.
    assert_eq!(decode(0xA5043695, 0, FeatureSet::ALL).mnemonic().name(), "ld1rqw");
    // Structured LD3Q / LD4Q still decode (b21==1 guard).
    assert_eq!(decode(0xA5218000, 0, FeatureSet::ALL).mnemonic().name(), "ld3q");
    assert_eq!(decode(0xA5A18000, 0, FeatureSet::ALL).mnemonic().name(), "ld4q");
}

#[test]
fn quadword_single_feature_gated() {
    let no_p1 = without(FeatureSet::ALL, Feature::Sve2p1);
    assert!(decode(0xA5049695, 0, no_p1).is_invalid());
}

// ---------------------------------------------------------------------------
// 2. BFloat16 predicated / unpredicated arithmetic + clamp.
// ---------------------------------------------------------------------------

#[test]
fn bf16_arith_examples() {
    check(0x653C1046, "bfmla z6.h, p4/m, z2.h, z28.h");
    check(0x6527354D, "bfmls z13.h, p5/m, z10.h, z7.h");
    check(0x651E0823, "bfmul z3.h, z1.h, z30.h");
    check(0x65000000, "bfadd z0.h, z0.h, z0.h");
    check(0x65019074, "bfsub z20.h, p4/m, z20.h, z3.h");
    check(0x64232434, "bfclamp z20.h, z1.h, z3.h");
    // FCLAMP shares the BFCLAMP encoding (size != 00).
    check(0x64EB2507, "fclamp z7.d, z8.d, z11.d");
}

#[test]
fn bf16_predicated_binary_full_set() {
    // <18:16> selects bfadd/bfsub/bfmul/bfmaxnm/bfminnm/bfmax/bfmin (size==00).
    check(0x65009074, "bfadd z20.h, p4/m, z20.h, z3.h");
    check(0x65049074, "bfmaxnm z20.h, p4/m, z20.h, z3.h");
    check(0x65059074, "bfminnm z20.h, p4/m, z20.h, z3.h");
    check(0x65069074, "bfmax z20.h, p4/m, z20.h, z3.h");
    check(0x65079074, "bfmin z20.h, p4/m, z20.h, z3.h");
}

#[test]
fn bf16_feature_gated() {
    let no_b16 = without(FeatureSet::ALL, Feature::SveB16b16);
    assert!(decode(0x653C1046, 0, no_b16).is_invalid()); // bfmla
    assert!(decode(0x65000000, 0, no_b16).is_invalid()); // bfadd
    assert!(decode(0x64232434, 0, no_b16).is_invalid()); // bfclamp
    // FCLAMP (FEAT_SVE2p1) is unaffected by dropping SVE_B16B16.
    assert!(!decode(0x64EB2507, 0, no_b16).is_invalid());
}

// ---------------------------------------------------------------------------
// 3. PSEL.
// ---------------------------------------------------------------------------

#[test]
fn psel_examples() {
    check(0x25FC7463, "psel p3, p13, p3.b[w12, #0xf]");
    check(0x25244000, "psel p0, p0, p0.b[w12]");
    check(0x25E04000, "psel p0, p0, p0.d[w12, #0x1]");
    check(0x252B4000, "psel p0, p0, p0.h[w15]");
}

#[test]
fn psel_not_dup_overdecode() {
    // The slot is PSEL, never a predicate DUP.
    assert_eq!(decode(0x25FC7463, 0, FeatureSet::ALL).mnemonic().name(), "psel");
    // The all-zero `tsz` (no element marker) is reserved.
    assert!(decode(0x25204000, 0, FeatureSet::ALL).is_invalid());
    // `tsz == 10000` (index bit set, no element marker) is reserved.
    assert!(decode(0x25A34886, 0, FeatureSet::ALL).is_invalid());
}

// ---------------------------------------------------------------------------
// 4. FP8 vector MLAL / MMLA + FDOT (over-decode fix).
// ---------------------------------------------------------------------------

#[test]
fn fp8_vector_mlal_examples() {
    check(0x6427BAC9, "fmlalltt z9.s, z22.b, z7.b");
    check(0x6423A9B0, "fmlalltb z16.s, z13.b, z3.b");
    check(0x643A9A99, "fmlallbt z25.s, z20.b, z26.b");
    check(0x64368980, "fmlallbb z0.s, z12.b, z22.b");
    check(0x64AF9A3D, "fmlalt z29.h, z17.b, z15.b");
    check(0x64AB8924, "fmlalb z4.h, z9.b, z11.b");
}

#[test]
fn fp8_bf16_matrix_examples() {
    check(0x646AE249, "fmmla z9.h, z18.b, z10.b");
    check(0x64F5E2A5, "bfmmla z5.h, z21.h, z21.h");
}

#[test]
fn fdot_overdecode_fixed() {
    // These were mis-decoded as fmlalt/fmlalb/bfdot/bfmlalt (bit23 ignored);
    // they are all `fdot` per LLVM.
    check(0x642B4666, "fdot z6.h, z19.b, z3.b[2]");
    check(0x64284045, "fdot z5.s, z2.h, z0.h[1]");
    check(0x64634593, "fdot z19.s, z12.b, z3.b[0]");
    check(0x64698763, "fdot z3.s, z27.b, z9.b");
    for &w in &[0x642B4666u32, 0x64284045, 0x64634593, 0x64698763] {
        assert_eq!(decode(w, 0, FeatureSet::ALL).mnemonic().name(), "fdot");
    }
}

#[test]
fn fdot_all_widths() {
    // .s<-.h vector/indexed (SVE2.1), .h<-.b, .s<-.b (FP8).
    check(0x64228020, "fdot z0.s, z1.h, z2.h");
    check(0x64224020, "fdot z0.s, z1.h, z2.h[0]");
    check(0x64228420, "fdot z0.h, z1.b, z2.b");
    check(0x64628420, "fdot z0.s, z1.b, z2.b");
}

#[test]
fn fp8_vector_feature_gated() {
    let no_fp8 = without(FeatureSet::ALL, Feature::Fp8);
    assert!(decode(0x6427BAC9, 0, no_fp8).is_invalid()); // fmlalltt
    assert!(decode(0x64AB8924, 0, no_fp8).is_invalid()); // fmlalb
}

// ---------------------------------------------------------------------------
// 5. Smaller additions.
// ---------------------------------------------------------------------------

#[test]
fn bfmlsl_examples() {
    check(0x64F8634A, "bfmlslb z10.s, z26.h, z0.h[6]");
    check(0x64F96E42, "bfmlslt z2.s, z18.h, z1.h[7]");
    check(0x64E2A020, "bfmlslb z0.s, z1.h, z2.h");
    check(0x64E2A420, "bfmlslt z0.s, z1.h, z2.h");
}

#[test]
fn abal_examples() {
    check(0x44D9D5AB, "sabal z11.d, z13.s, z25.s");
    check(0x444ADEFD, "uabal z29.h, z23.b, z10.b");
}

#[test]
fn frint_zeroing_examples() {
    check(0x641C9276, "frint32z z22.s, p4/z, z19.s");
    check(0x641CA000, "frint32x z0.s, p0/z, z0.s");
    check(0x641D8000, "frint64z z0.s, p0/z, z0.s");
    check(0x641DA000, "frint64x z0.s, p0/z, z0.s");
    check(0x641CC000, "frint32z z0.d, p0/z, z0.d");
}

#[test]
fn frint_zeroing_not_scvtf() {
    // The size != 00 slot is scvtf/ucvtf, not frint32/64 (the over-decode guard).
    assert_ne!(decode(0x645CD7FB, 0, FeatureSet::ALL).mnemonic().name(), "frint32z");
}

#[test]
fn lastp_firstp_examples() {
    check(0x2522B97F, "lastp xzr, p14, p11.b");
    check(0x25A1A157, "firstp x23, p8, p10.s");
    check(0x25228000, "lastp x0, p0, p0.b");
    check(0x25218000, "firstp x0, p0, p0.b");
    // Pg<3>==0 case that previously over-decoded as uqincp.
    check(0x2521817D, "firstp x29, p0, p11.b");
}

#[test]
fn lastp_firstp_not_incdec() {
    // The shared INC/DEC-by-predicate-count slot must yield lastp/firstp.
    assert_eq!(decode(0x2521817D, 0, FeatureSet::ALL).mnemonic().name(), "firstp");
    // A genuine UQINCP (op field 01001, not 00001) still decodes as uqincp.
    assert_eq!(decode(0x25298C00, 0, FeatureSet::ALL).mnemonic().name(), "uqincp");
}

// ---------------------------------------------------------------------------
// Exhaustive semantic round-trip sweeps over the sub-spaces.
// ---------------------------------------------------------------------------

/// Decode, re-encode, re-decode; require a stable mnemonic + operand count.
fn rt_stable(word: u32) {
    let insn = decode(word, 0, FeatureSet::ALL);
    if insn.is_invalid() {
        return;
    }
    let enc = match encode(&insn) {
        Ok(e) => e,
        Err(_) => return,
    };
    let insn2 = decode(enc, 0, FeatureSet::ALL);
    assert_eq!(insn.mnemonic(), insn2.mnemonic(), "{:08X} mnemonic drift -> {:08X}", word, enc);
    assert_eq!(insn.op_count(), insn2.op_count(), "{:08X} operand-count drift", word);
}

#[test]
fn sweep_quadword_single() {
    // 0xa5 / 0xe5 contiguous quadrant: many `.q` single-register encodings.
    let mut x: u32 = 0xCAFE;
    for _ in 0..20000 {
        x = x.wrapping_mul(1103515245).wrapping_add(12345);
        let top = if x & 1 == 0 { 0xa5 } else { 0xe5 };
        rt_stable((top << 24) | (x & 0x00ff_ffff));
    }
}

#[test]
fn sweep_fp_64_65() {
    // 0x64 / 0x65 SVE FP quadrant: BF16 arith, FDOT, FP8 MLAL, FMMLA, FRINT.
    let mut x: u32 = 0xBEEF;
    for _ in 0..30000 {
        x = x.wrapping_mul(1103515245).wrapping_add(12345);
        let top = if x & 1 == 0 { 0x64 } else { 0x65 };
        rt_stable((top << 24) | (x & 0x00ff_ffff));
    }
}

#[test]
fn sweep_pred_25() {
    // 0x25 predicate quadrant: PSEL, LASTP, FIRSTP.
    let mut x: u32 = 0xF00D;
    for _ in 0..20000 {
        x = x.wrapping_mul(1103515245).wrapping_add(12345);
        rt_stable((0x25 << 24) | (x & 0x00ff_ffff));
    }
}
