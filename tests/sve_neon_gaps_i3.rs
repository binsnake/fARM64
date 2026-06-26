//! SVE / NEON / FEAT_LRCPC3 gap fills, batch i3: decode + render + bidirectional
//! round-trip + feature gating + reserved-slot Invalid checks.
//!
//! Every canonical word is an LLVM (`clang .inst` + `llvm-objdump --mattr=+all`)
//! oracle encoding. Families covered (each a "LLVM-valid, fARM64-Invalid" gap):
//!
//! 1.  **NEON FP16->FP32 2-way `FDOT`** (`FEAT_F16F32DOT`) — the `.2s/.4s <-
//!     .4h/.8h` vector form (`lo=111111`, `(U,size)=(0,10)`) and the `.2h[idx]`
//!     by-element form (opcode `1001`, `size=01`).
//! 2.  **SVE2.3 `ADDQP` / `ADDSUBP`** (`FEAT_SVE2p3`) — quadword pair add /
//!     add-subtract, sharing the unpredicated-multiply `<12:10>` slot.
//! 3.  **SVE `LUTI6`** (`FEAT_LUT`) — 2-table lookup (`.b` plain / `.h` indexed),
//!     filling the `(<12>,<11>)=(0,1)` LUT sub-slot.
//! 4.  **SVE FP16->FP32 matrix `FMMLA`** (`FEAT_F16F32MM`) — `.s <- .h`, the
//!     `<23:22>=00` slot of the `0x64` `111001` matrix block.
//! 5.  **SVE2.3 2-way `UDOT`/`SDOT`** (`.h <- .b`, `FEAT_SVE2p3`).
//! 6.  **SVE2.2 predicated `SQABS`/`SQNEG` zeroing** (`/z`, `FEAT_SVE2p2`).
//! 7.  **FEAT_CPA `MADPT`/`MLAPT`/`SUBP`** — checked pointer multiply-add (`.d`)
//!     and the predicated subtract pointer.
//! 8.  **SVE `FAMAX`/`FAMIN` predicated** (`FEAT_FAMINMAX`).
//! 9.  **SVE `FRINT32/64 Z/X` merging** (`/m`, `FEAT_SVE2p2`) — the `0x65`
//!     analogues of the `0x64` `/z` forms.
//! 10. **FEAT_LRCPC3 `LDAPP`/`LDAP`/`STLP`** — X-only ordered load/store pair.

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

fn without(fs: FeatureSet, f: Feature) -> FeatureSet {
    let bit = f as u32;
    FeatureSet { features0: fs.features0 & !(1u64 << bit), features1: fs.features1 & !(1u64 << bit) }
}

/// Assert `word` decodes to Invalid once `f` is cleared from the feature set.
#[track_caller]
fn gated_off(word: u32, f: Feature) {
    let insn = decode(word, 0, without(FeatureSet::ALL, f));
    assert!(insn.is_invalid(), "{:08X} still decoded with {:?} cleared", word, f);
}

/// Assert `word` is Invalid (reserved / unallocated) under the full feature set.
#[track_caller]
fn invalid(word: u32) {
    let insn = decode(word, 0, FeatureSet::ALL);
    assert!(insn.is_invalid(), "{:08X} unexpectedly decoded as `{}`", word, text(word));
}

// ---------------------------------------------------------------------------
// 1. NEON FP16->FP32 2-way FDOT.
// ---------------------------------------------------------------------------

#[test]
fn neon_fdot_f16_examples() {
    // Vector (.2s/.4s <- .4h/.8h).
    check(0x0E85FF12, "fdot v18.2s, v24.4h, v5.4h");
    check(0x4E85FF12, "fdot v18.4s, v24.8h, v5.8h");
    // By element (.2h[i], 5-bit Vm, 2-bit index).
    check(0x0F429020, "fdot v0.2s, v1.4h, v2.2h[0]");
    check(0x0F629020, "fdot v0.2s, v1.4h, v2.2h[1]");
    check(0x0F429820, "fdot v0.2s, v1.4h, v2.2h[2]");
    check(0x0F629820, "fdot v0.2s, v1.4h, v2.2h[3]");
    check(0x4F629020, "fdot v0.4s, v1.8h, v2.2h[1]");
    check(0x0F529020, "fdot v0.2s, v1.4h, v18.2h[0]");
    check(0x0F7F9820, "fdot v0.2s, v1.4h, v31.2h[3]");
}

#[test]
fn neon_fdot_f16_gating() {
    gated_off(0x0E85FF12, Feature::F16f32dot);
    gated_off(0x0F429020, Feature::F16f32dot);
}

#[test]
fn neon_fdot_f16_neighbours() {
    // size==00/01 are the FP8 FDOT (FEAT_FP8); size==11 is FMLALB/T — all still
    // decode (untouched). The (0,10) FP8 slot we added must not steal them.
    assert_eq!(decode(0x0E05FF12, 0, FeatureSet::ALL).mnemonic().name(), "fdot");
    assert_eq!(decode(0x0E45FF12, 0, FeatureSet::ALL).mnemonic().name(), "fdot");
}

// ---------------------------------------------------------------------------
// 2. SVE2.3 ADDQP / ADDSUBP.
// ---------------------------------------------------------------------------

#[test]
fn sve_addqp_addsubp_examples() {
    check(0x043779B1, "addqp z17.b, z13.b, z23.b");
    check(0x04E27D59, "addsubp z25.d, z10.d, z2.d");
    check(0x04227820, "addqp z0.b, z1.b, z2.b");
    check(0x04627820, "addqp z0.h, z1.h, z2.h");
    check(0x04A27820, "addqp z0.s, z1.s, z2.s");
    check(0x04E27820, "addqp z0.d, z1.d, z2.d");
    check(0x04227C20, "addsubp z0.b, z1.b, z2.b");
    check(0x04E27C20, "addsubp z0.d, z1.d, z2.d");
}

#[test]
fn sve_addqp_addsubp_gating() {
    gated_off(0x043779B1, Feature::Sve2p3);
    gated_off(0x04E27D59, Feature::Sve2p3);
}

// ---------------------------------------------------------------------------
// 3. SVE LUTI6.
// ---------------------------------------------------------------------------

#[test]
fn sve_luti6_examples() {
    check(0x4526AFD1, "luti6 z17.b, {z30.b, z31.b}, z6");
    check(0x4523AC20, "luti6 z0.b, {z1.b, z2.b}, z3");
    check(0x453FAC1F, "luti6 z31.b, {z0.b, z1.b}, z31");
    check(0x4566AFD1, "luti6 z17.h, {z30.h, z31.h}, z6[0]");
    check(0x45E6AFD1, "luti6 z17.h, {z30.h, z31.h}, z6[1]");
}

#[test]
fn sve_luti6_gating_and_reserved() {
    gated_off(0x4526AFD1, Feature::Lut);
    // `.b` with <23>!=0 is unallocated (LLVM <unknown>).
    invalid(0x45A6AFD1);
}

// ---------------------------------------------------------------------------
// 4. SVE FP16->FP32 matrix FMMLA.
// ---------------------------------------------------------------------------

#[test]
fn sve_fmmla_f16f32_examples() {
    check(0x6430E7F3, "fmmla z19.s, z31.h, z16.h");
    check(0x6422E420, "fmmla z0.s, z1.h, z2.h");
}

#[test]
fn sve_fmmla_f16f32_gating() {
    gated_off(0x6430E7F3, Feature::F16f32mm);
    // The neighbouring BFMMLA (.s<-.h, <23:22>=01) still decodes.
    assert_eq!(decode(0x6470E7F3, 0, FeatureSet::ALL).mnemonic().name(), "bfmmla");
}

// ---------------------------------------------------------------------------
// 5. SVE2.3 2-way UDOT / SDOT (.h <- .b).
// ---------------------------------------------------------------------------

#[test]
fn sve_dot_hb_examples() {
    check(0x44560750, "udot z16.h, z26.b, z22.b");
    check(0x4453012F, "sdot z15.h, z9.b, z19.b");
    check(0x44420420, "udot z0.h, z1.b, z2.b");
    check(0x44420020, "sdot z0.h, z1.b, z2.b");
}

#[test]
fn sve_dot_hb_gating_and_neighbours() {
    gated_off(0x44560750, Feature::Sve2p3);
    // The existing .s<-.b (size=10) and .d<-.h (size=11) dots still decode.
    assert_eq!(decode(0x44820420, 0, FeatureSet::ALL).mnemonic().name(), "udot");
    assert_eq!(decode(0x44C20420, 0, FeatureSet::ALL).mnemonic().name(), "udot");
}

// ---------------------------------------------------------------------------
// 6. SVE2.2 predicated SQABS / SQNEG zeroing (/z).
// ---------------------------------------------------------------------------

#[test]
fn sve_sqabs_sqneg_zeroing_examples() {
    check(0x440ABD9E, "sqabs z30.b, p7/z, z12.b");
    check(0x44CBA737, "sqneg z23.d, p1/z, z25.d");
    check(0x4408A020, "sqabs z0.b, p0/m, z1.b"); // merging sibling still works.
    check(0x440AA020, "sqabs z0.b, p0/z, z1.b");
    check(0x440BA020, "sqneg z0.b, p0/z, z1.b");
}

#[test]
fn sve_sqabs_sqneg_zeroing_gating() {
    gated_off(0x440ABD9E, Feature::Sve2p2);
    gated_off(0x44CBA737, Feature::Sve2p2);
    // Merging forms are NOT gated on SVE2.2 (they are the older SVE2 forms).
    assert!(!decode(0x4408A020, 0, without(FeatureSet::ALL, Feature::Sve2p2)).is_invalid());
}

// ---------------------------------------------------------------------------
// 7. FEAT_CPA MADPT / MLAPT / SUBP.
// ---------------------------------------------------------------------------

#[test]
fn sve_cpa_examples() {
    check(0x44D7DA82, "madpt z2.d, z23.d, z20.d");
    check(0x44CFD317, "mlapt z23.d, z24.d, z15.d");
    check(0x44C1D840, "madpt z0.d, z1.d, z2.d");
    check(0x44C2D020, "mlapt z0.d, z1.d, z2.d");
    check(0x4490A1E9, "subp z9.s, p0/m, z9.s, z15.s");
    check(0x4410A020, "subp z0.b, p0/m, z0.b, z1.b");
    check(0x4450A020, "subp z0.h, p0/m, z0.h, z1.h");
    check(0x44D0A020, "subp z0.d, p0/m, z0.d, z1.d");
}

#[test]
fn sve_cpa_gating() {
    gated_off(0x44D7DA82, Feature::Cpa);
    gated_off(0x44CFD317, Feature::Cpa);
    gated_off(0x4490A1E9, Feature::Cpa);
}

// ---------------------------------------------------------------------------
// 8. SVE FAMAX / FAMIN predicated.
// ---------------------------------------------------------------------------

#[test]
fn sve_famax_famin_examples() {
    check(0x658E8BF9, "famax z25.s, p2/m, z25.s, z31.s");
    check(0x658F96F1, "famin z17.s, p5/m, z17.s, z23.s");
    check(0x654E8020, "famax z0.h, p0/m, z0.h, z1.h");
    check(0x65CE8020, "famax z0.d, p0/m, z0.d, z1.d");
    check(0x654F8020, "famin z0.h, p0/m, z0.h, z1.h");
    check(0x65CF8020, "famin z0.d, p0/m, z0.d, z1.d");
}

#[test]
fn sve_famax_famin_gating() {
    gated_off(0x658E8BF9, Feature::Faminmax);
    gated_off(0x658F96F1, Feature::Faminmax);
}

// ---------------------------------------------------------------------------
// 9. SVE FRINT32/64 Z/X merging (/m).
// ---------------------------------------------------------------------------

#[test]
fn sve_frint_merging_examples() {
    check(0x6516A0FB, "frint64z z27.d, p0/m, z7.d");
    check(0x6510A020, "frint32z z0.s, p0/m, z1.s");
    check(0x6511A020, "frint32x z0.s, p0/m, z1.s");
    check(0x6514A020, "frint64z z0.s, p0/m, z1.s");
    check(0x6515A020, "frint64x z0.s, p0/m, z1.s");
    check(0x6512A020, "frint32z z0.d, p0/m, z1.d");
}

#[test]
fn sve_frint_merging_gating() {
    gated_off(0x6516A0FB, Feature::Sve2p2);
    gated_off(0x6510A020, Feature::Sve2p2);
    // The /z analogue (top byte 0x64) still decodes.
    assert!(text(0x641DC0FB).starts_with("frint64z"));
}

// ---------------------------------------------------------------------------
// 10. FEAT_LRCPC3 LDAPP / LDAP / STLP.
// ---------------------------------------------------------------------------

#[test]
fn lrcpc3_ldapp_ldap_stlp_examples() {
    check(0xD9527BB3, "ldapp x19, x18, [x29]");
    check(0xD94A5981, "ldap x1, x10, [x12]");
    check(0xD9125A54, "stlp x20, x18, [x18]");
    check(0xD9417840, "ldapp x0, x1, [x2]"); // opc2=0111, L=1.
    check(0xD9415840, "ldap x0, x1, [x2]"); // opc2=0101, L=1.
    check(0xD9015840, "stlp x0, x1, [x2]"); // opc2=0101, L=0.
}

#[test]
fn lrcpc3_ldapp_ldap_stlp_gating_and_reserved() {
    gated_off(0xD9527BB3, Feature::Rcpc3);
    gated_off(0xD9125A54, Feature::Rcpc3);
    // The legacy LDIAPP/STILP siblings (opc2=000x) still decode.
    assert_eq!(decode(0xD9411840, 0, FeatureSet::ALL).mnemonic().name(), "ldiapp");
    assert_eq!(decode(0xD9011840, 0, FeatureSet::ALL).mnemonic().name(), "stilp");
    // X-only: the 32-bit-size (sz=10/W) opc2=0101/0111 forms are unallocated.
    invalid(0x99415840); // would-be ldap with sz=10 (W)
    invalid(0x99415840 ^ (1 << 22)); // would-be stlp (L=0) with sz=10 (W)
    invalid(0x99417840); // would-be ldapp with sz=10 (W)
    // No STLPP: opc2=0111 with L=0 is unallocated even at 64-bit width.
    invalid(0xD9017840);
}
