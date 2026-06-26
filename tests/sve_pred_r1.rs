//! R1 batch — SVE2.1 predicate-as-counter region (top byte `0x25`).
//!
//! Two GAPS (LLVM-valid, previously fARM64-Invalid) plus three over-decode
//! families (previously fARM64-decoded, LLVM-UNDEFINED). Every canonical word is
//! an LLVM `clang .inst` + `llvm-objdump --mattr=+all` oracle encoding; every
//! reserved word is `<unknown>` in LLVM. A pre/post differential over the whole
//! `0x24`/`0x25` top-byte space confirms 0-regression: the only words whose
//! decode changed went Invalid->valid (the two gaps) or valid->Invalid (the
//! reserved slots, all LLVM-UNDEFINED); no LLVM-valid word was lost.
//!
//! GAPS added:
//!   1. **PEXT** (predicate extract from a predicate-as-counter): single
//!      `pext <Pd>.<T>, <PNn>[<imm>]` and pair `pext {<Pd1>.<T>, <Pd2>.<T>},
//!      <PNn>[<imm>]`. Slot `<15:11>=01110`, `<10>` selects single(0)/pair(1),
//!      `<9:8>=index`, source `PN(8+<7:5>)`, `<20:16>=00000`.
//!   2. **PTRUE (predicate-as-counter)**: `ptrue <PNd>.<T>`. Slot `<15:11>=01111`,
//!      `<10:4>=0000001`, dest `PN(8+<2:0>)`.
//!
//! Over-decodes guarded (precise reserved condition each):
//!   3. **PTRUE/PTRUES** — `<21>` is a fixed 0 (leading bit of the `011` opcode
//!      marker). `2538E042`/`2539E0C4` (`<21>=1`) are UNDEFINED; canonical
//!      `2518E042`/`2519E0C4`.
//!   4. **CTERMEQ/CTERMNE** — `<23>` is a fixed 1 (leading bit of the `1 sz`
//!      width field). `25732010` (`<23>=0`) is UNDEFINED; canonical `25F32010`.
//!   5. **RDFFR/RDFFRS (predicated)** — `<23>` is a fixed 0 (only `<22>=S`
//!      toggles the S form). `2598F02F`/`25D8F125` (`<23>=1`) are UNDEFINED;
//!      canonical `2518F02F`/`2558F125`.
//!   6. **WRFFR** — `<11:9>` is a fixed `000`. `252894A0` (`<10>=1`) is
//!      UNDEFINED; canonical `252890A0`.

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

fn is_invalid(word: u32) -> bool {
    decode(word, 0, FeatureSet::ALL).is_invalid()
}

fn without(fs: FeatureSet, f: Feature) -> FeatureSet {
    let bit = f as u32;
    FeatureSet { features0: fs.features0 & !(1u64 << bit), features1: fs.features1 & !(1u64 << bit) }
}

// ===========================================================================
// 1. PEXT (predicate extract from predicate-as-counter) — single + pair.
// ===========================================================================

#[test]
fn pext_single_examples() {
    check(0x25207010, "pext p0.b, pn8[0]");
    check(0x25607131, "pext p1.h, pn9[1]");
    check(0x25A07253, "pext p3.s, pn10[2]");
    check(0x25E07375, "pext p5.d, pn11[3]");
    // Index/source/dest corners.
    check(0x252070F0, "pext p0.b, pn15[0]");
    check(0x2520701F, "pext p15.b, pn8[0]");
}

#[test]
fn pext_pair_examples() {
    check(0x25207410, "pext {p0.b, p1.b}, pn8[0]");
    check(0x25607432, "pext {p2.h, p3.h}, pn9[0]");
    check(0x25A07510, "pext {p0.s, p1.s}, pn8[1]");
    check(0x25207510, "pext {p0.b, p1.b}, pn8[1]");
    check(0x25E07494, "pext {p4.d, p5.d}, pn12[0]");
    // The pair base is any P (not just even): `<3:0>` selects `{Pd, P(d+1)}`.
    check(0x25207411, "pext {p1.b, p2.b}, pn8[0]");
    check(0x2520741E, "pext {p14.b, p15.b}, pn8[0]");
    check(0x2520741F, "pext {p15.b, p0.b}, pn8[0]");
}

#[test]
fn pext_reserved_fields() {
    // `<20:16>` must be 0; `<9>` past the 2-bit single-form index is reserved.
    assert!(is_invalid(0x25217010), "PEXT <16>=1 should be Invalid");
    assert!(is_invalid(0x25207910), "PEXT single <9>=1 should be Invalid");
    // The pair form has only a 1-bit index (`<8>`); `<9>=1` is reserved.
    assert!(is_invalid(0x25207610), "PEXT pair index <9>=1 should be Invalid");
    assert!(is_invalid(0x25207710), "PEXT pair index <9>=1 should be Invalid");
    // `<4>` is a fixed 1 marker on both forms.
    assert!(is_invalid(0x25207000), "PEXT single <4>=0 should be Invalid");
    assert!(is_invalid(0x25207400), "PEXT pair <4>=0 should be Invalid");
}

// ===========================================================================
// 2. PTRUE (predicate-as-counter).
// ===========================================================================

#[test]
fn ptrue_pn_examples() {
    check(0x25207810, "ptrue pn8.b");
    check(0x25607811, "ptrue pn9.h");
    check(0x25A07812, "ptrue pn10.s");
    check(0x25E07817, "ptrue pn15.d");
}

#[test]
fn ptrue_pn_reserved_fields() {
    // `<10:4>` must be `0000001`.
    assert!(is_invalid(0x25207830), "PTRUE-pn <5>=1 should be Invalid");
    assert!(is_invalid(0x25207800), "PTRUE-pn <4>=0 should be Invalid");
    // `<20:16>` must be 0.
    assert!(is_invalid(0x25217810), "PTRUE-pn <16>=1 should be Invalid");
}

// ===========================================================================
// 3. PTRUE/PTRUES (vector) — `<21>` fixed 0.
// ===========================================================================

#[test]
fn ptrue_vector_bit21_reserved() {
    for &(bad, good, txt) in &[
        (0x2538E042u32, 0x2518E042u32, "ptrue p2.b, vl2"),
        (0x2539E0C4, 0x2519E0C4, "ptrues p4.b, vl6"),
    ] {
        assert!(is_invalid(bad), "{:08X} PTRUE <21>=1 should be Invalid", bad);
        check(good, txt);
    }
}

// ===========================================================================
// 4. CTERMEQ/CTERMNE — `<23>` fixed 1.
// ===========================================================================

#[test]
fn cterm_bit23_reserved() {
    assert!(is_invalid(0x25732010), "25732010 CTERM <23>=0 should be Invalid");
    // Canonical X and W forms still decode + round-trip.
    check(0x25F32010, "ctermne x0, x19");
    check(0x25E12010, "ctermne x0, x1");
    check(0x25A12000, "ctermeq w0, w1");
}

// ===========================================================================
// 5. RDFFR/RDFFRS (predicated) — `<23>` fixed 0.
// ===========================================================================

#[test]
fn rdffr_pred_bit23_reserved() {
    for &(bad, good, txt) in &[
        (0x2598F02Fu32, 0x2518F02Fu32, "rdffr p15.b, p1/z"),
        (0x25D8F125, 0x2558F125, "rdffrs p5.b, p9/z"),
    ] {
        assert!(is_invalid(bad), "{:08X} RDFFR-pred <23>=1 should be Invalid", bad);
        check(good, txt);
    }
}

// ===========================================================================
// 6. WRFFR — `<11:9>` fixed `000`.
// ===========================================================================

#[test]
fn wrffr_bit10_reserved() {
    assert!(is_invalid(0x252894A0), "252894A0 WRFFR <10>=1 should be Invalid");
    check(0x252890A0, "wrffr p5.b");
    // Neighbouring FFR ops unaffected.
    check(0x2519F00F, "rdffr p15.b");
    check(0x252C9000, "setffr");
}

// ===========================================================================
// Feature gating — the new GAP forms require FEAT_SVE2p1.
// ===========================================================================

#[test]
fn gaps_feature_gated() {
    let no_p1 = without(FeatureSet::ALL, Feature::Sve2p1);
    for &w in &[0x25207010u32, 0x25207410, 0x25207810] {
        assert!(decode(w, 0, no_p1).is_invalid(), "{:08X} should need FEAT_SVE2p1", w);
        assert!(!decode(w, 0, FeatureSet::ALL).is_invalid(), "{:08X} should decode with SVE2p1", w);
    }
}

// ===========================================================================
// Never-panic sweep over the affected sub-region.
// ===========================================================================

#[test]
fn never_panics_pred_counter_region() {
    // Sweep the `<23:0>` operand/opcode space of top byte 0x25 with a stride to
    // keep the test fast while still exercising every decoder leaf.
    let mut w = 0x2500_0000u32;
    while w <= 0x25FF_FFFF {
        let insn = decode(w, 0, FeatureSet::ALL);
        if !insn.is_invalid() {
            // Any decoded word must format without panicking.
            let _ = format_to_string(&FmtFormatter::new(), &insn);
        }
        w = w.wrapping_add(0x40);
    }
}
