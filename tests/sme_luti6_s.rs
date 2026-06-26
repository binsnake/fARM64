//! S batch — SME2 `LUTI6` ZT0 lookup with a **register-group table source**
//! (`luti6 { Zd.b - Zd+3.b }, zt0, { Zn - Zn+2 }`), the last remaining LUTI6 gap
//! variant in the SME `0xC0` region.
//!
//! This is the sibling of the single-vector `LUTI6` (`luti6 Zd.b, zt0, Zn`, Q3)
//! and the multi-vector indexed `LUTI6`/`LUTI4` ZT0 forms already in
//! `src/decode/sme/sme_lut.rs`. Unlike those, the table operand here is a
//! 3-register consecutive group `{ Zn - Zn+2 }` (no element suffix, no ZT0
//! element index); the destination is a 4-register `.b` group, consecutive or
//! strided step-4.
//!
//! Field layout (LLVM 21 oracle): top byte `0xC0`, `word<23>=1`, `word<22>=0`
//! (multi), `word<21>=0`, `word<20>=St` (0 consecutive / 1 strided), `word<19>=1`,
//! `word<18:16>=010`, `word<15:10>=000000`. The table-source base is `Zn =
//! word<9:7>` (z0..z7, `word<6:5>` RES0); the destination base is `word<4:0>`
//! (consecutive: multiple of 4; strided: `word<3:2>==0`, bases z0..z3 / z16..z19).
//! `.b`-only. FEAT_LUT.
//!
//! Every canonical word is an LLVM `clang .inst` + `llvm-objdump --mattr=+all`
//! oracle encoding. A decode-render assertion plus a bit-exact encode round-trip
//! (and an exhaustive 128-word round-trip sweep) confirm both directions; a
//! feature-gating check confirms it is dark without FEAT_LUT.

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
// 1. Canonical examples — consecutive and strided destinations.
// ===========================================================================

#[test]
fn luti6_group_consecutive_examples() {
    // word<20>=0: consecutive 4-register `.b` destination rendered as a range;
    // the table source is the 3-register consecutive group `{ Zn - Zn+2 }`.
    check(0xC08A0000, "luti6 { z0.b - z3.b }, zt0, { z0 - z2 }");
    check(0xC08A0004, "luti6 { z4.b - z7.b }, zt0, { z0 - z2 }");
    check(0xC08A0008, "luti6 { z8.b - z11.b }, zt0, { z0 - z2 }");
    check(0xC08A0010, "luti6 { z16.b - z19.b }, zt0, { z0 - z2 }");
    check(0xC08A001C, "luti6 { z28.b - z31.b }, zt0, { z0 - z2 }");
    // Source base z1..z7 (Zn = word<9:7>).
    check(0xC08A0080, "luti6 { z0.b - z3.b }, zt0, { z1 - z3 }");
    check(0xC08A0100, "luti6 { z0.b - z3.b }, zt0, { z2 - z4 }");
    check(0xC08A0380, "luti6 { z0.b - z3.b }, zt0, { z7 - z9 }");
}

#[test]
fn luti6_group_strided_examples() {
    // word<20>=1: strided step-4 destination `{ Zd, Zd+4, Zd+8, Zd+12 }`
    // rendered as a comma list. Bases z0..z3 (word<4>=0) / z16..z19 (word<4>=1).
    check(0xC09A0000, "luti6 { z0.b, z4.b, z8.b, z12.b }, zt0, { z0 - z2 }");
    check(0xC09A0001, "luti6 { z1.b, z5.b, z9.b, z13.b }, zt0, { z0 - z2 }");
    check(0xC09A0003, "luti6 { z3.b, z7.b, z11.b, z15.b }, zt0, { z0 - z2 }");
    check(0xC09A0010, "luti6 { z16.b, z20.b, z24.b, z28.b }, zt0, { z0 - z2 }");
    check(0xC09A0013, "luti6 { z19.b, z23.b, z27.b, z31.b }, zt0, { z0 - z2 }");
    check(0xC09A0380, "luti6 { z0.b, z4.b, z8.b, z12.b }, zt0, { z7 - z9 }");
}

// ===========================================================================
// 2. Reserved-field / out-of-range guards (LLVM UNDEFINED -> fARM64 Invalid).
// ===========================================================================

#[test]
fn luti6_group_reserved_fields() {
    // Consecutive dest base must be a multiple of 4 (word<1:0> RES0).
    assert!(is_invalid(0xC08A0001), "consec dest base z1 should be Invalid");
    assert!(is_invalid(0xC08A0002), "consec dest base z2 should be Invalid");
    // Strided dest base window: word<3:2> RES0 (only z0..z3 / z16..z19).
    assert!(is_invalid(0xC09A0004), "strided dest <2>=1 should be Invalid");
    assert!(is_invalid(0xC09A0008), "strided dest <3>=1 should be Invalid");
    // Source 3-register group: word<6:5> RES0 (Zn lives only in word<9:7>).
    assert!(is_invalid(0xC08A0020), "source <5>=1 should be Invalid");
    assert!(is_invalid(0xC08A0040), "source <6>=1 should be Invalid");
    // word<15:10> RES0 — any set bit there is unallocated for this form.
    assert!(is_invalid(0xC08A0400), "word<10>=1 should be Invalid");
    assert!(is_invalid(0xC08A1000), "word<12>=1 (non-.b size) should be Invalid");
    assert!(is_invalid(0xC08A8000), "word<15>=1 should be Invalid");
    // Single-vector marker (word<22>=1) is a different (LUTI2/4/6-single) form,
    // never this register-group LUTI6 — the 0x8A+word<22> word is UNDEFINED.
    assert!(is_invalid(0xC0CA0400), "word<22>=1 here should be Invalid");
}

// ===========================================================================
// 3. Feature gating — dark without FEAT_LUT.
// ===========================================================================

#[test]
fn luti6_group_needs_lut() {
    for w in [0xC08A0000u32, 0xC09A0000] {
        let insn = decode(w, 0, without(FeatureSet::ALL, Feature::Lut));
        assert!(insn.is_invalid(), "{:08X} LUTI6 group must gate on FEAT_LUT", w);
    }
}

// ===========================================================================
// 4. Exhaustive round-trip over the 128 allocated encodings.
// ===========================================================================

#[test]
fn luti6_group_roundtrip_all_128() {
    let mut count = 0;
    for st in 0u32..2 {
        // Destination bases: consecutive = {0,4,8,12,16,20,24,28};
        // strided = {0,1,2,3,16,17,18,19}.
        let dest_bases: &[u32] = if st == 0 {
            &[0, 4, 8, 12, 16, 20, 24, 28]
        } else {
            &[0, 1, 2, 3, 16, 17, 18, 19]
        };
        for &zd in dest_bases {
            for zn in 0u32..8 {
                let word = 0xC08A0000 | (st << 20) | (zn << 7) | zd;
                let insn = decode(word, 0, FeatureSet::ALL);
                assert!(!insn.is_invalid(), "{:08X} LUTI6 group should decode", word);
                assert_eq!(encode(&insn).unwrap(), word, "{:08X} round-trip", word);
                count += 1;
            }
        }
    }
    assert_eq!(count, 128, "expected 128 valid LUTI6 register-group words");
}
