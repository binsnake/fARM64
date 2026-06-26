//! R2 batch — SME `ZERO` (ZA tile mask / ZT0 / ZA array) + SME2 `MOVT` (move
//! to/from the ZT0 lookup table), the last two GAP mnemonics in the `0xC0`
//! region.
//!
//! All forms live in the SME `110` quadrant (top byte `0xC0`, `word<24> == 0`,
//! `word<23> == 0`, `word<19> == 1`) and were previously fARM64-Invalid while
//! LLVM-valid. Every canonical word here is an LLVM `clang .inst` +
//! `llvm-objdump --mattr=+all` (LLVM 21) oracle encoding; a focused
//! decode-render + encode-round-trip differential over the whole dispatch shell
//! confirms decode matches LLVM (modulo the documented intentional `mova`/`mov`
//! alias and hex/decimal radix differences on the *neighbouring* base forms),
//! with **0 over-decode** and **0 round-trip failure**.
//!
//! GAPS added:
//!   1. **ZERO (ZA tile mask)** — `zero { za0.d, .. }` / `{ za }` / `{}`. The
//!      8-bit mask `word<7:0>` tiles the destination at the largest cleanly
//!      tiling element width (`.h` `0x55`/`0xAA`, `.s` `0x11<<i`, `.d` `1<<i`).
//!      `word<23:16> == 0x08`, `word<15:8> == 0`. FEAT_SME.
//!   2. **ZERO (ZT0)** — `zero { zt0 }`, exactly `0xC0480001`. FEAT_SME2.
//!   3. **ZERO (ZA array)** — `zero za.d[<Ws>, ..]`, `.d`-only, with the
//!      `(span, vgxN)` shapes selected by `word<17:16:15>`. FEAT_SME2.
//!   4. **MOVT** — three directions: `movt zt0[idx, mul vl], Zt`
//!      (`word<23:16> == 0x4F`), `movt zt0[off], Xt` (`0x4E`) and
//!      `movt Xt, zt0[off]` (`0x4C`), byte offset = `word<14:12> * 8`. FEAT_SME2.

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
// 1. ZERO (ZA tile mask) — the brace-list destination.
// ===========================================================================

#[test]
fn zero_mask_examples() {
    // The two extremes and a couple of `.d` lists.
    check(0xC0080000, "zero {}");
    check(0xC00800FF, "zero { za }");
    check(0xC0080001, "zero { za0.d }");
    check(0xC0080080, "zero { za7.d }");
    check(0xC0080021, "zero { za0.d, za5.d }");
    check(0xC008000F, "zero { za0.d, za1.d, za2.d, za3.d }");
    // `.s` tiles (`0x11 << i`) and `.h` tiles (`0x55`/`0xAA`) take precedence
    // over the `.d` decomposition when the mask tiles cleanly.
    check(0xC0080011, "zero { za0.s }");
    check(0xC0080022, "zero { za1.s }");
    check(0xC0080044, "zero { za2.s }");
    check(0xC0080088, "zero { za3.s }");
    check(0xC0080099, "zero { za0.s, za3.s }");
    check(0xC0080077, "zero { za0.s, za1.s, za2.s }");
    check(0xC0080055, "zero { za0.h }");
    check(0xC00800AA, "zero { za1.h }");
    // A mask that does not tile at `.s`/`.h` falls back to `.d`.
    check(0xC0080056, "zero { za1.d, za2.d, za4.d, za6.d }");
}

#[test]
fn zero_mask_reserved_fields() {
    // `word<15:8>` and `word<17:16>` are RES0 for the tile-mask form.
    assert!(is_invalid(0xC0080100), "ZERO mask <8>=1 should be Invalid");
    assert!(is_invalid(0xC0090000), "ZERO mask <16>=1 should be Invalid");
    assert!(is_invalid(0xC00A0000), "ZERO mask <17>=1 should be Invalid");
    // `word<21:20>` RES0.
    assert!(is_invalid(0xC0180000), "ZERO mask <20>=1 should be Invalid");
}

#[test]
fn zero_mask_needs_sme() {
    // The tile-mask form is base FEAT_SME.
    let insn = decode(0xC0080021, 0, without(FeatureSet::ALL, Feature::Sme));
    assert!(insn.is_invalid(), "ZERO mask must gate on FEAT_SME");
}

// ===========================================================================
// 2. ZERO (ZT0) — `zero { zt0 }` (exactly one word).
// ===========================================================================

#[test]
fn zero_zt0_example() {
    check(0xC0480001, "zero { zt0 }");
    // Every neighbour is UNDEFINED.
    assert!(is_invalid(0xC0480000), "ZT0 zero <0>=0 should be Invalid");
    assert!(is_invalid(0xC0480003), "ZT0 zero <1>=1 should be Invalid");
    assert!(is_invalid(0xC04800FF), "ZT0 zero mask bits should be Invalid");
}

#[test]
fn zero_zt0_needs_sme2() {
    let insn = decode(0xC0480001, 0, without(FeatureSet::ALL, Feature::Sme2));
    assert!(insn.is_invalid(), "ZERO {{ zt0 }} must gate on FEAT_SME2");
}

// ===========================================================================
// 3. ZERO (ZA array) — `.d`-only slice group with span / vgxN shapes.
// ===========================================================================

#[test]
fn zero_array_examples() {
    // Single offset, vgx2 / vgx4.
    check(0xC00C0000, "zero za.d[w8, 0, vgx2]");
    check(0xC00C6007, "zero za.d[w11, 7, vgx2]");
    check(0xC00E0000, "zero za.d[w8, 0, vgx4]");
    // 2-slice / 4-slice ranges (no vgxN). fARM64 renders the range decimal
    // (an intentional radix difference from LLVM's `0x0:0x1`).
    check(0xC00C8000, "zero za.d[w8, 0:1]");
    check(0xC00C8003, "zero za.d[w8, 6:7]");
    check(0xC00E8000, "zero za.d[w8, 0:3]");
    check(0xC00E8001, "zero za.d[w8, 4:7]");
    // Ranges with vgxN.
    check(0xC00D0000, "zero za.d[w8, 0:1, vgx2]");
    check(0xC00D0001, "zero za.d[w8, 2:3, vgx2]");
    check(0xC00D8000, "zero za.d[w8, 0:1, vgx4]");
    check(0xC00F0000, "zero za.d[w8, 0:3, vgx2]");
    check(0xC00F8000, "zero za.d[w8, 0:3, vgx4]");
}

#[test]
fn zero_array_reserved_fields() {
    // `word<12:3>` are RES0.
    assert!(is_invalid(0xC00C0008), "ZERO array <3>=1 should be Invalid");
    assert!(is_invalid(0xC00C1000), "ZERO array <12>=1 should be Invalid");
    // `word<22:20>` RES0.
    assert!(is_invalid(0xC01C0000), "ZERO array <20>=1 should be Invalid");
    // Out-of-range field for the narrow shapes: span-4 vgx4 allows fld 0..=1,
    // so fld==2 (`word<2:0>==2`) is UNDEFINED.
    assert!(is_invalid(0xC00F8002), "ZERO array span4/vgx4 fld=2 should be Invalid");
    // span-2 vgx2 allows fld 0..=3, so fld==4 is UNDEFINED.
    assert!(is_invalid(0xC00D0004), "ZERO array span2/vgx2 fld=4 should be Invalid");
}

#[test]
fn zero_array_needs_sme2() {
    let insn = decode(0xC00C0000, 0, without(FeatureSet::ALL, Feature::Sme2));
    assert!(insn.is_invalid(), "ZERO za.d[..] must gate on FEAT_SME2");
}

// ===========================================================================
// 4. MOVT — three ZT0-move directions.
// ===========================================================================

#[test]
fn movt_z_examples() {
    // `movt zt0[idx, mul vl], Zt`: idx 0 elides the bracket (LLVM `movt zt0, z`).
    check(0xC04F03E0, "movt zt0, z0");
    check(0xC04F13E0, "movt zt0[1, mul vl], z0");
    check(0xC04F23E3, "movt zt0[2, mul vl], z3");
    check(0xC04F33FF, "movt zt0[3, mul vl], z31");
}

#[test]
fn movt_gp_store_examples() {
    // `movt zt0[off], Xt`: byte off = word<14:12>*8 (0,8,..,56).
    check(0xC04E03E0, "movt zt0[0], x0");
    check(0xC04E13E1, "movt zt0[8], x1");
    check(0xC04E73E0, "movt zt0[56], x0");
    check(0xC04E03FF, "movt zt0[0], xzr");
}

#[test]
fn movt_gp_load_examples() {
    // `movt Xt, zt0[off]`.
    check(0xC04C03E0, "movt x0, zt0[0]");
    check(0xC04C13E0, "movt x0, zt0[8]");
    check(0xC04C73E5, "movt x5, zt0[56]");
    check(0xC04C03FF, "movt xzr, zt0[0]");
}

#[test]
fn movt_reserved_fields() {
    // `word<11:5> == 0x1F` is fixed for every MOVT skeleton.
    assert!(is_invalid(0xC04F01E0), "MOVT <11:5>!=0x1F should be Invalid");
    assert!(is_invalid(0xC04F07E0), "MOVT <11:5>!=0x1F should be Invalid");
    // Z-form: `word<15:14>` RES0 (index is only 2 bits).
    assert!(is_invalid(0xC04F43E0), "MOVT Z <14>=1 should be Invalid");
    assert!(is_invalid(0xC04F83E0), "MOVT Z <15>=1 should be Invalid");
    // GP forms: `word<15>` RES0.
    assert!(is_invalid(0xC04E83E0), "MOVT GP-store <15>=1 should be Invalid");
    assert!(is_invalid(0xC04C83E0), "MOVT GP-load <15>=1 should be Invalid");
    // `word<17:16> == 01` is unallocated (no `0x4D` MOVT row).
    assert!(is_invalid(0xC04D03E0), "MOVT <17:16>=01 should be Invalid");
}

#[test]
fn movt_needs_sme2() {
    for w in [0xC04F03E0u32, 0xC04E03E0, 0xC04C03E0] {
        let insn = decode(w, 0, without(FeatureSet::ALL, Feature::Sme2));
        assert!(insn.is_invalid(), "{:08X} MOVT must gate on FEAT_SME2", w);
    }
}

// ===========================================================================
// 5. Exhaustive round-trip sweeps over the allocated encodings.
// ===========================================================================

#[test]
fn zero_mask_roundtrip_all_256() {
    // Every 8-bit mask is a valid `ZERO` (the empty mask renders `{}`).
    for m in 0u32..=0xFF {
        let word = 0xC0080000 | m;
        let insn = decode(word, 0, FeatureSet::ALL);
        assert!(!insn.is_invalid(), "{:08X} (mask {:#04x}) should decode", word, m);
        assert_eq!(encode(&insn).unwrap(), word, "mask {:#04x} round-trip", m);
    }
}

#[test]
fn movt_roundtrip_sweep() {
    // Z-form: idx 0..=3, Zt 0..=31.
    for idx in 0u32..4 {
        for zt in 0u32..32 {
            let word = 0xC04F03E0 | (idx << 12) | zt;
            let insn = decode(word, 0, FeatureSet::ALL);
            assert!(!insn.is_invalid(), "{:08X} MOVT-Z should decode", word);
            assert_eq!(encode(&insn).unwrap(), word, "{:08X} MOVT-Z round-trip", word);
        }
    }
    // GP forms: off field 0..=7 (byte 0..56), Xt 0..=31, both directions.
    for base in [0xC04E03E0u32, 0xC04C03E0] {
        for off in 0u32..8 {
            for xt in 0u32..32 {
                let word = base | (off << 12) | xt;
                let insn = decode(word, 0, FeatureSet::ALL);
                assert!(!insn.is_invalid(), "{:08X} MOVT-GP should decode", word);
                assert_eq!(encode(&insn).unwrap(), word, "{:08X} MOVT-GP round-trip", word);
            }
        }
    }
}

#[test]
fn zero_array_roundtrip_sweep() {
    // Sweep the allocated `(g, v, r, fld)` shapes with their valid `fld` ranges,
    // across all four `Ws`. 160 valid words total.
    let shapes: &[(u32, u32)] = &[
        // (word<17:15> as a 3-bit value, max_field)
        (0b000, 7), // span1 vgx2
        (0b100, 7), // span1 vgx4
        (0b001, 7), // span2
        (0b101, 3), // span4
        (0b010, 3), // span2 vgx2
        (0b011, 3), // span2 vgx4
        (0b110, 1), // span4 vgx2
        (0b111, 1), // span4 vgx4
    ];
    let mut count = 0;
    for &(sel, maxf) in shapes {
        for ws in 0u32..4 {
            for fld in 0..=maxf {
                let word = 0xC00C0000 | (sel << 15) | (ws << 13) | fld;
                let insn = decode(word, 0, FeatureSet::ALL);
                assert!(!insn.is_invalid(), "{:08X} ZERO-array should decode", word);
                assert_eq!(encode(&insn).unwrap(), word, "{:08X} ZERO-array round-trip", word);
                count += 1;
            }
        }
    }
    assert_eq!(count, 160, "expected 160 valid ZERO za.d words");
}
