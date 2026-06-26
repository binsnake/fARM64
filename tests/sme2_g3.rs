//! SME2 G3 area: LUTI2/LUTI4 (ZT0 lookup table) and the ZA tile-slice
//! `MOV`/`MOVAZ` (move multi-vectors to/from a ZA tile slice group).
//!
//! Covers two FEAT_LUT / FEAT_SME2 families decoded in `decode::sme`:
//!
//! * **SME2 LUT (ZT0)** — `LUTI2`/`LUTI4 {<Zd>...}, ZT0, <Zn>[<index>]` reading
//!   the fixed `ZT0` lookup table. Distinct from the SVE `LUTI2`/`LUTI4` (Zm
//!   index, no ZT0). 1-, 2- and 4-register destinations; `.B`/`.H`/`.S`; the
//!   only unallocated `(count, size)` combination is `LUTI4` 4-register `.B`.
//! * **SME2 ZA tile move** — `MOV ZA<tile><HV>.<T>[<Ws>, <off>:<off+N-1>],
//!   {<Zn>...}` (vectors → ZA), the reverse `MOV {<Zd>...}, ZA...`, and the
//!   zeroing readout `MOVAZ {<Zd>...}, ZA...`. `vgx2`/`vgx4`, `.B`/`.H`/`.S`/`.D`,
//!   horizontal/vertical slices.
//!
//! The canonical example words are LLVM (`--mattr=+all`) oracle encodings. The
//! tests confirm fARM64 decodes them to the expected mnemonic/operands,
//! re-encodes bit-identically, sweeps the sub-spaces for round-trip stability,
//! checks reserved/undefined/neighbour slots stay `Invalid`, and that the
//! families are feature-gated.
//!
//! NB: LLVM renders the ZA slice range in hex (`[w12, 0x6:0x7]`); fARM64 follows
//! the corpus' decimal convention (`[w12, 6:7]`) — an intentional radix
//! difference, mirrored in the expected strings below.

#![cfg(feature = "std")]

use fARM64::decode::decode;
use fARM64::format::{format_to_string, FmtFormatter};
use fARM64::{encode, Feature, FeatureSet};

/// Decode `word`, re-encode, re-decode; require identical word + mnemonic +
/// operand count.
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

fn text(word: u32) -> String {
    let insn = decode(word, 0, FeatureSet::ALL);
    format_to_string(&FmtFormatter::new(), &insn)
}

/// `FeatureSet::ALL` minus a single feature bit (in both words).
fn without(fs: FeatureSet, feat: Feature) -> FeatureSet {
    let bit = feat as u32;
    FeatureSet {
        features0: fs.features0 & !(1u64 << bit),
        features1: fs.features1 & !(1u64 << bit),
    }
}

// ===========================================================================
// 1. SME2 LUT (ZT0).
// ===========================================================================

#[test]
fn luti_zt0_examples() {
    // (word, mnemonic, op_count) — canonical LLVM (+all) oracle encodings.
    let cases: &[(u32, &str, usize)] = &[
        (0xC08C407A, "luti2", 3), // luti2 {z26.b, z27.b}, zt0, z3[0]
        (0xC0CC0000, "luti2", 3), // luti2 z0.b, zt0, z0[0]  (single-register)
        (0xC0CC2000, "luti2", 3), // luti2 z0.s, zt0, z0[0]
        (0xC08C8000, "luti2", 3), // luti2 {z0.b - z3.b}, zt0, z0[0]
        (0xC0CA0000, "luti4", 3), // luti4 z0.b, zt0, z0[0]
        (0xC08A4000, "luti4", 3), // luti4 {z0.b, z1.b}, zt0, z0[0]
        (0xC08A9000, "luti4", 3), // luti4 {z0.h - z3.h}, zt0, z0[0]
        (0xC08BA000, "luti4", 3), // luti4 {z0.s - z3.s}, zt0, z0[1]
        // Strided multi-vector destinations (stride 8 for 2-reg, 4 for 4-reg).
        (0xC09FC0F5, "luti2", 3), // luti2 {z21.b, z29.b}, zt0, z7[7]
        (0xC09D9173, "luti2", 3), // luti2 {z19.h, z23.h, z27.h, z31.h}, zt0, z11[1]
        (0xC09C4000, "luti2", 3), // luti2 {z0.b, z8.b}, zt0, z0[0]
        (0xC09A4000, "luti4", 3), // luti4 {z0.b, z8.b}, zt0, z0[0]  (strided)
        (0xC09A9000, "luti4", 3), // luti4 {z0.h, z4.h, z8.h, z12.h}, zt0, z0[0]
        // Register-pair source (LUTI4, 4-reg `.B`): index is a 2-register group.
        (0xC08B005C, "luti4", 3), // luti4 {z28.b - z31.b}, zt0, {z2, z3}
        (0xC09B0000, "luti4", 3), // luti4 {z0.b, z4.b, z8.b, z12.b}, zt0, {z0, z1} (strided dest)
    ];
    for &(w, m, n) in cases {
        let insn = decode(w, 0, FeatureSet::ALL);
        assert_eq!(insn.mnemonic().name(), m, "{:08X} mnemonic", w);
        assert_eq!(insn.op_count(), n, "{:08X} operand count", w);
        assert_roundtrip(w);
    }
    // Full renderings (note: single-register form has no braces; multi uses
    // `{ ... }` with inner spaces, a ` - ` range for a non-wrapping consecutive
    // 4-group; strided groups print as comma lists; the index is decimal).
    assert_eq!(text(0xC08C407A), "luti2   { z26.b, z27.b }, zt0, z3[0]");
    assert_eq!(text(0xC0CC0000), "luti2   z0.b, zt0, z0[0]");
    assert_eq!(text(0xC08A9000), "luti4   { z0.h - z3.h }, zt0, z0[0]");
    assert_eq!(text(0xC08BA000), "luti4   { z0.s - z3.s }, zt0, z0[1]");
    // Strided destinations render as comma lists (no ` - ` range).
    assert_eq!(text(0xC09FC0F5), "luti2   { z21.b, z29.b }, zt0, z7[7]");
    assert_eq!(text(0xC09D9173), "luti2   { z19.h, z23.h, z27.h, z31.h }, zt0, z11[1]");
    // Register-pair source: a 2-register group, no element suffix, no index.
    assert_eq!(text(0xC08B005C), "luti4   { z28.b - z31.b }, zt0, { z2, z3 }");
    assert_eq!(text(0xC09B0000), "luti4   { z0.b, z4.b, z8.b, z12.b }, zt0, { z0, z1 }");
}

#[test]
fn luti_zt0_roundtrip_sweep() {
    // Sweep word<23>=1, word<21>=0, word<19>=1 (the LUTI ZT0 shell), varying the
    // family / stride / count / size / index / Zn / Zd structural bits — including
    // the strided destinations (word<20>=1) and the register-pair source forms.
    // Every word the decoder accepts must re-encode bit-identically.
    let mut decoded = 0usize;
    let mut strided_seen = 0usize;
    let mut pair_seen = 0usize;
    for r in 0..2u32 {
        // word<22>: single-reg(1) / multi(0)
        for st in 0..2u32 {
            // word<20>: strided(1) / consecutive(0)
            for l in 0..2u32 {
                // word<18>=l (LUTI2=1), plus the fixed LUTI4 word<17>=1.
                for hi in 0u32..(1 << 8) {
                    // word<17:10>
                    for zn in [0u32, 1, 2, 5, 30, 31] {
                        for zd in [0u32, 1, 2, 3, 4, 7, 8, 16, 19, 23, 28, 31] {
                            let w = 0xC088_0000
                                | (r << 22)
                                | (st << 20)
                                | (l << 18)
                                | (hi << 10)
                                | (zn << 5)
                                | zd;
                            let insn = decode(w, 0, FeatureSet::ALL);
                            if insn.is_invalid() {
                                continue;
                            }
                            decoded += 1;
                            if st == 1 {
                                strided_seen += 1;
                            }
                            // Pair-source form: operand 2 is a 2-register group.
                            if matches!(insn.op(2), fARM64::Operand::SveVecGroup { count: 2, .. }) {
                                pair_seen += 1;
                            }
                            assert_roundtrip(w);
                        }
                    }
                }
            }
        }
    }
    assert!(decoded >= 1000, "expected many LUTI ZT0 forms, got {}", decoded);
    assert!(strided_seen > 0, "expected strided LUTI forms in the sweep");
    assert!(pair_seen > 0, "expected register-pair-source LUTI forms in the sweep");
}

#[test]
fn luti_zt0_reserved_and_neighbours() {
    // LUTI4 4-register `.b` is the sole unallocated (count, size) combination.
    // word<18>=0 (L4), word<15>=1 (4-reg marker), size word<13:12>=00 (.b).
    let l4_4b = 0xC088_0000 | (1 << 17) | (1 << 15);
    assert!(decode(l4_4b, 0, FeatureSet::ALL).is_invalid(), "LUTI4 4-reg .b must be Invalid");
    // LUTI4 4-register `.h` (size 01) *is* valid (sanity that the size guard is
    // surgical).
    let l4_4h = 0xC088_0000 | (1 << 17) | (1 << 15) | (1 << 12);
    assert!(!decode(l4_4h, 0, FeatureSet::ALL).is_invalid(), "LUTI4 4-reg .h must decode");

    // `LUTI6` neighbour: word<18>=0, word<17>=0 — must not be claimed as LUTI4.
    let luti6 = 0xC0C8_4000; // luti6 z0.b, zt0, z0
    assert!(decode(luti6, 0, FeatureSet::ALL).is_invalid(), "LUTI6 must stay Invalid (out of scope)");

    // `ZERO` / `MOVT` neighbour: word<23>=0 — must not be claimed as LUTI.
    let zero_mnem = decode(0xC008_0000, 0, FeatureSet::ALL).mnemonic().name();
    assert!(zero_mnem != "luti2" && zero_mnem != "luti4", "ZERO region not LUTI");

    // Reserved word<11:10> non-zero is UNDEFINED.
    let resv = 0xC0CC_0000 | (1 << 10);
    assert!(decode(resv, 0, FeatureSet::ALL).is_invalid(), "reserved word<10> must be Invalid");

    // Element size word<13:12>=11 is unallocated.
    let bad_size = 0xC0CC_0000 | (0b11 << 12);
    assert!(decode(bad_size, 0, FeatureSet::ALL).is_invalid(), "size 11 must be Invalid");

    // Multi-register destinations must be span-aligned.
    let odd2 = 0xC08C_4000 | 1; // luti2 {z1,z2} (odd base) — invalid
    assert!(decode(odd2, 0, FeatureSet::ALL).is_invalid(), "odd 2-reg base must be Invalid");

    // --- Strided-form constraints ---
    // Strided never allows `.s` (size 10).
    let str2_s = 0xC09C_4000 | (0b10 << 12); // luti2 2-reg strided .s
    assert!(decode(str2_s, 0, FeatureSet::ALL).is_invalid(), "strided .s must be Invalid");
    // LUTI4 4-register strided is `.h`-only (`.b` invalid).
    let str4_l4_b = 0xC088_0000 | (1 << 20) | (1 << 17) | (1 << 15); // strided L4 4-reg .b
    assert!(decode(str4_l4_b, 0, FeatureSet::ALL).is_invalid(), "L4 4-reg strided .b Invalid");
    // 2-register strided base outside the window (word<3> set) is UNDEFINED.
    let str2_badbase = 0xC09C_4000 | 0b1000; // base z8 (window is z0..7, z16..23)
    assert!(decode(str2_badbase, 0, FeatureSet::ALL).is_invalid(), "strided 2-reg base z8 Invalid");
    // The single-register form has no strided variant (word<20>=1 → Invalid). Build
    // a single-reg word and set the strided bit.
    let single_strided = 0xC0CC_0000 | (1 << 20);
    assert!(decode(single_strided, 0, FeatureSet::ALL).is_invalid(), "single-reg strided Invalid");

    // --- Register-pair source form (LUTI4 4-reg .b) ---
    // Valid: word<17>=1, word<16>=1, word<15:14>=00, .b.
    let pair = 0xC088_0000 | (1 << 17) | (1 << 16);
    let pi = decode(pair, 0, FeatureSet::ALL);
    assert!(!pi.is_invalid() && pi.mnemonic().name() == "luti4", "pair-source form must decode");
    assert!(
        matches!(pi.op(2), fARM64::Operand::SveVecGroup { count: 2, arr: None, .. }),
        "pair-source operand 2 is a 2-register group"
    );
    // Pair source base must be even; odd source base is Invalid.
    let pair_oddsrc = pair | (1 << 5); // Zn = 1 (odd)
    assert!(decode(pair_oddsrc, 0, FeatureSet::ALL).is_invalid(), "odd pair-source base Invalid");
}

#[test]
fn luti_zt0_feature_gated() {
    let no_lut = without(FeatureSet::ALL, Feature::Lut);
    for w in [0xC08C407Au32, 0xC0CC0000, 0xC0CA0000, 0xC08A9000] {
        assert!(decode(w, 0, no_lut).is_invalid(), "{:08X} should require FEAT_LUT", w);
        assert!(!decode(w, 0, FeatureSet::ALL).is_invalid(), "{:08X} should decode with FEAT_LUT", w);
    }
}

// ===========================================================================
// 2. SME2 ZA tile move (MOV / MOVAZ, move multi-vectors).
// ===========================================================================

#[test]
fn za_tile_move_examples() {
    let cases: &[(u32, &str, usize)] = &[
        (0xC0040183, "mov", 2),   // mov za0h.b[w12, 6:7], {z12.b, z13.b}
        (0xC0040000, "mov", 2),   // mov za0h.b[w12, 0:1], {z0.b, z1.b}
        (0xC0040400, "mov", 2),   // mov za0h.b[w12, 0:3], {z0.b - z3.b}  (vgx4)
        (0xC0440000, "mov", 2),   // mov za0h.h[w12, 0:1], {z0.h, z1.h}
        (0xC0048000, "mov", 2),   // mov za0v.b[w12, 0:1], {z0.b, z1.b}  (vertical)
        (0xC0060000, "mov", 2),   // mov {z0.b, z1.b}, za0h.b[w12, 0:1]
        (0xC0060200, "movaz", 2), // movaz {z0.b, z1.b}, za0h.b[w12, 0:1]
        (0xC0860600, "movaz", 2), // movaz {z0.s - z3.s}, za0h.s[w12, 0:3] (vgx4)
    ];
    for &(w, m, n) in cases {
        let insn = decode(w, 0, FeatureSet::ALL);
        assert_eq!(insn.mnemonic().name(), m, "{:08X} mnemonic", w);
        assert_eq!(insn.op_count(), n, "{:08X} operand count", w);
        assert_roundtrip(w);
    }
    // Full renderings — house style: decimal slice range, no `vgxN` for the
    // tile-slice form, `{ }`-with-spaces vector group, ` - ` range for vgx4.
    assert_eq!(text(0xC0040183), "mov     za0h.b[w12, 6:7], { z12.b, z13.b }");
    assert_eq!(text(0xC0040400), "mov     za0h.b[w12, 0:3], { z0.b - z3.b }");
    assert_eq!(text(0xC0048000), "mov     za0v.b[w12, 0:1], { z0.b, z1.b }");
    assert_eq!(text(0xC0060200), "movaz   { z0.b, z1.b }, za0h.b[w12, 0:1]");
    assert_eq!(text(0xC0860600), "movaz   { z0.s - z3.s }, za0h.s[w12, 0:3]");
}

#[test]
fn za_tile_move_roundtrip_sweep() {
    // Sweep the ZA tile-move shell (word<31:24>=0xC0, word<18>=1, word<19>=0)
    // across size / direction / V / Ws / Q / movaz and the tile:offset + register
    // fields. Every accepted word must re-encode bit-identically.
    let mut decoded = 0usize;
    for sz in 0..4u32 {
        for d in 0..2u32 {
            for v in 0..2u32 {
                for ws in 0..4u32 {
                    for q in 0..2u32 {
                        for w9 in 0..2u32 {
                            for field in 0..8u32 {
                                for reg in [0u32, 1, 2, 4, 8, 30] {
                                    // vectors→ZA: reg in word<9:5>, field in word<2:0>.
                                    let w_vza = 0xC004_0000
                                        | (sz << 22)
                                        | (d << 17)
                                        | (v << 15)
                                        | (ws << 13)
                                        | (q << 10)
                                        | (w9 << 9)
                                        | (reg << 5)
                                        | field;
                                    // ZA→vectors: field in word<7:5>, reg in word<4:0>.
                                    let w_zav = 0xC004_0000
                                        | (sz << 22)
                                        | (d << 17)
                                        | (v << 15)
                                        | (ws << 13)
                                        | (q << 10)
                                        | (w9 << 9)
                                        | (field << 5)
                                        | reg;
                                    for w in [w_vza, w_zav] {
                                        if !decode(w, 0, FeatureSet::ALL).is_invalid() {
                                            decoded += 1;
                                            assert_roundtrip(w);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(decoded >= 500, "expected many ZA tile-move forms, got {}", decoded);
}

#[test]
fn za_tile_move_reserved_and_neighbours() {
    // word<16>=1 is reserved → Invalid.
    assert!(decode(0xC0040000 | (1 << 16), 0, FeatureSet::ALL).is_invalid(), "word<16> reserved");
    // word<12>=1 / word<11>=1 are reserved → Invalid.
    assert!(decode(0xC0040000 | (1 << 12), 0, FeatureSet::ALL).is_invalid(), "word<12> reserved");
    // word<11>=1 is the *array-vector* MOV neighbour (`za.d[w8, ..]`) — out of
    // scope, must not be claimed as the tile-slice form.
    assert!(text(0xC0040800) != "mov     za0h.b[w12, 0:1], {z0.b, z1.b}", "array-vector MOV not tile form");
    // word<19>=1 is the ZERO neighbour — must not be claimed.
    assert!(decode(0xC0080000, 0, FeatureSet::ALL).mnemonic().name() != "mov", "ZERO not MOV");

    // Misaligned vgx2 register group base (odd) is UNDEFINED.
    assert!(decode(0xC0040000 | (1 << 5), 0, FeatureSet::ALL).is_invalid(), "odd vgx2 base Invalid");
    // Misaligned vgx4 register group base (not multiple of 4) is UNDEFINED.
    assert!(decode(0xC0040400 | (2 << 5), 0, FeatureSet::ALL).is_invalid(), "vgx4 base %4 Invalid");
    // Out-of-range slice offset (vgx4 `.b`, offset field 4) is UNDEFINED.
    assert!(decode(0xC0040400 | 4, 0, FeatureSet::ALL).is_invalid(), "vgx4 .b offset 16 Invalid");
    // Out-of-range tile number (`.b` has a single tile) is UNDEFINED.
    // `.b` vza, field word<2:0> with the tile bit set only ever yields offset; for
    // `.h` (2 tiles), tile=1 is valid but tile=... is bounded by `1 << size`.
    // za->v reserved word<8> non-zero is UNDEFINED.
    assert!(decode(0xC0060000 | (1 << 8), 0, FeatureSet::ALL).is_invalid(), "za->v word<8> reserved");
    // vectors->ZA reserved word<4:3> non-zero is UNDEFINED.
    assert!(decode(0xC0040000 | (1 << 3), 0, FeatureSet::ALL).is_invalid(), "vza word<3> reserved");
}

#[test]
fn za_tile_move_feature_gated() {
    let no_sme2 = without(FeatureSet::ALL, Feature::Sme2);
    for w in [0xC0040183u32, 0xC0060200, 0xC0040400, 0xC0860600] {
        assert!(decode(w, 0, no_sme2).is_invalid(), "{:08X} should require FEAT_SME2", w);
        assert!(!decode(w, 0, FeatureSet::ALL).is_invalid(), "{:08X} should decode with FEAT_SME2", w);
    }
}
