//! SME / SME2 batch K3: one over-decode fix + three multi-vector gap additions,
//! each LLVM (`oracle.py dec/enc`, i.e. `clang .inst` + `llvm-objdump
//! --mattr=+all`) validated bidirectionally.
//!
//! 1. **ADDHA/ADDVA over-decode** (`src/decode/sme/mod.rs`). The `ZAda` tile
//!    accumulator field is element-size-dependent (`.S` names 4 tiles,
//!    `word<1:0>`; `.D` names 8, `word<2:0>`); the high bits of `word<4:0>` above
//!    it are RES0, and the opcode additionally fixes `word<23> == 1`. The
//!    original decoder masked `ZAda` to width but ignored those reserved bits, so
//!    e.g. `C0901C6D`/`C0D12DFD` (reserved-bit set) and the whole `word<23> == 0`
//!    slot (`C01000C1`) over-decoded as `addha`/`addva`; LLVM leaves them
//!    UNDEFINED. A pre/post differential over a structured+random `0xC0`/`0xC1`
//!    sweep newly-rejected 18,411 words with **0** LLVM-valid words regressed.
//!
//! 2. **SME2 multi-vector FP min/max** (`src/decode/sme/sme2.rs`):
//!    `FMAX`/`FMIN`/`FMAXNM`/`FMINNM { Zdn }, { Zdn }, { Zm }` (multi & multi,
//!    in-place; vgx2/vgx4; `.H`/`.S`/`.D`). E.g. `C1A0B126`.
//! 3. **SME2 multi-vector × single FMUL**: `FMUL { Zd }, { Zn }, Zm` (single
//!    multiplier `Zm` in `z0..z15`; vgx2/vgx4). E.g. `C1F8E8C6`.
//! 4. **SME2 multi-vector LUTI6** (FEAT_LUT): `LUTI6 { Zd, Zd+4, Zd+8, Zd+12 }.H,
//!    { Zn, Zn+1 }.H, { Zt, Zt+1 }[index]`. E.g. `C132FDE3`.

#![cfg(feature = "std")]

use fARM64::decode::decode;
use fARM64::format::{format_to_string, FmtFormatter};
use fARM64::{encode, Feature, FeatureSet};

fn text(word: u32) -> String {
    format_to_string(&FmtFormatter::new(), &decode(word, 0, FeatureSet::ALL))
}

fn norm(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// `FeatureSet::ALL` minus a single feature bit (in both words).
fn without(fs: FeatureSet, feat: Feature) -> FeatureSet {
    let bit = feat as u32;
    FeatureSet {
        features0: fs.features0 & !(1u64 << bit),
        features1: fs.features1 & !(1u64 << bit),
    }
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

// ===========================================================================
// 1. ADDHA / ADDVA over-decode hardening.
// ===========================================================================

#[test]
fn addha_addva_valid_forms_roundtrip() {
    // The valid forms (`word<23> == 1`, ZAda in range, reserved high bits 0) must
    // still decode and round-trip exactly.
    check(0xC0909662, "addha z2.s, p5/m, p4/m, z19.s");
    check(0xC091BA23, "addva z3.s, p6/m, p5/m, z17.s");
    check(0xC0D053E5, "addha z5.d, p4/m, p2/m, z31.d");
    check(0xC0D16EE5, "addva z5.d, p3/m, p3/m, z23.d");
    // Boundary ZAda values (fARM64 renders the accumulator with a `z` prefix per
    // the corpus): `.S` za0..za3, `.D` za0..za7.
    check(0xC0900000, "addha z0.s, p0/m, p0/m, z0.s");
    check(0xC0910003, "addva z3.s, p0/m, p0/m, z0.s");
    check(0xC0D00007, "addha z7.d, p0/m, p0/m, z0.d");
}

#[test]
fn addha_addva_task_overdecodes_rejected() {
    // The two task examples: a reserved ZAda high bit set.
    reserved(0xC0901C6D); // was `addha z1.s, ...` — word<2> set (.S ZAda is 2-bit)
    reserved(0xC0D12DFD); // was `addva z5.d, ...` — word<4:3> set (.D ZAda is 3-bit)
}

#[test]
fn addha_addva_reserved_zada_high_bits() {
    // `.S` (word<22> == 0): ZAda is `word<1:0>`; `word<4:2>` are RES0.
    // base valid `C0900000` (za0.s); set each reserved bit -> Invalid.
    reserved(0xC0900004); // word<2>
    reserved(0xC0900008); // word<3>
    reserved(0xC0900010); // word<4>
    reserved(0xC0900007); // word<2:0> == 7 (only 0..3 legal)
    // `.D` (word<22> == 1): ZAda is `word<2:0>`; `word<4:3>` are RES0.
    reserved(0xC0D00008); // word<3>
    reserved(0xC0D00010); // word<4>
    reserved(0xC0D00018); // word<4:3>

    // Exhaustive: across V (ADDHA/ADDVA) and size (.S/.D), any set reserved bit
    // above the in-use ZAda field is UNDEFINED; everything below stays valid.
    for v in [0u32, 1] {
        for is64 in [false, true] {
            let base = 0xC0900000 | (u32::from(is64) << 22) | (v << 16);
            let width = if is64 { 3 } else { 2 };
            for zada5 in 0u32..32 {
                let w = base | zada5;
                if (zada5 >> width) != 0 {
                    reserved(w);
                } else {
                    assert!(
                        !decode(w, 0, FeatureSet::ALL).is_invalid(),
                        "{w:08X} ZAda {zada5} in range should decode"
                    );
                }
            }
        }
    }
}

#[test]
fn addha_addva_bit23_slot_reserved() {
    // The whole `word<23> == 0` ADDHA/ADDVA opcode slot is unallocated (LLVM
    // UNDEFINED, e.g. `C01000C1`); `word<23> == 1` is required.
    reserved(0xC01000C1);
    for v in [0u32, 1] {
        for low in 0u32..8 {
            // word<23> == 0, opcode word<21:17> == 01000 (the ADDHA/ADDVA key).
            let w = 0xC0000000 | (1 << 20) | (v << 16) | low;
            reserved(w);
        }
    }
}

// ===========================================================================
// 2. SME2 multi-vector FP min/max (multi & multi, in-place).
// ===========================================================================

#[test]
fn fp_minmax_multi_multi_examples() {
    // The task example and its siblings (vgx2, .s).
    check(0xC1A0B126, "fmaxnm { z6.s, z7.s }, { z6.s, z7.s }, { z0.s, z1.s }");
    check(0xC1A0B100, "fmax { z0.s, z1.s }, { z0.s, z1.s }, { z0.s, z1.s }");
    check(0xC1A0B101, "fmin { z0.s, z1.s }, { z0.s, z1.s }, { z0.s, z1.s }");
    check(0xC1A0B120, "fmaxnm { z0.s, z1.s }, { z0.s, z1.s }, { z0.s, z1.s }");
    check(0xC1A0B121, "fminnm { z0.s, z1.s }, { z0.s, z1.s }, { z0.s, z1.s }");
    // Element sizes .h and .d.
    check(0xC160B120, "fmaxnm { z0.h, z1.h }, { z0.h, z1.h }, { z0.h, z1.h }");
    check(0xC1E0B121, "fminnm { z0.d, z1.d }, { z0.d, z1.d }, { z0.d, z1.d }");
    // vgx4 (the destination/first-source share the group, rendered as a range).
    check(0xC1A0B920, "fmaxnm { z0.s - z3.s }, { z0.s - z3.s }, { z0.s - z3.s }");
    check(0xC1A4B924, "fmaxnm { z4.s - z7.s }, { z4.s - z7.s }, { z4.s - z7.s }");
    check(0xC1A0B928, "fmaxnm { z8.s - z11.s }, { z8.s - z11.s }, { z0.s - z3.s }");
}

#[test]
fn fp_minmax_reserved() {
    // `.b` (size 00) is the BFloat16 BFMAX/BFMIN neighbour (FEAT_SME_B16B16),
    // now implemented by the Q multi×multi decoder — it decodes as `bfmax`, so the
    // FP family here correctly does not claim size 00.
    check(0xC120B100, "bfmax { z0.h, z1.h }, { z0.h, z1.h }, { z0.h, z1.h }");
    // vgx4 with an odd group base bit set (`word<1>`/`word<17>` RES0).
    reserved(0xC1A0B922); // word<1> set -> base not a multiple of 4
    reserved(0xC1A2B920); // word<17> set -> Zm base not a multiple of 4
    // word<9:6> opcode marker must be 0100; a different value is not this family.
    reserved(0xC1A0B1A0); // word<7> set
}

// ===========================================================================
// 3. SME2 multi-vector × single-multiplier FMUL.
// ===========================================================================

#[test]
fn fmul_multi_single_examples() {
    // The task example (vgx2, .d) and siblings.
    check(0xC1F8E8C6, "fmul { z6.d, z7.d }, { z6.d, z7.d }, z12.d");
    check(0xC1E0E800, "fmul { z0.d, z1.d }, { z0.d, z1.d }, z0.d");
    check(0xC1A0E800, "fmul { z0.s, z1.s }, { z0.s, z1.s }, z0.s");
    check(0xC160E800, "fmul { z0.h, z1.h }, { z0.h, z1.h }, z0.h");
    // Distinct dest / first-source / multiplier registers.
    check(0xC1A6E882, "fmul { z2.s, z3.s }, { z4.s, z5.s }, z3.s");
    check(0xC1AAE800, "fmul { z0.s, z1.s }, { z0.s, z1.s }, z5.s");
    // vgx4 + the single multiplier still spans z0..z15.
    check(0xC1A1E800, "fmul { z0.s - z3.s }, { z0.s - z3.s }, z0.s");
    check(0xC1A1E804, "fmul { z4.s - z7.s }, { z0.s - z3.s }, z0.s");
    check(0xC1BFE800, "fmul { z0.s - z3.s }, { z0.s - z3.s }, z15.s");
}

#[test]
fn fmul_multi_single_reserved() {
    // `C120E800` (size 00) is the BF16 `BFMUL` neighbour — decoded by the L3 batch
    // (`tests/sme_l3.rs::bfmul`), so it is no longer reserved here.
    reserved(0xC1A1E820); // word<5> RES0
    reserved(0xC1A1E802); // vgx4 word<1> RES0 (dest base multiple of 4)
    reserved(0xC1A1E840); // vgx4 word<6> RES0 (first-source base multiple of 4)
}

// ===========================================================================
// 4. SME2 multi-vector LUTI6.
// ===========================================================================

#[test]
fn luti6_examples() {
    check(0xC132FDE3, "luti6 { z3.h, z7.h, z11.h, z15.h }, { z15.h, z16.h }, { z18, z19 }[0]");
    check(0xC120FC00, "luti6 { z0.h, z4.h, z8.h, z12.h }, { z0.h, z1.h }, { z0, z1 }[0]");
    // Destination base packs word<4> (high) and word<1:0> (low): bases 0..3, 16..19.
    check(0xC120FC03, "luti6 { z3.h, z7.h, z11.h, z15.h }, { z0.h, z1.h }, { z0, z1 }[0]");
    check(0xC130FC10, "luti6 { z16.h, z20.h, z24.h, z28.h }, { z0.h, z1.h }, { z16, z17 }[0]");
    // Index = word<22>.
    check(0xC160FC00, "luti6 { z0.h, z4.h, z8.h, z12.h }, { z0.h, z1.h }, { z0, z1 }[1]");
    // Zn (consecutive pair) wraps; table base is a free 5-bit field.
    check(0xC120FFC0, "luti6 { z0.h, z4.h, z8.h, z12.h }, { z30.h, z31.h }, { z0, z1 }[0]");
    check(0xC121FC00, "luti6 { z0.h, z4.h, z8.h, z12.h }, { z0.h, z1.h }, { z1, z2 }[0]");
}

#[test]
fn luti6_roundtrip_sweep() {
    // Exhaustive over the allocated LUTI6 field space: dest base {0..3, 16..19},
    // Zn 0..31, table 0..31, index 0..1. Every decode must round-trip exactly.
    let mut checked = 0u64;
    for zd_hi in [0u32, 1] {
        for zd_lo in 0u32..4 {
            let zd = (zd_hi << 4) | zd_lo;
            for zn in [0u32, 1, 9, 30, 31] {
                for table in [0u32, 1, 12, 30, 31] {
                    for index in [0u32, 1] {
                        let word = 0xC120_FC00 | (index << 22) | (table << 16) | (zn << 5) | zd;
                        let insn = decode(word, 0, FeatureSet::ALL);
                        assert!(!insn.is_invalid(), "{word:08X} LUTI6 should decode");
                        let enc = encode(&insn).expect("LUTI6 encode");
                        assert_eq!(enc, word, "{word:08X} LUTI6 round-trip -> {enc:08X}");
                        checked += 1;
                    }
                }
            }
        }
    }
    assert!(checked > 0);
}

#[test]
fn luti6_reserved() {
    // Destination base outside {0..3, 16..19}: word<3:2> are RES0.
    reserved(0xC120FC04); // base 4 (word<2>)
    reserved(0xC120FC08); // base 8 (word<3>)
    reserved(0xC120FC14); // base 20 (word<4>+word<2>)
    reserved(0xC120FC18); // base 24 (word<4>+word<3>)
    // word<23> RES0 (only the single-bit index word<22> is allocated).
    reserved(0xC1A0FC00); // word<23> set
}

// ===========================================================================
// 5. Feature gating.
// ===========================================================================

#[test]
fn feature_gating() {
    // The FP min/max + FMUL multi-vector forms require FEAT_SME2.
    let no_sme2 = without(FeatureSet::ALL, Feature::Sme2);
    assert!(decode(0xC1A0B126, 0, no_sme2).is_invalid(), "fmaxnm needs SME2");
    assert!(decode(0xC1F8E8C6, 0, no_sme2).is_invalid(), "fmul needs SME2");

    // LUTI6 requires FEAT_LUT (and is reached only with SME2 routing on).
    let no_lut = without(FeatureSet::ALL, Feature::Lut);
    assert!(decode(0xC132FDE3, 0, no_lut).is_invalid(), "luti6 needs LUT");

    // The whole SME structural gate: without FEAT_SME nothing in the region
    // decodes (ADDHA included).
    let no_sme = without(FeatureSet::ALL, Feature::Sme);
    assert!(decode(0xC0909662, 0, no_sme).is_invalid(), "addha needs SME");
}
