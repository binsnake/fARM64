//! FEAT_LSFE atomic floating-point in-memory ops: decode + round-trip coverage.
//!
//! Covers the atomic-float read-modify-write family `LDF{ADD,MAX,MIN,MAXNM,MINNM}`
//! / `STF*` (H/S/D data) and the BFloat16 `LDBF*`/`STBF*` (data size `00`). They
//! are the `V==1` sibling of the integer LSE atomic major and share its layout
//! `size 111 1 00 A R 1 Rs o3 opc 00 Rn Rt`: `opc` selects the op, `A:R` the
//! ordering, `o3` load(0)/store(1). The store form requires `Rt==31` and `A==0`.
//!
//! The canonical example words are LLVM (`clang`/`llvm-objdump --mattr=+all`)
//! oracle encodings. The tests confirm fARM64 decodes them to the expected
//! mnemonic/operand-count, re-encodes to the identical word, sweeps the whole
//! sub-space for semantic round-trip stability, and that the reserved/undefined
//! slots stay `Invalid`.

#![cfg(feature = "std")]

use fARM64::decode::decode;
use fARM64::{encode, Feature, FeatureSet};

/// Build an LSFE word: `size 111 1 00 A R 1 Rs o3 opc 00 Rn Rt`.
#[allow(clippy::too_many_arguments)]
fn mk(size: u32, a: u32, r: u32, rs: u32, o3: u32, opc: u32, rn: u32, rt: u32) -> u32 {
    (size << 30)
        | (0b111 << 27)
        | (1 << 26)
        | (a << 23)
        | (r << 22)
        | (1 << 21)
        | (rs << 16)
        | (o3 << 15)
        | (opc << 12)
        | (rn << 5)
        | rt
}

/// Decode `word`, re-encode, re-decode; require identical mnemonic + operands.
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
    for i in 0..insn.op_count() {
        assert_eq!(
            format!("{:?}", insn.op(i)),
            format!("{:?}", insn2.op(i)),
            "{:08X} ({}) operand {} differs",
            word,
            insn.mnemonic().name(),
            i
        );
    }
}

#[test]
fn examples_decode_as_expected() {
    // (word, mnemonic, operand_count) — canonical LLVM (+all) oracle encodings.
    let cases: &[(u32, &str, usize)] = &[
        // LDF<op> (load form: Rs, Rt, [Xn]) — H/S/D + ordering.
        (0x7C20039A, "ldfadd", 3),    // ldfadd   h0, h26, [x28]
        (0xBCA10062, "ldfadda", 3),   // ldfadda  s1, s2,  [x3]
        (0xFC6403E5, "ldfaddl", 3),   // ldfaddl  d4, d5,  [sp]
        (0x7CE60107, "ldfaddal", 3),  // ldfaddal h6, h7,  [x8]
        (0xBC204041, "ldfmax", 3),    // ldfmax   s0, s1,  [x2]
        (0xBC205041, "ldfmin", 3),    // ldfmin   s0, s1,  [x2]
        (0xBC206041, "ldfmaxnm", 3),  // ldfmaxnm s0, s1,  [x2]
        (0x7C207041, "ldfminnm", 3),  // ldfminnm h0, h1,  [x2]
        // STF<op> (store form: Rs, [Xn]) — Rt==31, ordering none/L.
        (0xBC20803F, "stfadd", 2),    // stfadd     s0, [x1]
        (0xFC60803F, "stfaddl", 2),   // stfaddl    d0, [x1]
        (0x7C20C03F, "stfmax", 2),    // stfmax     h0, [x1]
        (0xBC60E03F, "stfmaxnml", 2), // stfmaxnml  s0, [x1]
        (0xFC20D03F, "stfmin", 2),    // stfmin     d0, [x1]
        (0x7C20F03F, "stfminnm", 2),  // stfminnm   h0, [x1]
        // BF16 LDBF<op> / STBF<op> (data size 00, H register view).
        (0x3C200041, "ldbfadd", 3),   // ldbfadd   h0, h1, [x2]
        (0x3C204041, "ldbfmax", 3),   // ldbfmax   h0, h1, [x2]
        (0x3C207041, "ldbfminnm", 3), // ldbfminnm h0, h1, [x2]
        (0x3C20803F, "stbfadd", 2),   // stbfadd   h0, [x1]
        (0x3C60803F, "stbfaddl", 2),  // stbfaddl  h0, [x1]
    ];
    for &(w, m, n) in cases {
        let insn = decode(w, 0, FeatureSet::ALL);
        assert_eq!(insn.mnemonic().name(), m, "{:08X} mnemonic", w);
        assert_eq!(insn.op_count(), n, "{:08X} operand count", w);
        assert_roundtrip(w);
    }
}

#[test]
fn exhaustive_roundtrip() {
    // (rs, rn, rt) register tuples to exercise.
    let regs = [(0u32, 0u32, 0u32), (1, 2, 3), (5, 31, 9), (31, 1, 7), (26, 28, 0)];
    let mut decoded = 0usize;
    for size in 0..4 {
        for a in 0..2 {
            for r in 0..2 {
                for o3 in 0..2 {
                    for opc in [0b000u32, 0b100, 0b101, 0b110, 0b111] {
                        for &(rs, rn, rt) in &regs {
                            // Store form forces Rt=31, A=0 in the architecture.
                            let (a_eff, rt_eff) = if o3 == 1 { (0, 0b11111) } else { (a, rt) };
                            let w = mk(size, a_eff, r, rs, o3, opc, rn, rt_eff);
                            if decode(w, 0, FeatureSet::ALL).is_invalid() {
                                continue;
                            }
                            decoded += 1;
                            assert_roundtrip(w);
                        }
                    }
                }
            }
        }
    }
    assert!(decoded >= 200, "expected many LSFE forms to decode, got {}", decoded);
}

#[test]
fn reserved_and_store_constraints() {
    // opc 001/010/011 are unallocated for any o3/size.
    for size in 0..4 {
        for o3 in 0..2 {
            for opc in [0b001u32, 0b010, 0b011] {
                let w = mk(size, 0, 0, 1, o3, opc, 2, if o3 == 1 { 31 } else { 3 });
                assert!(decode(w, 0, FeatureSet::ALL).is_invalid(), "{:08X} opc={:03b} should be Invalid", w, opc);
            }
        }
    }
    // Store form (o3=1) with Rt != 31 is UNDEFINED.
    let st_badrt = mk(2, 0, 0, 1, 1, 0b000, 2, 5);
    assert!(decode(st_badrt, 0, FeatureSet::ALL).is_invalid(), "{:08X} store Rt!=31 should be Invalid", st_badrt);
    // Store form (o3=1) with A=1 is UNDEFINED.
    let st_acquire = mk(2, 1, 0, 1, 1, 0b000, 2, 31);
    assert!(decode(st_acquire, 0, FeatureSet::ALL).is_invalid(), "{:08X} store A=1 should be Invalid", st_acquire);
    // The load form with the same fields *does* decode (sanity: layout is right).
    let ld = mk(2, 1, 0, 1, 0, 0b000, 2, 5);
    assert!(!decode(ld, 0, FeatureSet::ALL).is_invalid(), "{:08X} load form should decode", ld);
}

#[test]
fn gated_by_feature() {
    let no_lsfe = without_lsfe(FeatureSet::ALL);
    let words = [0x7C20039Au32, 0xBC20803F, 0x3C200041, 0x3C20803F];
    for w in words {
        assert!(decode(w, 0, no_lsfe).is_invalid(), "{:08X} should require FEAT_LSFE", w);
        assert!(!decode(w, 0, FeatureSet::ALL).is_invalid(), "{:08X} should decode with FEAT_LSFE", w);
    }
}

/// `FeatureSet::ALL` minus the `Lsfe` bit (in both words).
fn without_lsfe(fs: FeatureSet) -> FeatureSet {
    let bit = Feature::Lsfe as u32;
    FeatureSet {
        features0: fs.features0 & !(1u64 << bit),
        features1: fs.features1 & !(1u64 << bit),
    }
}
